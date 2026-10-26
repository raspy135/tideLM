//! Checkpoint access, from `engine/experts.py`: safetensors shard headers, the weight
//! index, and O_DIRECT reads into aligned (pinned, in serving) buffers.
//!
//! The layout facts this crate encodes (and asserts in tests against the real
//! checkpoint): a routed expert is six tensors whose shard placement groups them into
//! exactly two contiguous file runs — a ~1.1 MB scale run and a ~17.7 MB weight run —
//! and reading spans in `read_chunk`-sized aligned pieces in parallel is what keeps NVMe
//! at ~5 GB/s when only a few experts (or one dense tensor) are being read.

pub mod direct_io;
pub mod safetensors;

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use direct_io::{AlignedBuf, ALIGN};
pub use safetensors::TensorMeta;
use safetensors::Shard;
#[cfg(test)]
use safetensors::EXPERT_NAMES;

/// Bytes of one FP4 routed expert: 3 weight planes + 3 scale planes.
pub const EXPERT_BYTES: usize = 3 * (2304 * 2560 + 2304 * 160);
/// Staging headroom for run-alignment slack, as in the Python store.
pub const STAGE_BYTES: usize = EXPERT_BYTES + 8 * ALIGN;

pub const W13_SHAPE: (usize, usize) = (2304, 2560);
pub const S13_SHAPE: (usize, usize) = (2304, 160);
pub const W2_SHAPE: (usize, usize) = (5120, 1152);
pub const S2_SHAPE: (usize, usize) = (5120, 72);

/// `model.safetensors.index.json`: tensor name -> shard file name.
pub struct WeightIndex {
    pub model_dir: PathBuf,
    pub weight_map: HashMap<String, String>,
}

impl WeightIndex {
    pub fn load(model_dir: &Path) -> io::Result<Self> {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let raw = std::fs::read(model_dir.join("model.safetensors.index.json"))?;
        let index: Index = serde_json::from_slice(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(WeightIndex { model_dir: model_dir.to_path_buf(), weight_map: index.weight_map })
    }
}

/// Bytes needed to read `[lo, hi)` with O_DIRECT: the span widened to ALIGN boundaries.
pub fn aligned_len(lo: u64, hi: u64) -> usize {
    let alo = lo - lo % ALIGN as u64;
    let ahi = hi.div_ceil(ALIGN as u64) * ALIGN as u64;
    (ahi - alo) as usize
}

/// The whole checkpoint: lazily opened shards (kept open for the process, as the Python
/// store keeps its fds) and span reads from any of them.
pub struct Checkpoint {
    index: WeightIndex,
    shards: Mutex<HashMap<String, &'static Shard>>,
    /// tensors served from an overlay file instead of their shard (`overlay`)
    overlay: HashMap<String, &'static Shard>,
    /// Aligned piece size for parallel reads (the Python DSV41_READ_CHUNK_MB=4).
    pub read_chunk: usize,
}

impl Checkpoint {
    pub fn open(model_dir: &Path) -> io::Result<Self> {
        Ok(Checkpoint {
            index: WeightIndex::load(model_dir)?,
            shards: Mutex::new(HashMap::new()),
            overlay: HashMap::new(),
            read_chunk: 4 * 1024 * 1024 / ALIGN * ALIGN,
        })
    }

    /// Serve every tensor of the safetensors file `path` from it instead of its shard (the
    /// abliteration overlay, tidelm-engine's ablit): each must be a checkpoint tensor of the same
    /// dtype and shape, so an overlay replaces values and never adds, drops or reshapes a tensor.
    /// Every read resolves through `shard`, so every load and reader of this checkpoint sees the
    /// overlay's bytes; the checkpoint's files are never written. Returns the names, sorted.
    pub fn overlay(&mut self, path: &Path) -> io::Result<Vec<String>> {
        let shard: &'static Shard = Box::leak(Box::new(Shard::open(path)?));
        let mut names: Vec<String> = shard.names().map(str::to_string).collect();
        names.sort();
        for name in &names {
            let (have, want) = (shard.meta(name)?, self.meta(name)?);
            if (&have.dtype, &have.shape) != (&want.dtype, &want.shape) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                    "{name} is {} {:?} in {} but {} {:?} in the checkpoint", have.dtype, have.shape, path.display(),
                    want.dtype, want.shape)));
            }
        }
        for name in &names {
            self.overlay.insert(name.clone(), shard);
        }
        Ok(names)
    }

    pub fn model_dir(&self) -> &Path {
        &self.index.model_dir
    }

    /// Every shard opened now (its header parsed and kept for the process), so no read made
    /// while serving opens one and the cache's memory is the boot's.
    pub fn open_all(&self) -> io::Result<()> {
        let mut one: HashMap<&String, &String> = HashMap::new();
        for (tensor, file) in &self.index.weight_map {
            one.entry(file).or_insert(tensor);
        }
        for tensor in one.values() {
            self.shard(tensor)?;
        }
        Ok(())
    }

    pub fn has(&self, tensor: &str) -> bool {
        self.index.weight_map.contains_key(tensor)
    }

    pub fn shard(&self, tensor: &str) -> io::Result<&'static Shard> {
        if let Some(&s) = self.overlay.get(tensor) {
            return Ok(s);
        }
        let file = self.index.weight_map.get(tensor).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("{tensor} not in the weight index"))
        })?;
        let mut shards = self.shards.lock().unwrap();
        if let Some(s) = shards.get(file) {
            return Ok(s);
        }
        // Leaked on purpose: shards live as long as the process, which gives the run
        // cache a stable address without self-referential lifetimes.
        let shard: &'static Shard =
            Box::leak(Box::new(Shard::open(&self.index.model_dir.join(file))?));
        shards.insert(file.clone(), shard);
        Ok(shard)
    }

    pub fn meta(&self, tensor: &str) -> io::Result<TensorMeta> {
        self.shard(tensor)?.meta(tensor).cloned()
    }

    /// O_DIRECT-read bytes `[lo, hi)` of `tensor` (offsets relative to the tensor's
    /// start) into `buf`, which must be ALIGN-aligned and at least `aligned_len` long.
    /// Returns the offset in `buf` where byte `lo` landed.
    pub fn read_tensor_range(&self, tensor: &str, lo: usize, hi: usize, buf: &mut [u8]) -> io::Result<usize> {
        let shard = self.shard(tensor)?;
        let meta = shard.meta(tensor)?;
        if hi > meta.nbytes() || lo > hi {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{tensor}: range {lo}..{hi} out of bounds")));
        }
        read_span(shard, meta.lo + lo as u64, meta.lo + hi as u64, buf, self.read_chunk)
    }

    /// The whole tensor; see `read_tensor_range`.
    pub fn read_tensor(&self, tensor: &str, buf: &mut [u8]) -> io::Result<(usize, TensorMeta)> {
        let meta = self.meta(tensor)?;
        let off = self.read_tensor_range(tensor, 0, meta.nbytes(), buf)?;
        Ok((off, meta))
    }

    /// Convenience for small tensors and tests: an owned copy of the tensor's bytes.
    pub fn read_tensor_vec(&self, tensor: &str) -> io::Result<(Vec<u8>, TensorMeta)> {
        let meta = self.meta(tensor)?;
        let mut buf = AlignedBuf::new(aligned_len(meta.lo, meta.hi))?;
        let off = self.read_tensor_range(tensor, 0, meta.nbytes(), &mut buf)?;
        Ok((buf[off..off + meta.nbytes()].to_vec(), meta))
    }
}

/// Read absolute file span `[lo, hi)` of `shard` into `buf` with O_DIRECT, in aligned
/// `chunk`-sized pieces issued in parallel. Returns the offset of `lo` in `buf`.
fn read_span(shard: &Shard, lo: u64, hi: u64, buf: &mut [u8], chunk: usize) -> io::Result<usize> {
    let alo = lo - lo % ALIGN as u64;
    let n = aligned_len(lo, hi);
    if buf.len() < n {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("buffer {} < aligned span {n}", buf.len())));
    }
    if (buf.as_ptr() as usize) % ALIGN != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "O_DIRECT buffer is not 4096-aligned"));
    }
    let step = if chunk > 0 && chunk < n { chunk } else { n };
    let need_total = (hi - alo) as usize;
    std::thread::scope(|scope| -> io::Result<()> {
        let mut handles = Vec::new();
        let mut first: Option<(&mut [u8], u64, usize)> = None;
        for (i, piece) in buf[..n].chunks_mut(step).enumerate() {
            let off = i * step;
            let need = need_total.saturating_sub(off).min(piece.len());
            if need == 0 {
                continue;
            }
            let at = alo + off as u64;
            if first.is_none() {
                first = Some((piece, at, need));
            } else {
                handles.push(scope.spawn(move || shard.file.pread_exact(piece, at, need)));
            }
        }
        if let Some((piece, at, need)) = first {
            shard.file.pread_exact(piece, at, need)?;
        }
        for h in handles {
            h.join().expect("read thread panicked")?;
        }
        Ok(())
    })?;
    Ok((lo - alo) as usize)
}

/// Where each of the six tensors landed in the staging buffer: `(offset, nbytes)` in
/// `EXPERT_NAMES` order (w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale).
pub type ExpertViews = [(usize, usize); 6];

/// Reads routed experts straight out of their safetensors shards with O_DIRECT.
pub struct ExpertReader<'a> {
    ckpt: &'a Checkpoint,
}

impl<'a> ExpertReader<'a> {
    pub fn new(ckpt: &'a Checkpoint) -> Self {
        ExpertReader { ckpt }
    }

    pub fn expert_prefix(layer: usize, expert: usize) -> String {
        format!("layers.{layer}.ffn.experts.{expert}.")
    }

    /// O_DIRECT-read one expert into `staging` (ALIGN-aligned, at least STAGE_BYTES) and
    /// return where its six tensors landed. The two file runs are each read in aligned
    /// pieces in parallel, so one expert alone keeps several NVMe requests in flight —
    /// the Python store's measured requirement (serial reads collapsed the device from
    /// ~4.7 GB/s to ~0.6 GB/s).
    pub fn read_expert_into(&self, staging: &mut [u8], prefix: &str) -> io::Result<ExpertViews> {
        let shard = self.ckpt.shard(&format!("{prefix}w1.weight"))?;
        let runs = shard.expert_runs(prefix)?;
        let mut views: ExpertViews = [(0, 0); 6];
        let mut cur = 0usize;
        for run in &runs {
            let n = aligned_len(run.lo, run.hi);
            if cur + n > staging.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "staging buffer too small"));
            }
            let off = read_span(shard, run.lo, run.hi, &mut staging[cur..cur + n], self.ckpt.read_chunk)?;
            for &(i, o, nb) in &run.members {
                views[i] = (cur + off + o, nb);
            }
            cur += n;
        }
        Ok(views)
    }

    /// Rank `rank`'s share of one expert out of `world`: the scale run whole (~1.1 MB, one
    /// request) and the rank's contiguous 1/world of each weight tensor (w1/w3 intermediate
    /// rows, w2 output rows - the tensor-parallel slot layout), one request each: ~10 MB
    /// read instead of 18.8. `views[i]` is where tensor i's rank slice landed and its bytes.
    pub fn read_expert_share_into(&self, staging: &mut [u8], prefix: &str, rank: usize, world: usize)
                                  -> io::Result<ExpertViews> {
        let shard = self.ckpt.shard(&format!("{prefix}w1.weight"))?;
        let runs = shard.expert_runs(prefix)?;
        let mut views: ExpertViews = [(0, 0); 6];
        let mut cur = 0usize;
        let read = |lo: u64, hi: u64, staging: &mut [u8], cur: &mut usize| -> io::Result<usize> {
            let n = aligned_len(lo, hi);
            if *cur + n > staging.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "staging buffer too small"));
            }
            let off = read_span(shard, lo, hi, &mut staging[*cur..*cur + n], self.ckpt.read_chunk)?;
            let at = *cur + off;
            *cur += n;
            Ok(at)
        };
        for run in &runs {
            let scales = run.members.iter().all(|&(i, _, _)| i % 2 == 1);
            if scales || world == 1 {
                let at = read(run.lo, run.hi, staging, &mut cur)?;
                for &(i, o, nb) in &run.members {
                    if nb % world != 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{prefix}: {nb} bytes do not split {world} ways")));
                    }
                    views[i] = (at + o + rank * nb / world, nb / world);
                }
            } else {
                for &(i, o, nb) in &run.members {
                    if nb % world != 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{prefix}: {nb} bytes do not split {world} ways")));
                    }
                    let (lo, part) = (run.lo + o as u64 + (rank * nb / world) as u64, nb / world);
                    views[i] = (read(lo, lo + part as u64, staging, &mut cur)?, part);
                }
            }
        }
        Ok(views)
    }

    /// The six tensors of one expert as owned byte vectors (tests and cold paths).
    pub fn read_expert(&self, layer: usize, expert: usize, prefix: Option<&str>) -> io::Result<Vec<Vec<u8>>> {
        let p = prefix.map(str::to_string).unwrap_or_else(|| Self::expert_prefix(layer, expert));
        let mut staging = AlignedBuf::new(STAGE_BYTES)?;
        let views = self.read_expert_into(&mut staging, &p)?;
        Ok(views.iter().map(|&(off, len)| staging[off..off + len].to_vec()).collect())
    }
}

/// The expected byte sizes of the six tensors, in `EXPERT_NAMES` order.
pub fn expert_tensor_sizes() -> [usize; 6] {
    let w13 = W13_SHAPE.0 * W13_SHAPE.1;
    let s13 = S13_SHAPE.0 * S13_SHAPE.1;
    let w2 = W2_SHAPE.0 * W2_SHAPE.1;
    let s2 = S2_SHAPE.0 * S2_SHAPE.1;
    [w13, s13, w2, s2, w13, s13]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> Option<Checkpoint> {
        let dir = std::env::var("TIDELM_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
        });
        dir.join("model.safetensors.index.json").exists().then(|| Checkpoint::open(&dir).unwrap())
    }

    #[test]
    fn expert_shares_are_the_full_reads_slices() {
        let Some(ckpt) = checkpoint() else { return };
        let reader = ExpertReader::new(&ckpt);
        let mut staging = AlignedBuf::new(STAGE_BYTES).unwrap();
        for (layer, expert) in [(0usize, 7usize), (39, 383)] {
            let full = reader.read_expert(layer, expert, None).unwrap();
            for rank in 0..2 {
                let views = reader.read_expert_share_into(&mut staging, &ExpertReader::expert_prefix(layer, expert), rank, 2).unwrap();
                for (i, &(off, n)) in views.iter().enumerate() {
                    assert_eq!(n, full[i].len() / 2);
                    assert_eq!(&staging[off..off + n], &full[i][rank * n..(rank + 1) * n], "{layer}/{expert} tensor {i} rank {rank}");
                }
            }
        }
    }

    #[test]
    fn expert_reads_match_buffered_io_on_the_real_checkpoint() {
        let Some(ckpt) = checkpoint() else {
            eprintln!("checkpoint not present; skipping");
            return;
        };
        let reader = ExpertReader::new(&ckpt);
        let sizes = expert_tensor_sizes();
        for (layer, expert) in [(0usize, 0usize), (17, 211), (39, 383)] {
            let p = ExpertReader::expert_prefix(layer, expert);
            let shard = ckpt.shard(&format!("{p}w1.weight")).unwrap();
            assert_eq!(shard.expert_runs(&p).unwrap().len(), 2, "expert {layer}/{expert} is not two runs");
            let tensors = reader.read_expert(layer, expert, None).unwrap();
            assert_eq!(tensors.iter().map(Vec::len).sum::<usize>(), EXPERT_BYTES);
            for (i, (t, name)) in tensors.iter().zip(EXPERT_NAMES).enumerate() {
                assert_eq!(t.len(), sizes[i], "{p}{name}");
                let (lo, hi) = shard.span(&format!("{p}{name}")).unwrap();
                let mut want = vec![0u8; (hi - lo) as usize];
                shard.file.pread_buffered(&mut want, lo).unwrap();
                assert_eq!(t, &want, "{p}{name} diverged from buffered IO");
            }
        }
    }

    #[test]
    fn dense_reads_match_buffered_io_and_carry_types() {
        let Some(ckpt) = checkpoint() else {
            eprintln!("checkpoint not present; skipping");
            return;
        };
        for name in ["layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale", "layers.3.hc_attn_fn",
                     "mtp.2.confidence_head.proj.weight"] {
            let (bytes, meta) = ckpt.read_tensor_vec(name).unwrap();
            let mut want = vec![0u8; meta.nbytes()];
            ckpt.shard(name).unwrap().file.pread_buffered(&mut want, meta.lo).unwrap();
            assert_eq!(bytes, want, "{name}");
        }
        let m = ckpt.meta("layers.0.attn.wq_a.weight").unwrap();
        assert_eq!((m.dtype.as_str(), m.shape.as_slice()), ("F8_E4M3", &[1280usize, 5120][..]));
        // a row range from the middle of a large tensor
        let row = 1000usize * 5120 * 2;
        let mut buf = AlignedBuf::new(aligned_len(0, 5120 * 2) + ALIGN).unwrap();
        let meta = ckpt.meta("embed.weight").unwrap();
        let off = ckpt.read_tensor_range("embed.weight", row, row + 5120 * 2, &mut buf).unwrap();
        let mut want = vec![0u8; 5120 * 2];
        ckpt.shard("embed.weight").unwrap().file.pread_buffered(&mut want, meta.lo + row as u64).unwrap();
        assert_eq!(&buf[off..off + 5120 * 2], &want[..]);
    }

    /// A safetensors file of `(name, dtype, shape, bytes)`, in that order.
    fn write_st(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, dtype, shape, bytes) in tensors {
            header.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape,
                                                               "data_offsets": [data.len(), data.len() + bytes.len()]}));
            data.extend_from_slice(bytes);
        }
        let hdr = serde_json::to_vec(&header).unwrap();
        std::fs::write(path, [&(hdr.len() as u64).to_le_bytes()[..], &hdr[..], &data[..]].concat()).unwrap();
    }

    #[test]
    fn an_overlay_serves_its_tensors_and_refuses_new_or_reshaped_ones() {
        let dir = std::env::temp_dir().join(format!("tidelm-overlay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stock = |n: usize, v: u8| vec![v; n];
        write_st(&dir.join("model-1.safetensors"), &[("a.weight", "F8_E4M3", vec![4, 8], stock(32, 1)),
                                                     ("a.scale", "F8_E8M0", vec![4, 1], stock(4, 2)),
                                                     ("b.weight", "BF16", vec![3], stock(6, 3))]);
        std::fs::write(dir.join("model.safetensors.index.json"),
                       r#"{"weight_map": {"a.weight": "model-1.safetensors", "a.scale": "model-1.safetensors",
                                          "b.weight": "model-1.safetensors"}}"#).unwrap();
        let ov = dir.join("overlay.safetensors");
        write_st(&ov, &[("a.weight", "F8_E4M3", vec![4, 8], stock(32, 9)), ("a.scale", "F8_E8M0", vec![4, 1], stock(4, 8))]);
        let mut ckpt = Checkpoint::open(&dir).unwrap();
        assert_eq!(ckpt.overlay(&ov).unwrap(), vec!["a.scale".to_string(), "a.weight".to_string()]);
        assert_eq!(ckpt.read_tensor_vec("a.weight").unwrap().0, stock(32, 9));
        assert_eq!(ckpt.read_tensor_vec("a.scale").unwrap().0, stock(4, 8));
        assert_eq!(ckpt.read_tensor_vec("b.weight").unwrap().0, stock(6, 3), "a tensor the overlay lacks stays the checkpoint's");
        assert_eq!(ckpt.shard("a.weight").unwrap().path, ov);
        // another shape, another dtype, or a tensor the checkpoint lacks: refused, nothing applied
        for bad in [("a.weight", "F8_E4M3", vec![8, 4], stock(32, 9)), ("a.weight", "BF16", vec![4, 4], stock(32, 9)),
                    ("c.weight", "BF16", vec![3], stock(6, 9))] {
            write_st(&ov, &[bad.clone()]);
            let mut ckpt = Checkpoint::open(&dir).unwrap();
            assert!(ckpt.overlay(&ov).is_err(), "{} {} {:?} accepted", bad.0, bad.1, bad.2);
            assert_eq!(ckpt.read_tensor_vec("a.weight").unwrap().0, stock(32, 1));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dspark_experts_are_reachable() {
        let Some(ckpt) = checkpoint() else {
            eprintln!("checkpoint not present; skipping");
            return;
        };
        let tensors = ExpertReader::new(&ckpt).read_expert(40, 5, Some("mtp.0.ffn.experts.5.")).unwrap();
        assert_eq!(tensors.iter().map(Vec::len).sum::<usize>(), EXPERT_BYTES);
    }
}
