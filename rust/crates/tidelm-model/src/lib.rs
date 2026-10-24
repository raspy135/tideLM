//! Checkpoint access, from `engine/experts.py`: safetensors shard headers, the weight
//! index, and O_DIRECT expert reads into aligned staging buffers.
//!
//! The layout facts this crate encodes (and asserts in tests against the real
//! checkpoint): a routed expert is six tensors whose shard placement groups them into
//! exactly two contiguous file runs — a ~1.1 MB scale run and a ~17.7 MB weight run —
//! and reading those two runs in `read_chunk`-sized aligned pieces in parallel is what
//! keeps NVMe at ~5 GB/s when a decode layer misses only a few experts.

pub mod direct_io;
pub mod safetensors;

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use direct_io::{AlignedBuf, ALIGN};
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

/// Reads routed experts straight out of their safetensors shards with O_DIRECT.
/// Shards are opened lazily and kept open; `read_expert_into` is safe to call from
/// many threads at once (each call brings its own staging buffer).
pub struct ExpertReader {
    index: WeightIndex,
    shards: Mutex<HashMap<String, &'static Shard>>,
    /// Aligned chunk size for parallel piece reads (the Python DSV41_READ_CHUNK_MB=4).
    pub read_chunk: usize,
}

/// Where each of the six tensors landed in the staging buffer: `(offset, nbytes)` in
/// `EXPERT_NAMES` order (w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale).
pub type ExpertViews = [(usize, usize); 6];

impl ExpertReader {
    pub fn new(index: WeightIndex) -> Self {
        ExpertReader {
            index,
            shards: Mutex::new(HashMap::new()),
            read_chunk: 4 * 1024 * 1024 / ALIGN * ALIGN,
        }
    }

    pub fn expert_prefix(layer: usize, expert: usize) -> String {
        format!("layers.{layer}.ffn.experts.{expert}.")
    }

    fn shard(&self, tensor: &str) -> io::Result<&'static Shard> {
        let file = self.index.weight_map.get(tensor).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("{tensor} not in the weight index"))
        })?;
        let mut shards = self.shards.lock().unwrap();
        if let Some(s) = shards.get(file) {
            return Ok(s);
        }
        // Shards stay open for the life of the process, exactly as the Python store
        // keeps its fds; leaking the box gives the run cache a stable address without
        // self-referential lifetimes.
        let shard: &'static Shard =
            Box::leak(Box::new(Shard::open(&self.index.model_dir.join(file))?));
        shards.insert(file.clone(), shard);
        Ok(shard)
    }

    /// O_DIRECT-read one expert into `staging` (at least [`STAGE_BYTES`], ALIGN-aligned)
    /// and return where its six tensors landed. The two file runs are cut into
    /// `read_chunk` aligned pieces read in parallel, so one expert alone keeps several
    /// NVMe requests in flight — the Python store's measured requirement, not a guess
    /// (serial reads collapsed the device from ~4.7 GB/s to ~0.6 GB/s).
    pub fn read_expert_into(
        &self,
        staging: &mut AlignedBuf,
        layer: usize,
        expert: usize,
        prefix: Option<&str>,
    ) -> io::Result<ExpertViews> {
        let p = match prefix {
            Some(p) => p.to_string(),
            None => Self::expert_prefix(layer, expert),
        };
        let shard = self.shard(&format!("{p}w1.weight"))?;
        let runs = shard.expert_runs(&p)?;

        struct Job {
            off: u64,
            need: usize,
        }
        let mut views: ExpertViews = [(0, 0); 6];
        let mut jobs: Vec<Job> = Vec::new();
        let mut windows: Vec<(usize, usize)> = Vec::new(); // (staging offset, len)
        let mut cur = 0usize;
        for run in runs.iter() {
            let alo = run.lo - run.lo % ALIGN as u64;
            let ahi = run.hi.div_ceil(ALIGN as u64) * ALIGN as u64;
            let n = (ahi - alo) as usize;
            let step = if self.read_chunk > 0 && self.read_chunk < n { self.read_chunk } else { n };
            let mut off = 0usize;
            while off < n {
                let m = step.min(n - off);
                // the aligned tail may run past the tensor (and past EOF on the last
                // tensor of a shard): request the aligned length, require only `need`
                let need = ((run.hi - alo) as usize - off).min(m);
                if need > 0 {
                    jobs.push(Job { off: alo + off as u64, need });
                    windows.push((cur + off, m));
                }
                off += m;
            }
            let base = cur + (run.lo - alo) as usize;
            for &(i, o, nb) in &run.members {
                views[i] = (base + o, nb);
            }
            cur += n;
        }
        if cur > staging.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("staging buffer too small: need {cur}, have {}", staging.len()),
            ));
        }

        // Hand each job a disjoint &mut window of the staging buffer.
        let mut slices: Vec<&mut [u8]> = Vec::with_capacity(windows.len());
        let mut rest: &mut [u8] = staging;
        let mut consumed = 0usize;
        for &(w_off, w_len) in &windows {
            let (_, tail) = rest.split_at_mut(w_off - consumed);
            let (window, tail) = tail.split_at_mut(w_len);
            slices.push(window);
            rest = tail;
            consumed = w_off + w_len;
        }

        std::thread::scope(|scope| -> io::Result<()> {
            let mut handles = Vec::new();
            let mut first: Option<(&Job, &mut [u8])> = None;
            for (job, slice) in jobs.iter().zip(slices) {
                match first {
                    None => first = Some((job, slice)),
                    Some(_) => {
                        handles
                            .push(scope.spawn(move || shard.file.pread_exact(slice, job.off, job.need)));
                    }
                }
            }
            let (job, slice) = first.expect("an expert always has at least one run");
            shard.file.pread_exact(slice, job.off, job.need)?;
            for h in handles {
                h.join().expect("read thread panicked")?;
            }
            Ok(())
        })?;
        Ok(views)
    }

    /// The six tensors of one expert as owned byte vectors (tests and cold paths; the
    /// serving path hands the staging buffer to the arena H2D copy instead).
    pub fn read_expert(
        &self,
        layer: usize,
        expert: usize,
        prefix: Option<&str>,
    ) -> io::Result<Vec<Vec<u8>>> {
        let mut staging = AlignedBuf::new(STAGE_BYTES)?;
        let views = self.read_expert_into(&mut staging, layer, expert, prefix)?;
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

    fn checkpoint() -> Option<PathBuf> {
        let dir = std::env::var("TIDELM_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
            });
        dir.join("model.safetensors.index.json").exists().then_some(dir)
    }

    #[test]
    fn expert_reads_match_buffered_io_on_the_real_checkpoint() {
        let Some(dir) = checkpoint() else {
            eprintln!("checkpoint not present; skipping");
            return;
        };
        let reader = ExpertReader::new(WeightIndex::load(&dir).unwrap());
        let sizes = expert_tensor_sizes();
        for (layer, expert) in [(0usize, 0usize), (17, 211), (39, 383)] {
            let p = ExpertReader::expert_prefix(layer, expert);
            let shard = reader.shard(&format!("{p}w1.weight")).unwrap();
            let runs = shard.expert_runs(&p).unwrap();
            // the layout fact the chunked reader depends on: scales front, weights back
            assert_eq!(runs.len(), 2, "expert {layer}/{expert} is not two contiguous runs");

            let tensors = reader.read_expert(layer, expert, None).unwrap();
            let total: usize = tensors.iter().map(Vec::len).sum();
            assert_eq!(total, EXPERT_BYTES);
            for (i, (t, name)) in tensors.iter().zip(EXPERT_NAMES).enumerate() {
                assert_eq!(t.len(), sizes[i], "{p}{name}");
                // byte-exact against ordinary buffered reads of the same span
                let (lo, hi) = shard.span(&format!("{p}{name}")).unwrap();
                let mut want = vec![0u8; (hi - lo) as usize];
                shard.file.pread_buffered(&mut want, lo).unwrap();
                assert_eq!(t, &want, "{p}{name} diverged from buffered IO");
            }
        }
    }

    #[test]
    fn dspark_experts_are_reachable() {
        let Some(dir) = checkpoint() else {
            eprintln!("checkpoint not present; skipping");
            return;
        };
        let reader = ExpertReader::new(WeightIndex::load(&dir).unwrap());
        let tensors = reader.read_expert(40, 5, Some("mtp.0.ffn.experts.5.")).unwrap();
        assert_eq!(tensors.iter().map(Vec::len).sum::<usize>(), EXPERT_BYTES);
    }
}
