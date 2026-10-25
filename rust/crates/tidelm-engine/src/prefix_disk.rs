//! The prefix cache's disk tier (engine/prefix_persistence.py + prefix_disk.py with
//! DSV41_PREFIX_DISK=1): a bundle per saved prompt in `DIR/rank-R/` - its tokens, its prefix
//! snapshots, and its lane's compressed rows (packed KV) and index keys up to its length,
//! stored once for every snapshot - and an index of the prefixes each bundle restores, by a
//! hash of their tokens. Bundles are this engine's own (the Python ones are torch pickles
//! keyed by its sources): a namespace of the executable, the model config and the settings
//! that change what is stored keeps a rebuild from loading another build's state.
//!
//! Writes run on a thread of their own (temporary file, fsync, rename, index, eviction to the
//! byte budget, least recently used first); a load checks the file's size and checksum before
//! any of it reaches the device. Each rank of a pair keeps its own copy of every bundle under
//! the same id (rank 0 decides; Engine mirrors).


use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::Result;

/// One prefix snapshot on the host (forward.rs LaneSnap's contents).
#[derive(Clone, Debug, PartialEq)]
pub struct SnapHost {
    pub n: usize,
    pub w: usize,
    pub rows: Vec<u16>,
    pub pending: Vec<f32>,
    pub flags: Vec<i32>,
}

/// A saved prompt: its tokens, snapshots, and per kv-source layer the compressed rows (the
/// cache's bytes as stored: packed or bf16) and index keys below its length.
#[derive(Clone, Debug, PartialEq)]
pub struct Bundle {
    pub id: String,
    pub namespace: u64,
    pub tokens: Vec<u32>,
    pub snaps: Vec<SnapHost>,
    pub ckv: Vec<(usize, Vec<u8>)>,
    pub ik: Vec<(usize, Vec<u8>)>,
}

const MAGIC: &[u8; 8] = b"TIDEPFX1";

/// A 64-bit checksum of the file's bytes, fast enough to run on every load (several GB/s).
pub fn checksum(b: &[u8]) -> u64 {
    let mut h = 0x243f_6a88_85a3_08d3u64 ^ b.len() as u64;
    let mut it = b.chunks_exact(8);
    for c in &mut it {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    for &x in it.remainder() {
        h = (h.rotate_left(5) ^ x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    h ^ (h >> 29)
}

/// FNV-1a over the tokens' little-endian bytes, at every length: `out[n]` hashes tokens[..n].
pub fn prefix_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut out = Vec::with_capacity(tokens.len() + 1);
    out.push(h);
    for &t in tokens {
        for b in t.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        out.push(h);
    }
    out
}

fn put_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn put_slice<T: Copy>(o: &mut Vec<u8>, v: &[T]) {
    put_u64(o, v.len() as u64);
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    o.extend_from_slice(bytes);
}

/// The bundle as bytes: magic, fields, then the checksum of everything before it.
pub fn encode(b: &Bundle) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(MAGIC);
    put_u64(&mut o, b.namespace);
    put_slice(&mut o, b.id.as_bytes());
    put_slice(&mut o, &b.tokens);
    put_u64(&mut o, b.snaps.len() as u64);
    for s in &b.snaps {
        put_u64(&mut o, s.n as u64);
        put_u64(&mut o, s.w as u64);
        put_slice(&mut o, &s.rows);
        put_slice(&mut o, &s.pending);
        put_slice(&mut o, &s.flags);
    }
    for part in [&b.ckv, &b.ik] {
        put_u64(&mut o, part.len() as u64);
        for (l, bytes) in part {
            put_u64(&mut o, *l as u64);
            put_slice(&mut o, bytes);
        }
    }
    let c = checksum(&o);
    put_u64(&mut o, c);
    o
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl Rd<'_> {
    fn u64(&mut self) -> Result<u64> {
        let v = self.b.get(self.at..self.at + 8).ok_or("prefix bundle: truncated")?;
        self.at += 8;
        Ok(u64::from_le_bytes(v.try_into().unwrap()))
    }

    fn vec<T: Copy + Default>(&mut self) -> Result<Vec<T>> {
        let n = self.u64()? as usize;
        let sz = n.checked_mul(std::mem::size_of::<T>()).ok_or("prefix bundle: bad length")?;
        let src = self.b.get(self.at..self.at + sz).ok_or("prefix bundle: truncated")?;
        let mut v = vec![T::default(); n];
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), v.as_mut_ptr() as *mut u8, sz) };
        self.at += sz;
        Ok(v)
    }
}

pub fn decode(b: &[u8]) -> Result<Bundle> {
    if b.len() < 16 || &b[..8] != MAGIC {
        return Err("prefix bundle: not a bundle".into());
    }
    let body = &b[..b.len() - 8];
    if checksum(body) != u64::from_le_bytes(b[b.len() - 8..].try_into().unwrap()) {
        return Err("prefix bundle: checksum mismatch".into());
    }
    let mut r = Rd { b: body, at: 8 };
    let namespace = r.u64()?;
    let id = String::from_utf8(r.vec::<u8>()?).map_err(|_| "prefix bundle: bad id")?;
    let tokens = r.vec::<u32>()?;
    let mut snaps = Vec::new();
    for _ in 0..r.u64()? {
        let (n, w) = (r.u64()? as usize, r.u64()? as usize);
        snaps.push(SnapHost { n, w, rows: r.vec()?, pending: r.vec()?, flags: r.vec()? });
    }
    let mut parts = [Vec::new(), Vec::new()];
    for part in parts.iter_mut() {
        for _ in 0..r.u64()? {
            let l = r.u64()? as usize;
            part.push((l, r.vec::<u8>()?));
        }
    }
    let [ckv, ik] = parts;
    if r.at != body.len() {
        return Err("prefix bundle: trailing bytes".into());
    }
    Ok(Bundle { id, namespace, tokens, snaps, ckv, ik })
}

/// One bundle in the index: the prefixes it restores (length, token hash), its size and
/// checksum, and when it was last saved or used.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub id: String,
    pub namespace: u64,
    pub size: u64,
    pub checksum: u64,
    pub used: f64,
    pub prefixes: Vec<(usize, u64)>,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct IndexFile {
    bundles: Vec<Entry>,
}

/// DSV41_PREFIX_DISK_GB / _MIN_INTERVAL_S / _DEDUP.
#[derive(Clone, Debug)]
pub struct DiskConfig {
    pub dir: PathBuf,
    pub budget: u64,
    pub min_interval_s: f64,
    pub dedup: bool,
}

fn now_s() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// A rank's bundle directory, its index and the one write in flight.
pub struct Disk {
    pub cfg: DiskConfig,
    dir: PathBuf,
    index: Arc<Mutex<Vec<Entry>>>,
    writer: Option<std::thread::JoinHandle<Result<()>>>,
    pub last_save: Option<std::time::Instant>,
}

impl Disk {
    /// `DIR/rank-R`, created 0700; unindexed bundles and stale temporaries removed, then the
    /// budget enforced.
    pub fn open(cfg: DiskConfig, rank: usize) -> Result<Self> {
        if cfg.budget == 0 {
            return Err("prefix disk: the budget must be > 0".into());
        }
        let dir = cfg.dir.join(format!("rank-{rank}"));
        std::fs::create_dir_all(&dir)?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        let index: Vec<Entry> = match std::fs::read(dir.join("index.json")) {
            Ok(b) => serde_json::from_slice::<IndexFile>(&b).map(|f| f.bundles).unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        let index: Vec<Entry> = index
            .into_iter()
            .filter(|e| std::fs::metadata(dir.join(format!("{}.bin", e.id))).map(|m| m.len() == e.size).unwrap_or(false))
            .collect();
        for f in std::fs::read_dir(&dir)?.flatten() {
            let name = f.file_name().to_string_lossy().to_string();
            let known = name.strip_suffix(".bin").is_some_and(|id| index.iter().any(|e| e.id == id));
            if (name.ends_with(".bin") && !known) || name.starts_with(".writing-") {
                let _ = std::fs::remove_file(f.path());
            }
        }
        let d = Disk { cfg, dir, index: Arc::new(Mutex::new(index)), writer: None, last_save: None };
        let mut idx = d.index.lock().unwrap();
        evict(&d.dir, &mut idx, d.cfg.budget);
        write_index(&d.dir, &idx)?;
        drop(idx);
        Ok(d)
    }

    /// Wait for the write in flight (a failed one is reported, not fatal).
    pub fn join(&mut self) {
        if let Some(h) = self.writer.take() {
            match h.join() {
                Ok(Err(e)) => eprintln!("prefix disk: a bundle write failed: {e}"),
                Err(_) => eprintln!("prefix disk: the writer panicked"),
                _ => {}
            }
        }
    }

    /// The bundles that restore a prefix of `prompt` of at most `max_n` tokens in
    /// `namespace`: (length, bundle id), the longest first, the most recently used first.
    pub fn candidates(&self, prompt: &[u32], namespace: u64, max_n: usize) -> Vec<(usize, String)> {
        let hs = prefix_hashes(&prompt[..max_n.min(prompt.len())]);
        let idx = self.index.lock().unwrap();
        let mut out: Vec<(usize, f64, String)> = idx
            .iter()
            .filter(|e| e.namespace == namespace)
            .flat_map(|e| e.prefixes.iter().filter(|&&(n, h)| n < hs.len() && hs[n] == h).map(|&(n, _)| (n, e.used, e.id.clone())))
            .collect();
        out.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)));
        out.into_iter().map(|(n, _, id)| (n, id)).collect()
    }

    /// A live bundle that already restores every one of `prefixes` (dedup: nothing to write).
    pub fn duplicate(&self, namespace: u64, prefixes: &[(usize, u64)]) -> Option<String> {
        let idx = self.index.lock().unwrap();
        idx.iter()
            .find(|e| e.namespace == namespace && prefixes.iter().all(|p| e.prefixes.contains(p)))
            .map(|e| e.id.clone())
    }

    pub fn touch(&self, id: &str) {
        let mut idx = self.index.lock().unwrap();
        if let Some(e) = idx.iter_mut().find(|e| e.id == id) {
            e.used = now_s();
        }
        let _ = write_index(&self.dir, &idx);
    }

    /// Write `b` on the writer thread (after the previous write), then index it and evict.
    pub fn save(&mut self, b: Bundle) {
        self.join();
        let (dir, index, budget) = (self.dir.clone(), self.index.clone(), self.cfg.budget);
        self.writer = Some(std::thread::spawn(move || -> Result<()> {
            let bytes = encode(&b);
            if bytes.len() as u64 > budget {
                return Err(format!("bundle {} is {} bytes, over the budget", b.id, bytes.len()).into());
            }
            let tmp = dir.join(format!(".writing-{}", b.id));
            let path = dir.join(format!("{}.bin", b.id));
            {
                let mut f = std::fs::File::create(&tmp)?;
                f.write_all(&bytes)?;
                f.sync_all()?;
            }
            std::fs::rename(&tmp, &path)?;
            std::fs::File::open(&dir)?.sync_all()?;
            let hs = prefix_hashes(&b.tokens);
            let entry = Entry {
                id: b.id.clone(),
                namespace: b.namespace,
                size: bytes.len() as u64,
                checksum: checksum(&bytes[..bytes.len() - 8]),
                used: now_s(),
                prefixes: b.snaps.iter().map(|s| (s.n, hs[s.n])).collect(),
            };
            let mut idx = index.lock().unwrap();
            idx.retain(|e| e.id != entry.id);
            idx.push(entry);
            evict(&dir, &mut idx, budget);
            write_index(&dir, &idx)
        }));
    }

    /// Bundle `id`, checked (size, checksum, namespace) before it is used.
    pub fn load(&mut self, id: &str, namespace: u64) -> Result<Bundle> {
        self.join();
        let entry = self.index.lock().unwrap().iter().find(|e| e.id == id).cloned().ok_or_else(|| format!("prefix disk: no bundle {id}"))?;
        let bytes = std::fs::read(self.dir.join(format!("{id}.bin")))?;
        if bytes.len() as u64 != entry.size {
            return Err(format!("prefix disk: bundle {id} is {} bytes, the index says {}", bytes.len(), entry.size).into());
        }
        let b = decode(&bytes)?;
        if b.namespace != namespace || b.id != id {
            return Err(format!("prefix disk: bundle {id} is another build's").into());
        }
        self.touch(id);
        Ok(b)
    }
}

/// Keep the most recently used bundles under `budget` bytes; delete the rest.
fn evict(dir: &Path, idx: &mut Vec<Entry>, budget: u64) {
    idx.sort_by(|a, b| b.used.partial_cmp(&a.used).unwrap_or(std::cmp::Ordering::Equal));
    let mut total = 0u64;
    let mut keep = Vec::new();
    for e in idx.drain(..) {
        total += e.size;
        if total <= budget {
            keep.push(e);
        } else {
            let _ = std::fs::remove_file(dir.join(format!("{}.bin", e.id)));
        }
    }
    *idx = keep;
}

fn write_index(dir: &Path, idx: &[Entry]) -> Result<()> {
    let tmp = dir.join(".writing-index");
    std::fs::write(&tmp, serde_json::to_vec(&IndexFile { bundles: idx.to_vec() })?)?;
    std::fs::rename(&tmp, dir.join("index.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle(id: &str, tokens: Vec<u32>) -> Bundle {
        Bundle {
            id: id.into(),
            namespace: 7,
            snaps: vec![SnapHost { n: 3, w: 3, rows: vec![1, 2, 3], pending: vec![0.5; 4], flags: vec![1] },
                        SnapHost { n: tokens.len() - 1, w: 4, rows: vec![9; 8], pending: vec![], flags: vec![] }],
            tokens,
            ckv: vec![(2, vec![1, 2, 3, 4]), (20, vec![5; 9])],
            ik: vec![(2, vec![7; 3])],
        }
    }

    #[test]
    fn a_bundle_round_trips_and_a_flipped_byte_is_caught() {
        let b = bundle("a", (0..10).collect());
        let mut bytes = encode(&b);
        assert_eq!(decode(&bytes).unwrap(), b);
        bytes[40] ^= 1;
        assert!(decode(&bytes).unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn saved_bundles_restore_their_prefixes_and_the_budget_holds() {
        let dir = std::env::temp_dir().join(format!("tidelm-pfx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = DiskConfig { dir: dir.clone(), budget: 1 << 20, min_interval_s: 300.0, dedup: true };
        let mut d = Disk::open(cfg.clone(), 0).unwrap();
        let toks: Vec<u32> = (100..140).collect();
        d.save(bundle("b1", toks.clone()));
        d.join();
        // the prompt shares 30 tokens: only the 3-token prefix applies; all 40: both
        let mut other = toks.clone();
        other[30] = 1;
        assert_eq!(d.candidates(&other, 7, other.len()), vec![(3, "b1".to_string())]);
        assert_eq!(d.candidates(&toks, 7, toks.len()), vec![(39, "b1".to_string()), (3, "b1".to_string())]);
        assert!(d.candidates(&toks, 8, toks.len()).is_empty(), "another namespace");
        assert_eq!(d.load("b1", 7).unwrap().tokens, toks);
        // reopened: the index is still there; a tiny budget evicts everything
        drop(d);
        let d = Disk::open(cfg.clone(), 0).unwrap();
        assert_eq!(d.candidates(&toks, 7, toks.len()).len(), 2);
        drop(d);
        let d = Disk::open(DiskConfig { budget: 10, ..cfg }, 0).unwrap();
        assert!(d.candidates(&toks, 7, toks.len()).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
