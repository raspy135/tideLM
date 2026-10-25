//! The prefix cache's disk tier (engine/prefix_persistence.py + prefix_disk.py with
//! DSV41_PREFIX_DISK=1): a bundle per saved prompt in `DIR/rank-R/` - its tokens, its prefix
//! snapshots, and its lane's compressed rows (packed KV) and index keys up to its length,
//! stored once for every snapshot - and an index of the prefixes each bundle restores, by a
//! hash of their tokens. Bundles are this engine's own (the Python ones are torch pickles
//! keyed by its sources): a namespace of the executable, the model config and the settings
//! that change what is stored keeps a rebuild from loading another build's state.
//!
//! Writes run on a thread of their own (temporary file, fsync, rename, index, eviction to the
//! byte budget, least recently used first). The engine streams a bundle both ways through a
//! pool of CHUNKS chunks of CHUNK bytes made when the tier opens (save_parts, BundleReader), so
//! a save or load holds the pool and nothing more however long the prompt is; a load checks the
//! file's size and checksum before the engine commits any of it. Each rank of a pair keeps its
//! own copy of every bundle under the same id (rank 0 decides; Engine mirrors).


use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
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
    /// the bounded replay's kept rows and decoder window floor (empty / 0 without it)
    pub rep_h: Vec<u16>,
    pub rep_pre: Vec<f32>,
    pub floor: i64,
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

// 2: snapshots carry the bounded replay's kept rows and window floor
pub const MAGIC: &[u8; 8] = b"TIDEPFX2";

/// A 64-bit checksum of the file's bytes, fast enough to run on every load (several GB/s).
pub fn checksum(b: &[u8]) -> u64 {
    let mut c = Checksum::new(b.len());
    c.update(b);
    c.finish()
}

/// `checksum` over bytes that arrive in pieces; `len` is the whole input's length.
pub struct Checksum {
    h: u64,
    carry: [u8; 8],
    nc: usize,
}

impl Checksum {
    pub fn new(len: usize) -> Self {
        Checksum { h: 0x243f_6a88_85a3_08d3u64 ^ len as u64, carry: [0; 8], nc: 0 }
    }

    fn mix(&mut self, w: u64) {
        self.h = (self.h.rotate_left(5) ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }

    pub fn update(&mut self, mut b: &[u8]) {
        if self.nc > 0 {
            let k = (8 - self.nc).min(b.len());
            self.carry[self.nc..self.nc + k].copy_from_slice(&b[..k]);
            self.nc += k;
            b = &b[k..];
            if self.nc < 8 {
                return;
            }
            self.mix(u64::from_le_bytes(self.carry));
            self.nc = 0;
        }
        let mut it = b.chunks_exact(8);
        for c in &mut it {
            self.mix(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let r = it.remainder();
        self.carry[..r.len()].copy_from_slice(r);
        self.nc = r.len();
    }

    pub fn finish(mut self) -> u64 {
        for i in 0..self.nc {
            let x = self.carry[i];
            self.mix(x as u64);
        }
        self.h ^ (self.h >> 29)
    }
}

/// The chunk pool's shape: 4 x 16 MB, allocated (and its pages written) when the tier opens.
pub const CHUNK: usize = 16 << 20;
pub const CHUNKS: usize = 4;

/// One run of a bundle's bytes in `encode`'s order: bytes on the host, or `len` bytes at a
/// device address that save_parts' `copy` fetches.
pub enum Part {
    Bytes(Vec<u8>),
    Device { ptr: usize, len: usize },
}

impl Part {
    fn len(&self) -> usize {
        match self {
            Part::Bytes(b) => b.len(),
            Part::Device { len, .. } => *len,
        }
    }
}

/// A chunk on its way to the writer thread, or the end of the bundle (or its abandonment).
enum Msg {
    Chunk(Vec<u8>, usize),
    Done,
    Abort,
}

/// `encode`'s little-endian field writers, for the engine's header parts.
pub fn le_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

/// A bundle read back in `encode`'s order through one of the pool's chunks; its checksum is
/// computed on the way and compared by `finish`, which the caller runs before it commits
/// anything it read.
pub struct BundleReader {
    file: std::fs::File,
    buf: Vec<u8>,
    have: usize,
    at: usize,
    /// body bytes not yet read into `buf`
    left: usize,
    sum: Option<Checksum>,
    want: u64,
    back: Sender<Vec<u8>>,
}

impl BundleReader {
    fn refill(&mut self) -> Result<()> {
        let n = self.left.min(self.buf.len());
        if n == 0 {
            return Err("prefix bundle: truncated".into());
        }
        self.file.read_exact(&mut self.buf[..n])?;
        self.sum.as_mut().unwrap().update(&self.buf[..n]);
        (self.have, self.at, self.left) = (n, 0, self.left - n);
        Ok(())
    }

    /// The next `n` bytes, through `each(piece, offset)` in the pieces the chunk holds.
    pub fn stream(&mut self, mut n: usize, each: &mut dyn FnMut(&[u8], usize) -> Result<()>) -> Result<()> {
        let mut off = 0;
        while n > 0 {
            if self.at == self.have {
                self.refill()?;
            }
            let k = n.min(self.have - self.at);
            each(&self.buf[self.at..self.at + k], off)?;
            (self.at, off, n) = (self.at + k, off + k, n - k);
        }
        Ok(())
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.stream(n, &mut |_, _| Ok(()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let mut v = [0u8; 8];
        self.stream(8, &mut |b, off| {
            v[off..off + b.len()].copy_from_slice(b);
            Ok(())
        })?;
        Ok(u64::from_le_bytes(v))
    }

    /// A length-prefixed field of `T`s into a host vector (the small ones: id, tokens, state).
    pub fn vec<T: Copy + Default>(&mut self) -> Result<Vec<T>> {
        let n = self.u64()? as usize;
        let sz = n.checked_mul(std::mem::size_of::<T>()).ok_or("prefix bundle: bad length")?;
        if sz > self.left + (self.have - self.at) {
            return Err("prefix bundle: truncated".into());
        }
        let mut v = vec![T::default(); n];
        let dst = v.as_mut_ptr() as *mut u8;
        self.stream(sz, &mut |b, off| {
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), dst.add(off), b.len()) };
            Ok(())
        })?;
        Ok(v)
    }

    /// Every byte read, and the checksum the file ends with matches them.
    pub fn finish(mut self) -> Result<()> {
        if self.at != self.have || self.left != 0 {
            return Err("prefix bundle: trailing bytes".into());
        }
        let mut c = [0u8; 8];
        self.file.read_exact(&mut c)?;
        let sum = self.sum.take().unwrap().finish();
        if sum != u64::from_le_bytes(c) || sum != self.want {
            return Err("prefix bundle: checksum mismatch".into());
        }
        Ok(())
    }
}

impl Drop for BundleReader {
    fn drop(&mut self) {
        let _ = self.back.send(std::mem::take(&mut self.buf));
    }
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
        put_slice(&mut o, &s.rep_h);
        put_slice(&mut o, &s.rep_pre);
        put_u64(&mut o, s.floor as u64);
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
        let (rows, pending, flags, rep_h, rep_pre) = (r.vec()?, r.vec()?, r.vec()?, r.vec()?, r.vec()?);
        snaps.push(SnapHost { n, w, rows, pending, flags, rep_h, rep_pre, floor: r.u64()? as i64 });
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

/// A rank's bundle directory, its index, the one write in flight and the chunk pool.
pub struct Disk {
    pub cfg: DiskConfig,
    dir: PathBuf,
    index: Arc<Mutex<Vec<Entry>>>,
    writer: Option<std::thread::JoinHandle<Result<()>>>,
    pub last_save: Option<std::time::Instant>,
    free: Receiver<Vec<u8>>,
    free_tx: Sender<Vec<u8>>,
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
        let (free_tx, free) = channel();
        for _ in 0..CHUNKS {
            let mut c = vec![0u8; CHUNK];
            // committed now, not on the first save
            for i in (0..CHUNK).step_by(4096) {
                unsafe { std::ptr::write_volatile(c.as_mut_ptr().add(i), 0) };
            }
            free_tx.send(c).map_err(|_| "prefix disk: the chunk pool")?;
        }
        let d = Disk { cfg, dir, index: Arc::new(Mutex::new(index)), writer: None, last_save: None, free, free_tx };
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

    /// Write `b` (a test's, or a caller with the bundle on the host): save_parts with one part.
    pub fn save(&mut self, b: Bundle) {
        let bytes = encode(&b);
        let hs = prefix_hashes(&b.tokens);
        let prefixes = b.snaps.iter().map(|s| (s.n, hs[s.n])).collect();
        let body = Part::Bytes(bytes[..bytes.len() - 8].to_vec());
        if let Err(e) = self.save_parts(&b.id, b.namespace, prefixes, &[body], &mut |_, _, _| Ok(())) {
            eprintln!("prefix disk: {e}");
        }
    }

    /// Write a bundle laid out as `parts` (`encode`'s bytes less the checksum): host parts are
    /// copied into the pool's chunks and device ones by `copy(dst, src, len)`, which returns
    /// once the bytes are there; the writer thread writes each full chunk and hands it back,
    /// then sums, renames, indexes and evicts. The caller waits only for a free chunk, so a save
    /// holds the pool's bytes and no more. `prefixes`: (length, prefix_hashes) it restores.
    pub fn save_parts(&mut self, id: &str, namespace: u64, prefixes: Vec<(usize, u64)>, parts: &[Part],
                      copy: &mut dyn FnMut(*mut u8, usize, usize) -> Result<()>) -> Result<()> {
        self.join();
        let body: usize = parts.iter().map(Part::len).sum();
        if (body + 8) as u64 > self.cfg.budget {
            return Err(format!("bundle {id} is {} bytes, over the budget", body + 8).into());
        }
        let (tx, rx) = channel::<Msg>();
        let (dir, index, budget, back, id_s) = (self.dir.clone(), self.index.clone(), self.cfg.budget, self.free_tx.clone(), id.to_string());
        self.writer = Some(std::thread::spawn(move || -> Result<()> {
            let tmp = dir.join(format!(".writing-{id_s}"));
            let path = dir.join(format!("{id_s}.bin"));
            let mut f = std::fs::File::create(&tmp);
            let mut sum = Checksum::new(body);
            let mut failed: Option<String> = f.as_ref().err().map(|e| e.to_string());
            let mut done = false;
            // every chunk goes back to the pool, whatever happened to the file
            while let Ok(m) = rx.recv() {
                match m {
                    Msg::Chunk(c, n) => {
                        if failed.is_none() {
                            match f.as_mut().unwrap().write_all(&c[..n]) {
                                Ok(()) => sum.update(&c[..n]),
                                Err(e) => failed = Some(e.to_string()),
                            }
                        }
                        let _ = back.send(c);
                    }
                    Msg::Done => {
                        done = true;
                        break;
                    }
                    Msg::Abort => break,
                }
            }
            let c = sum.finish();
            let finished = (|| -> Result<()> {
                if let Some(e) = failed {
                    return Err(e.into());
                }
                if !done {
                    return Err("the save was abandoned".into());
                }
                let mut f = f.map_err(|e| e.to_string())?;
                f.write_all(&c.to_le_bytes())?;
                f.sync_all()?;
                Ok(())
            })();
            if let Err(e) = finished {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!("bundle {id_s}: {e}").into());
            }
            std::fs::rename(&tmp, &path)?;
            std::fs::File::open(&dir)?.sync_all()?;
            let entry = Entry { id: id_s, namespace, size: (body + 8) as u64, checksum: c, used: now_s(), prefixes };
            let mut idx = index.lock().unwrap();
            idx.retain(|e| e.id != entry.id);
            idx.push(entry);
            evict(&dir, &mut idx, budget);
            write_index(&dir, &idx)
        }));
        let mut cur: Option<(Vec<u8>, usize)> = None;
        let mut fill = || -> Result<()> {
            for p in parts {
                let mut done = 0;
                while done < p.len() {
                    if cur.is_none() {
                        cur = Some((self.free.recv().map_err(|_| "prefix disk: the chunk pool is gone")?, 0));
                    }
                    let (c, at) = cur.as_mut().unwrap();
                    let k = (p.len() - done).min(c.len() - *at);
                    match p {
                        Part::Bytes(b) => c[*at..*at + k].copy_from_slice(&b[done..done + k]),
                        Part::Device { ptr, .. } => copy(c[*at..].as_mut_ptr(), ptr + done, k)?,
                    }
                    (*at, done) = (*at + k, done + k);
                    if *at == c.len() {
                        let (c, n) = cur.take().unwrap();
                        tx.send(Msg::Chunk(c, n)).map_err(|_| "prefix disk: the writer is gone")?;
                    }
                }
            }
            if let Some((c, n)) = cur.take() {
                tx.send(Msg::Chunk(c, n)).map_err(|_| "prefix disk: the writer is gone")?;
            }
            Ok(())
        };
        match fill() {
            Ok(()) => {
                let _ = tx.send(Msg::Done);
                Ok(())
            }
            Err(e) => {
                let _ = tx.send(Msg::Abort);
                Err(e)
            }
        }
    }

    /// Bundle `id` to stream back (its size checked against the index; its checksum by
    /// BundleReader::finish), through one of the pool's chunks.
    pub fn open_bundle(&mut self, id: &str) -> Result<BundleReader> {
        self.join();
        let entry = self.index.lock().unwrap().iter().find(|e| e.id == id).cloned().ok_or_else(|| format!("prefix disk: no bundle {id}"))?;
        let file = std::fs::File::open(self.dir.join(format!("{id}.bin")))?;
        let size = file.metadata()?.len();
        if size != entry.size || size < 16 {
            return Err(format!("prefix disk: bundle {id} is {size} bytes, the index says {}", entry.size).into());
        }
        let buf = self.free.recv().map_err(|_| "prefix disk: the chunk pool is gone")?;
        let body = size as usize - 8;
        let mut r = BundleReader { file, buf, have: 0, at: 0, left: body, sum: Some(Checksum::new(body)), want: entry.checksum,
                                   back: self.free_tx.clone() };
        let mut magic = [0u8; 8];
        r.stream(8, &mut |b, off| {
            magic[off..off + b.len()].copy_from_slice(b);
            Ok(())
        })?;
        if &magic != MAGIC {
            return Err(format!("prefix disk: bundle {id} is not a bundle").into());
        }
        Ok(r)
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
            snaps: vec![SnapHost { n: 3, w: 3, rows: vec![1, 2, 3], pending: vec![0.5; 4], flags: vec![1], rep_h: vec![7; 6],
                                   rep_pre: vec![0.25; 2], floor: 0 },
                        SnapHost { n: tokens.len() - 1, w: 4, rows: vec![9; 8], pending: vec![], flags: vec![], rep_h: vec![], rep_pre: vec![],
                                   floor: 5 }],
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
    fn the_streamed_checksum_is_the_whole_one_at_any_split() {
        let b: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        for split in [0, 1, 7, 8, 9, 333, 999, 1000] {
            let mut c = Checksum::new(b.len());
            c.update(&b[..split]);
            for piece in b[split..].chunks(5) {
                c.update(piece);
            }
            assert_eq!(c.finish(), checksum(&b), "split {split}");
        }
    }

    #[test]
    fn a_streamed_bundle_is_encode_byte_for_byte_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("tidelm-pfx-stream-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut d = Disk::open(DiskConfig { dir: dir.clone(), budget: 1 << 30, min_interval_s: 0.0, dedup: false }, 0).unwrap();
        // past one chunk, so a field straddles chunks on the way out and back
        let mut b = bundle("s1", (0..10).collect());
        b.ckv[1].1 = (0..CHUNK + 12_345).map(|i| (i % 253) as u8).collect();
        d.save(b.clone());
        d.join();
        let on_disk = std::fs::read(dir.join("rank-0/s1.bin")).unwrap();
        assert!(on_disk == encode(&b), "the file is encode's bytes");
        let mut r = d.open_bundle("s1").unwrap();
        assert_eq!((r.u64().unwrap(), r.vec::<u8>().unwrap()), (b.namespace, b"s1".to_vec()));
        assert_eq!(r.vec::<u32>().unwrap(), b.tokens);
        assert_eq!(r.u64().unwrap() as usize, b.snaps.len());
        for sh in &b.snaps {
            assert_eq!((r.u64().unwrap() as usize, r.u64().unwrap() as usize), (sh.n, sh.w));
            assert_eq!(r.vec::<u16>().unwrap(), sh.rows);
            assert_eq!(r.vec::<f32>().unwrap(), sh.pending);
            assert_eq!(r.vec::<i32>().unwrap(), sh.flags);
            assert_eq!(r.vec::<u16>().unwrap(), sh.rep_h);
            assert_eq!(r.vec::<f32>().unwrap(), sh.rep_pre);
            assert_eq!(r.u64().unwrap() as i64, sh.floor);
        }
        for part in [&b.ckv, &b.ik] {
            assert_eq!(r.u64().unwrap() as usize, part.len());
            for (l, bytes) in part {
                assert_eq!(r.u64().unwrap() as usize, *l);
                let n = r.u64().unwrap() as usize;
                let mut got = vec![0u8; n];
                r.stream(n, &mut |piece, off| {
                    got[off..off + piece.len()].copy_from_slice(piece);
                    Ok(())
                }).unwrap();
                assert!(&got == bytes);
            }
        }
        r.finish().unwrap();
        // a flipped byte: everything reads, the checksum does not hold
        let mut bad = on_disk.clone();
        bad[100] ^= 1;
        std::fs::write(dir.join("rank-0/s1.bin"), &bad).unwrap();
        let mut r = d.open_bundle("s1").unwrap();
        r.skip(bad.len() - 16).unwrap();
        assert!(r.finish().unwrap_err().to_string().contains("checksum"));
        let _ = std::fs::remove_dir_all(&dir);
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
