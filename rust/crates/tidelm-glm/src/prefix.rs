//! The disk prefix store (engine/glm/prefix_store.py, GLM_PREFIX_DISK=1): a prompt's KV rows
//! saved as a segment [start, end) of positions that extends its parent segment ending at start,
//! and restored by appending the rows of the deepest matching chain. GLM's caches are purely
//! positional rows, so a segment is, per rank, a list of contiguous device row ranges: its own
//! rows of every decoder layer (a split cache's positions 2j + rank), the MTP layer's rows
//! [start - 1, end - 1) (row p was made with token p + 1). Only prompt rows are saved; an
//! answer is saved as part of the next turn's prompt.
//!
//! Rank 0 keeps the index and plans; both ranks write and read their own files through
//! mirrored engine calls, and rank 1 reports each restore's outcome so the pair falls back
//! together. Files are 0600 under 0700 directories, read and written with O_DIRECT through a
//! pinned staging buffer (the page cache would take the GPU's memory on GB10).

use std::collections::BTreeMap;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use sha2::Digest;

use crate::Result;

/// The O_DIRECT block.
pub const ALIGN: usize = 4096;
/// Staging for a segment's rows on their way to or from the file.
pub const STAGE: usize = 64 << 20;
const MAGIC: &[u8; 8] = b"GLMSEG01";

/// sha256 of a token prefix (little-endian u32s): a segment's identity.
pub fn prefix_hash(tokens: &[u32]) -> String {
    let mut h = sha2::Sha256::new();
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub id: u64,
    pub parent: Option<u64>,
    pub start: usize,
    pub end: usize,
    pub hash: String,
    pub bytes: u64,
    pub used: u64,
}

/// Rank 0's index of a namespace's segments.
#[derive(Default)]
pub struct Index {
    pub segs: BTreeMap<u64, Segment>,
    pub next: u64,
    pub clock: u64,
}

pub struct Store {
    /// this rank's directory: <root>/<namespace>/rank-<r>
    pub dir: PathBuf,
    pub budget: u64,
    pub min_gain: usize,
    pub index: Index,
}

fn mkdir_private(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p)?;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

impl Store {
    pub fn open(root: &Path, ns: &str, rank: usize, budget: u64, min_gain: usize) -> Result<Self> {
        mkdir_private(root)?;
        mkdir_private(&root.join(ns))?;
        let dir = root.join(ns).join(format!("rank-{rank}"));
        mkdir_private(&dir)?;
        let mut s = Store { dir, budget, min_gain, index: Index::default() };
        if rank == 0 {
            s.load_index();
        }
        Ok(s)
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join("index.json")
    }

    fn load_index(&mut self) {
        let Ok(b) = std::fs::read(self.index_path()) else { return };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) else { return };
        let mut ix = Index::default();
        if let Some(m) = v.get("segments").and_then(|x| x.as_object()) {
            for (k, s) in m {
                let Ok(id) = k.parse::<u64>() else { continue };
                let g = |f: &str| s.get(f).and_then(|x| x.as_u64());
                let (Some(start), Some(end), Some(bytes)) = (g("start"), g("end"), g("bytes")) else { continue };
                ix.segs.insert(id, Segment { id, parent: g("parent"), start: start as usize, end: end as usize,
                                             hash: s.get("hash").and_then(|x| x.as_str()).unwrap_or("").into(),
                                             bytes, used: g("used").unwrap_or(0) });
                ix.next = ix.next.max(id + 1);
            }
        }
        ix.clock = v.get("clock").and_then(|x| x.as_u64()).unwrap_or(0);
        // a segment whose file is gone, or whose parent is, cannot be restored
        let ids: Vec<u64> = ix.segs.keys().copied().collect();
        for id in ids {
            if !self.dir.join(format!("{id}.seg")).exists() {
                ix.segs.remove(&id);
            }
        }
        loop {
            let orphans: Vec<u64> = ix.segs.values().filter(|s| s.parent.is_some_and(|p| !ix.segs.contains_key(&p))).map(|s| s.id).collect();
            if orphans.is_empty() {
                break;
            }
            for id in orphans {
                ix.segs.remove(&id);
            }
        }
        self.index = ix;
    }

    pub fn save_index(&self) -> Result<()> {
        let mut m = serde_json::Map::new();
        for s in self.index.segs.values() {
            m.insert(s.id.to_string(), serde_json::json!({"parent": s.parent, "start": s.start, "end": s.end, "hash": s.hash,
                                                          "bytes": s.bytes, "used": s.used}));
        }
        let v = serde_json::json!({"version": 1, "segments": m, "clock": self.index.clock});
        let tmp = self.index_path().with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(&v)?)?;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        std::fs::rename(&tmp, self.index_path())?;
        Ok(())
    }

    /// The segment chain from the root to `id`.
    pub fn chain(&self, id: u64) -> Vec<Segment> {
        let mut out = vec![];
        let mut cur = Some(id);
        while let Some(c) = cur {
            let Some(s) = self.index.segs.get(&c) else { break };
            out.push(s.clone());
            cur = s.parent;
        }
        out.reverse();
        out
    }

    /// The deepest segment whose prefix is `prompt`'s and that ends at or before `limit`.
    pub fn deepest(&self, prompt: &[u32], limit: usize) -> Option<Segment> {
        let mut best: Option<&Segment> = None;
        for s in self.index.segs.values() {
            if s.end > limit || best.is_some_and(|b| b.end > s.end || (b.end == s.end && b.used >= s.used)) {
                continue;
            }
            if prefix_hash(&prompt[..s.end]) == s.hash {
                best = Some(s);
            }
        }
        best.cloned()
    }

    /// A restore plan for `prompt` past `have` live positions: (the chain to load, the positions
    /// it gives), when that gains at least min_gain (one prompt token stays for its logits, so
    /// at most len - 1). Past the deepest whole match, the longest token match into one of its
    /// children (prefix_store.match's partial extension: an exact re-send's own segment, or a
    /// prompt that diverges inside a saved one); the child loads whole and its rows past the
    /// match are rewritten by the prefill.
    pub fn plan_restore(&self, prompt: &[u32], have: usize) -> Option<(Vec<Segment>, usize)> {
        let limit = prompt.len().saturating_sub(1);
        let full = self.deepest(prompt, limit);
        let base = full.as_ref().map(|s| s.end).unwrap_or(0);
        let mut best: Option<(Segment, usize)> = None;
        for c in self.index.segs.values() {
            if c.parent != full.as_ref().map(|s| s.id) || c.start != base {
                continue;
            }
            let Some(toks) = self.read_tokens(c.id) else { continue };
            let lcp = toks.iter().zip(&prompt[base..limit]).take_while(|(a, b)| a == b).count();
            let n = base + lcp;
            if n > base && best.as_ref().is_none_or(|b| n > b.1 || (n == b.1 && c.used > b.0.used)) {
                best = Some((c.clone(), n));
            }
        }
        let mut chain = full.as_ref().map(|s| self.chain(s.id)).unwrap_or_default();
        let mut n = base;
        if let Some((c, m)) = best {
            chain.push(c);
            n = m;
        }
        if chain.is_empty() || n < have + self.min_gain {
            return None;
        }
        Some((chain, n))
    }

    /// A segment's tokens [start, end) (rank 0 keeps them next to the index).
    pub fn write_tokens(&self, id: u64, toks: &[u32]) -> Result<()> {
        let p = self.dir.join(format!("{id}.tok"));
        let b: Vec<u8> = toks.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&p, b)?;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600))?;
        Ok(())
    }

    pub fn read_tokens(&self, id: u64) -> Option<Vec<u32>> {
        let b = std::fs::read(self.dir.join(format!("{id}.tok"))).ok()?;
        Some(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn touch(&mut self, ids: &[u64]) {
        for id in ids {
            self.index.clock += 1;
            if let Some(s) = self.index.segs.get_mut(id) {
                s.used = self.index.clock;
            }
        }
    }

    /// A save plan: (id, parent, start) for the prompt's rows past its deepest saved prefix, when
    /// at least min_gain new; with the segments to evict first so the store stays in budget.
    pub fn plan_save(&mut self, prompt: &[u32], bytes_per_pos: u64) -> Option<(Segment, Vec<u64>)> {
        let parent = self.deepest(prompt, prompt.len());
        let start = parent.as_ref().map(|p| p.end).unwrap_or(0);
        if prompt.len() < start + self.min_gain {
            return None;
        }
        let bytes = (prompt.len() - start) as u64 * bytes_per_pos;
        let keep: Vec<u64> = parent.as_ref().map(|p| self.chain(p.id).iter().map(|s| s.id).collect()).unwrap_or_default();
        // LRU leaves first, never the new segment's ancestors
        let mut evict = vec![];
        let mut total: u64 = self.index.segs.values().map(|s| s.bytes).sum();
        let mut segs: Vec<Segment> = self.index.segs.values().cloned().collect();
        while total + bytes > self.budget {
            let leaves: Vec<&Segment> = segs.iter().filter(|s| !keep.contains(&s.id) && !segs.iter().any(|c| c.parent == Some(s.id))).collect();
            let Some(v) = leaves.into_iter().min_by_key(|s| s.used) else { return None };
            let vid = v.id;
            total -= v.bytes;
            evict.push(vid);
            segs.retain(|s| s.id != vid);
        }
        let id = self.index.next;
        self.index.next += 1;
        let seg = Segment { id, parent: parent.map(|p| p.id), start, end: prompt.len(), hash: prefix_hash(prompt), bytes, used: 0 };
        Some((seg, evict))
    }

    pub fn commit(&mut self, seg: Segment) -> Result<()> {
        self.index.clock += 1;
        let mut s = seg;
        s.used = self.index.clock;
        self.index.segs.insert(s.id, s);
        self.save_index()
    }

    pub fn drop_segments(&mut self, ids: &[u64]) {
        for id in ids {
            let _ = std::fs::remove_file(self.dir.join(format!("{id}.seg")));
            let _ = std::fs::remove_file(self.dir.join(format!("{id}.tok")));
            self.index.segs.remove(id);
        }
    }

    pub fn path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id}.seg"))
    }
}

/// A segment file: MAGIC, then a 4 KB header (start, end, rank, the ranges' byte counts, the
/// prefix hash), then the ranges' bytes back to back, the total padded to the O_DIRECT block.
pub fn header(start: usize, end: usize, rank: usize, sizes: &[usize], hash: &str) -> Vec<u8> {
    let mut h = vec![0u8; ALIGN];
    let js = serde_json::to_vec(&serde_json::json!({"start": start, "end": end, "rank": rank, "sizes": sizes, "hash": hash}))
        .unwrap_or_default();
    h[..8].copy_from_slice(MAGIC);
    h[8..16].copy_from_slice(&(js.len() as u64).to_le_bytes());
    let n = js.len().min(ALIGN - 16);
    h[16..16 + n].copy_from_slice(&js[..n]);
    h
}

pub fn parse_header(h: &[u8]) -> Option<(usize, usize, usize, Vec<usize>, String)> {
    if &h[..8] != MAGIC {
        return None;
    }
    let n = u64::from_le_bytes(h[8..16].try_into().ok()?) as usize;
    let v: serde_json::Value = serde_json::from_slice(h.get(16..16 + n)?).ok()?;
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let sizes = v.get("sizes")?.as_array()?.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
    Some((u("start")?, u("end")?, u("rank")?, sizes, v.get("hash")?.as_str()?.to_string()))
}

pub fn open_direct_write(p: &Path) -> Result<std::fs::File> {
    Ok(std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).custom_flags(libc::O_DIRECT).open(p)?)
}

pub fn open_direct_read(p: &Path) -> Result<std::fs::File> {
    Ok(std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(p)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_match_restore_and_evict_leaves() {
        let root = std::env::temp_dir().join(format!("glm-prefix-test-{}", std::process::id()));
        let mut st = Store::open(&root, "ns", 0, 1000, 4).unwrap();
        let a: Vec<u32> = (0..10).collect();
        let (s1, ev) = st.plan_save(&a, 10).unwrap();
        assert!(ev.is_empty() && s1.start == 0 && s1.end == 10);
        std::fs::write(st.path(s1.id), b"x").unwrap();
        st.commit(s1.clone()).unwrap();
        let b: Vec<u32> = (0..20).collect();
        let (s2, _) = st.plan_save(&b, 10).unwrap();
        assert_eq!((s2.parent, s2.start, s2.end), (Some(s1.id), 10, 20));
        std::fs::write(st.path(s2.id), b"x").unwrap();
        st.commit(s2.clone()).unwrap();
        // a longer prompt restores the chain; a diverging one only the shared root
        let c: Vec<u32> = (0..30).collect();
        let (plan, n) = st.plan_restore(&c, 0).unwrap();
        assert_eq!((plan.iter().map(|s| s.id).collect::<Vec<_>>(), n), (vec![s1.id, s2.id], 20));
        // a prompt diverging inside s2: s1, then s2 partially (its tokens matched)
        st.write_tokens(s2.id, &b[10..20]).unwrap();
        let mut d: Vec<u32> = (0..30).collect();
        d[15] = 999;
        let (plan, n) = st.plan_restore(&d, 0).unwrap();
        assert_eq!((plan.len(), n), (2, 15));
        // an exact re-send of b: up to its last token
        assert_eq!(st.plan_restore(&b, 0).unwrap().1, 19);
        assert!(st.plan_restore(&d, 12).is_none()); // gains 3 < 4
        // reopened, the index survives
        let st2 = Store::open(&root, "ns", 0, 1000, 4).unwrap();
        assert_eq!(st2.index.segs.len(), 2);
        // a save past the budget evicts the oldest leaf that is not an ancestor
        let e: Vec<u32> = (100..190).collect();
        let (_, ev) = st.plan_save(&e, 10).unwrap();
        assert_eq!(ev, vec![s2.id]);
        let _ = std::fs::remove_dir_all(&root);
    }
}

// ------------------------------------------------------------------ the engine's side
use std::io::{Read, Write};
use std::os::raw::c_void;

use crate::forward::Engine;

pub(crate) const OP_DISK_SAVE: u32 = 30;
pub(crate) const OP_DISK_LOAD: u32 = 31;

impl Engine {
    /// This rank's device row ranges holding the selected lane's positions [start, end):
    /// (address, bytes) per cache tensor of every layer.
    pub(crate) fn seg_ranges(&self, start: usize, end: usize) -> Vec<(*mut u8, usize)> {
        let (rank, world) = (self.rank, self.world);
        let mut out = vec![];
        for (l, kv) in self.kv.iter().enumerate() {
            let mtp = self.cfg.mtp && l == self.cfg.mtp_layer();
            let (r0, r1) = if mtp {
                (start.saturating_sub(1), end.saturating_sub(1))
            } else if kv.split {
                let first = |p: usize| if p <= rank { 0 } else { (p - rank).div_ceil(world) };
                (first(start), first(end))
            } else {
                (start, end)
            };
            if r1 <= r0 {
                continue;
            }
            for (base, row) in kv.lane_bases(self.lane).into_iter().zip(crate::forward::KV_ROW_BYTES) {
                out.push((unsafe { (base as *mut u8).add(r0 * row) }, (r1 - r0) * row));
            }
        }
        out
    }

    /// A position's bytes in a segment on this rank (the store's budget estimate).
    pub fn bytes_per_position(&self) -> u64 {
        let r = self.seg_ranges(1000, 1002);
        r.iter().map(|x| x.1 as u64).sum::<u64>() / 2 + 4096 / 1000
    }

    fn ensure_stage(&mut self) -> Result<()> {
        if self.stage.is_none() {
            self.stage = Some(tidelm_cuda::PinnedBuffer::<u8>::uninit(STAGE)?);
        }
        Ok(())
    }

    /// Write this rank's rows of [start, end) to `path` (O_DIRECT, staged in 64 MB pieces).
    fn seg_write(&mut self, path: &Path, start: usize, end: usize, hash: &str) -> Result<()> {
        self.ensure_stage()?;
        let ranges = self.seg_ranges(start, end);
        let sizes: Vec<usize> = ranges.iter().map(|r| r.1).collect();
        let mut f = open_direct_write(path)?;
        let stage = self.stage.as_mut().unwrap();
        stage[..ALIGN].copy_from_slice(&header(start, end, self.rank, &sizes, hash));
        let mut fill = ALIGN;
        let st = &self.st;
        let flush = |stage: &mut tidelm_cuda::PinnedBuffer<u8>, fill: &mut usize, f: &mut std::fs::File, last: bool| -> Result<()> {
            st.synchronize()?;
            let n = if last { fill.div_ceil(ALIGN) * ALIGN } else { *fill / ALIGN * ALIGN };
            if n > 0 {
                stage[*fill..n.max(*fill)].iter_mut().for_each(|b| *b = 0);
                f.write_all(&stage[..n])?;
                let rest = *fill - n.min(*fill);
                stage.copy_within(n..n + rest, 0);
                *fill = rest;
            }
            Ok(())
        };
        for &(src, n) in &ranges {
            let mut off = 0;
            while off < n {
                let take = (STAGE - fill).min(n - off);
                unsafe {
                    tidelm_cuda::memcpy_async(stage[fill..].as_mut_ptr() as *mut c_void, src.add(off) as *const c_void, take,
                                              tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST, st)?;
                }
                fill += take;
                off += take;
                if fill == STAGE {
                    flush(stage, &mut fill, &mut f, false)?;
                }
            }
        }
        flush(stage, &mut fill, &mut f, true)?;
        f.sync_all()?;
        Ok(())
    }

    /// Read this rank's rows of a segment file back into the caches; Ok(false) when the file is
    /// missing or is not the segment it should be.
    fn seg_read(&mut self, path: &Path, start: usize, end: usize, hash: &str) -> Result<bool> {
        self.ensure_stage()?;
        let ranges = self.seg_ranges(start, end);
        let Ok(mut f) = open_direct_read(path) else { return Ok(false) };
        let stage = self.stage.as_mut().unwrap();
        if f.read_exact(&mut stage[..ALIGN]).is_err() {
            return Ok(false);
        }
        let Some((s0, e0, r0, sizes, h0)) = parse_header(&stage[..ALIGN]) else { return Ok(false) };
        if (s0, e0, r0) != (start, end, self.rank) || h0 != hash || sizes != ranges.iter().map(|r| r.1).collect::<Vec<_>>() {
            return Ok(false);
        }
        let total: usize = sizes.iter().sum();
        let (mut have, mut pos) = (0usize, 0usize); // bytes in the stage, bytes of the payload consumed
        let mut ri = 0usize;
        let mut roff = 0usize;
        let st = &self.st;
        while pos < total {
            // refill: the unread tail moved to the front, then whole blocks read after it
            let want = ((total - pos).min(STAGE / 2)).div_ceil(ALIGN) * ALIGN;
            let mut got = 0;
            while got < want {
                let r = f.read(&mut stage[got..want])?;
                if r == 0 {
                    break;
                }
                got += r;
            }
            have = got.min(total - pos);
            let mut used = 0;
            while used < have && ri < ranges.len() {
                let (dst, n) = ranges[ri];
                let take = (n - roff).min(have - used);
                unsafe {
                    tidelm_cuda::memcpy_async(dst.add(roff) as *mut c_void, stage[used..].as_ptr() as *const c_void, take,
                                              tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
                }
                used += take;
                roff += take;
                if roff == n {
                    ri += 1;
                    roff = 0;
                }
            }
            st.synchronize()?;
            pos += used;
            if used == 0 {
                return Ok(false);
            }
        }
        let _ = have;
        Ok(true)
    }

    fn pair_status(&mut self, ok: bool) -> Result<bool> {
        if self.world == 1 {
            return Ok(ok);
        }
        let p = self.pair.as_mut().ok_or("no pair")?;
        if self.rank == 1 {
            p.ctl.send(&[ok as u8])?;
            return Ok(ok);
        }
        let r = p.ctl.recv()?;
        Ok(ok && r.first() == Some(&1))
    }

    /// Save the selected lane's positions [start, end) as segment `id` on both ranks (after
    /// dropping `evict`'s files): true when both wrote it.
    pub fn disk_save(&mut self, dir_id: u64, start: usize, end: usize, hash: &str, evict: &[u64]) -> Result<bool> {
        let ev: Vec<u32> = evict.iter().map(|&x| x as u32).collect();
        let h = hash.to_string();
        let lane = self.lane;
        self.mirror_msg(|m| {
            m.u32(OP_DISK_SAVE).u32(lane as u32).u64(dir_id).u32(start as u32).u32(end as u32).u32s(&ev);
            m.u32s(&h.bytes().map(|b| b as u32).collect::<Vec<_>>());
        })?;
        self.disk_save_local(dir_id, start, end, hash, evict)
    }

    fn disk_save_local(&mut self, id: u64, start: usize, end: usize, hash: &str, evict: &[u64]) -> Result<bool> {
        let store_dir = self.prefix_dir.clone().ok_or("no prefix store")?;
        for e in evict {
            let _ = std::fs::remove_file(store_dir.join(format!("{e}.seg")));
        }
        let path = store_dir.join(format!("{id}.seg"));
        let ok = match self.seg_write(&path, start, end, hash) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[prefix] rank {} save failed: {e}", self.rank);
                let _ = std::fs::remove_file(&path);
                false
            }
        };
        self.pair_status(ok)
    }

    /// Restore a chain of segments [(id, start, end, hash)] into the selected lane on both
    /// ranks: true when both read every row (the lane then holds `tokens` up to the chain's end).
    /// The caller checks the pool can back n positions (kv_room).
    pub fn disk_load(&mut self, chain: &[(u64, usize, usize, String)], tokens: &[u32], n: usize) -> Result<bool> {
        let flat: Vec<u32> = chain.iter().flat_map(|(id, s, e, _)| [*id as u32, (*id >> 32) as u32, *s as u32, *e as u32]).collect();
        let hashes: String = chain.iter().map(|c| c.3.clone()).collect::<Vec<_>>().join(",");
        let lane = self.lane;
        self.mirror_msg(|m| {
            m.u32(OP_DISK_LOAD).u32(lane as u32).u32(n as u32).u32s(&flat).u32s(&hashes.bytes().map(|b| b as u32).collect::<Vec<_>>());
        })?;
        let ok = self.disk_load_local(chain, n)?;
        if ok {
            self.tokens = tokens[..n].to_vec();
        }
        Ok(ok)
    }

    fn disk_load_local(&mut self, chain: &[(u64, usize, usize, String)], n: usize) -> Result<bool> {
        let store_dir = self.prefix_dir.clone().ok_or("no prefix store")?;
        self.kv_ensure(self.lane, n)?;
        let mut ok = true;
        for (id, s, e, h) in chain {
            match self.seg_read(&store_dir.join(format!("{id}.seg")), *s, *e, h) {
                Ok(true) => {}
                Ok(false) => {
                    ok = false;
                    break;
                }
                Err(err) => {
                    eprintln!("[prefix] rank {} restore failed: {err}", self.rank);
                    ok = false;
                    break;
                }
            }
        }
        let ok = self.pair_status(ok)?;
        if ok {
            self.pos = n;
        }
        Ok(ok)
    }

    pub(crate) fn follow_disk(&mut self, op: u32, r: &mut tidelm_comm::Reader) -> Result<()> {
        self.select(r.u32()? as usize)?;
        if op == OP_DISK_SAVE {
            let id = r.u64()?;
            let (s, e) = (r.u32()? as usize, r.u32()? as usize);
            let ev: Vec<u64> = r.u32s()?.into_iter().map(|x| x as u64).collect();
            let h: String = r.u32s()?.into_iter().map(|b| b as u8 as char).collect();
            self.disk_save_local(id, s, e, &h, &ev)?;
        } else {
            let n = r.u32()? as usize;
            let flat = r.u32s()?;
            let hs: String = r.u32s()?.into_iter().map(|b| b as u8 as char).collect();
            let hs: Vec<String> = hs.split(',').map(|x| x.to_string()).collect();
            let chain: Vec<(u64, usize, usize, String)> = flat.chunks(4).zip(hs)
                .map(|(c, h)| (c[0] as u64 | ((c[1] as u64) << 32), c[2] as usize, c[3] as usize, h)).collect();
            self.disk_load_local(&chain, n)?;
        }
        Ok(())
    }
}

/// Free bytes on the filesystem holding `p`.
pub fn free_bytes(p: &Path) -> u64 {
    let Ok(c) = std::ffi::CString::new(p.as_os_str().to_string_lossy().as_bytes()) else { return 0 };
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return 0;
    }
    s.f_bavail as u64 * s.f_frsize as u64
}
