//! The per-lane compressed caches - each kv-source layer's compressed rows and index keys -
//! paged out of one pool of physical chunks (tidelm_cuda::vmm).
//!
//! Every cache keeps the layout the kernels address, `[lane slots][rows][width]` with a lane a
//! fixed stride apart, but the layout is a reserved address range, not an allocation: a lane's
//! part of it is backed by real chunks only as far as its contents reach, and everything past
//! that - every lane's tail and every lane nothing has used - maps one shared dummy chunk. So a
//! kernel reading past a lane's end (the indexer scores a power-of-two bucket of keys and masks
//! the ones past each row's count) reads the dummy and never faults, the kernels and their
//! graphs are unchanged, and a lane costs memory for the tokens it holds, not for max_seq.
//!
//! A chunk is the allocation granularity (2 MB on GB10): ~7K tokens of a ratio-1 layer's packed
//! rows, ~16K of a ratio-2 layer's keys. `ensure` maps a lane's chunks up to a step's end before
//! the step's device work (Engine::prepare), `release` gives a lane's back to the pool. Both run
//! the same way on both ranks of a pair, so their maps never differ. Remapping waits for the
//! device first: a kernel still in flight may be reading the range being changed.

use std::os::raw::c_void;

use tidelm_cuda::vmm::{DevPtr, Handle, Vmm};
use tidelm_cuda::Stream;
#[cfg(test)]
use tidelm_cuda::DeviceBuffer;

use crate::Result;

/// One cache's address range, as the kernels see it: u16 elements, lane L's part from
/// `L * lane_elems`.
#[derive(Clone, Copy, Debug)]
pub struct KvBuf {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for KvBuf {}

impl KvBuf {
    pub fn raw(&self) -> *mut c_void {
        self.ptr
    }

    /// Element `elem`'s address (bounds-checked against the whole range).
    pub fn ptr_at(&self, elem: usize) -> *mut c_void {
        assert!(elem <= self.len, "kv offset {elem} past length {}", self.len);
        unsafe { (self.ptr as *mut u16).add(elem) as *mut c_void }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device -> host from `[elem, elem + dst.len())`, ordered on `stream`.
    pub fn download_async(&self, elem: usize, dst: &mut [u16], stream: &Stream) -> Result<()> {
        assert!(elem + dst.len() <= self.len, "kv download past the end");
        unsafe {
            tidelm_cuda::memcpy_async(dst.as_mut_ptr() as *mut c_void, self.ptr_at(elem), std::mem::size_of_val(dst),
                                      tidelm_cuda::MEMCPY_DEVICE_TO_HOST, stream)?
        };
        Ok(())
    }
}

/// A cache to page: its layer's compression ratio and a row's bytes.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub ratio: usize,
    pub row_bytes: usize,
}

struct Region {
    spec: Spec,
    va: DevPtr,
    lane_bytes: usize,
    /// rows a lane (its part's stride, rounded up from max_seq / ratio + 1)
    rows: usize,
    /// per lane: the pool chunks behind its first granules, in order
    held: Vec<Vec<usize>>,
}

pub struct PagedKv {
    vmm: &'static Vmm,
    regions: Vec<Region>,
    lanes: usize,
    pool: Vec<Handle>,
    free: Vec<usize>,
    dummy: Handle,
    /// per lane: the tokens `release` keeps (the probe lane's filler and excerpt rows)
    floor: Vec<usize>,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl PagedKv {
    /// The granularity chunks come in (bytes).
    pub fn gran() -> Result<usize> {
        Ok(Vmm::get()?.gran)
    }

    /// Rows a lane for `max_rows` rows of every width in `row_bytes`: the next count whose part
    /// is whole chunks in each (a lane's part starts on a chunk, so no chunk spans two lanes).
    pub fn padded_rows(gran: usize, max_rows: usize, row_bytes: &[usize]) -> usize {
        let m = row_bytes.iter().map(|&b| gran / gcd(b, gran)).fold(1, |a, b| a / gcd(a, b) * b);
        max_rows.div_ceil(m) * m
    }

    /// Chunks one lane of `tokens` holds in a cache of `spec` (`rows` its lane's stride).
    pub fn chunks_for(gran: usize, spec: Spec, rows: usize, tokens: usize) -> usize {
        let r = (tokens / spec.ratio + 1).min(rows);
        (r * spec.row_bytes).div_ceil(gran)
    }

    /// `lanes` lane slots of each cache in `specs` (with its lane's rows), every granule on the
    /// dummy chunk, and a pool of `pool` chunks.
    pub fn new(lanes: usize, specs: &[(Spec, usize)], pool: usize) -> Result<Self> {
        let vmm = Vmm::get()?;
        let g = vmm.gran;
        let dummy = vmm.create()?;
        let mut kv = PagedKv { vmm, regions: Vec::new(), lanes, pool: Vec::with_capacity(pool), free: Vec::new(), dummy,
                               floor: vec![0; lanes] };
        for &(spec, rows) in specs {
            let lane_bytes = rows * spec.row_bytes;
            if lane_bytes % g != 0 {
                return Err(format!("kv: a lane of {rows} rows x {} bytes is not whole {g}-byte chunks", spec.row_bytes).into());
            }
            let va = vmm.reserve(lanes * lane_bytes)?;
            // into regions first, so Drop unmaps whatever this maps if a later map fails
            kv.regions.push(Region { spec, va, lane_bytes, rows, held: vec![Vec::new(); lanes] });
            for off in (0..lanes * lane_bytes).step_by(g) {
                vmm.map(va + off as u64, dummy)?;
            }
        }
        for i in 0..pool {
            kv.pool.push(vmm.create()?);
            kv.free.push(pool - 1 - i);
        }
        Ok(kv)
    }

    /// Cache `i`'s range (u16 elements).
    pub fn buf(&self, i: usize) -> KvBuf {
        let r = &self.regions[i];
        KvBuf { ptr: r.va as *mut c_void, len: self.lanes * r.lane_bytes / 2 }
    }

    /// Cache `i`'s rows a lane (the stride its kernels take, in rows).
    pub fn rows(&self, i: usize) -> usize {
        self.regions[i].rows
    }

    pub fn free_chunks(&self) -> usize {
        self.free.len()
    }

    pub fn total_chunks(&self) -> usize {
        self.pool.len()
    }

    /// Chunks `lane` holds over all caches.
    pub fn held(&self, lane: usize) -> usize {
        self.regions.iter().map(|r| r.held[lane].len()).sum()
    }

    /// Chunks `lane` still needs for its rows below position `end`.
    pub fn need(&self, lane: usize, end: usize) -> usize {
        let g = self.vmm.gran;
        self.regions.iter().map(|r| Self::chunks_for(g, r.spec, r.rows, end).saturating_sub(r.held[lane].len())).sum()
    }

    /// Map `lane`'s chunks for every row below position `end` (a no-op when it holds them).
    pub fn ensure(&mut self, lane: usize, end: usize) -> Result<()> {
        let need = self.need(lane, end);
        if need == 0 {
            return Ok(());
        }
        if need > self.free.len() {
            return Err(format!("kv pool: lane {lane} needs {need} more chunks to reach {end} tokens, {} of {} free",
                               self.free.len(), self.pool.len()).into());
        }
        tidelm_cuda::device_synchronize()?;
        let g = self.vmm.gran;
        for r in &mut self.regions {
            let want = Self::chunks_for(g, r.spec, r.rows, end);
            while r.held[lane].len() < want {
                let c = self.free.pop().unwrap();
                let va = r.va + (lane * r.lane_bytes + r.held[lane].len() * g) as u64;
                self.vmm.unmap(va)?;
                self.vmm.map(va, self.pool[c])?;
                r.held[lane].push(c);
            }
        }
        Ok(())
    }

    /// Give `lane`'s chunks past its floor back to the pool; its rows there read as the dummy's.
    /// The lane's contents past the floor are gone: the caller resets it. Returns the chunks freed.
    pub fn release(&mut self, lane: usize) -> Result<usize> {
        let (g, floor) = (self.vmm.gran, self.floor[lane]);
        let keep = |r: &Region| if floor > 0 { Self::chunks_for(g, r.spec, r.rows, floor) } else { 0 };
        if self.regions.iter().all(|r| r.held[lane].len() <= keep(r)) {
            return Ok(0);
        }
        tidelm_cuda::device_synchronize()?;
        let mut n = 0;
        for r in &mut self.regions {
            let keep = keep(r);
            while r.held[lane].len() > keep {
                let k = r.held[lane].len() - 1;
                let va = r.va + (lane * r.lane_bytes + k * g) as u64;
                self.vmm.unmap(va)?;
                self.vmm.map(va, self.dummy)?;
                self.free.push(r.held[lane].pop().unwrap());
                n += 1;
            }
        }
        Ok(n)
    }

    /// Keep `lane`'s rows below `tokens` mapped for good (`release` stops there).
    pub fn set_floor(&mut self, lane: usize, tokens: usize) -> Result<()> {
        self.ensure(lane, tokens)?;
        self.floor[lane] = tokens;
        Ok(())
    }
}

impl Drop for PagedKv {
    fn drop(&mut self) {
        let _ = tidelm_cuda::device_synchronize();
        let g = self.vmm.gran;
        for r in &self.regions {
            for off in (0..self.lanes * r.lane_bytes).step_by(g) {
                let _ = self.vmm.unmap(r.va + off as u64);
            }
            let _ = self.vmm.free(r.va, self.lanes * r.lane_bytes);
        }
        for &h in &self.pool {
            let _ = self.vmm.release(h);
        }
        let _ = self.vmm.release(self.dummy);
    }
}

/// A lane's rows of cache `buf` for tests: device -> host, `rows` rows of `width` u16 from lane
/// `lane` (its part `lane_elems` apart).
#[cfg(test)]
fn read_rows(buf: &KvBuf, lane: usize, lane_elems: usize, rows: usize, width: usize) -> Vec<u16> {
    let mut v = vec![0u16; rows * width];
    let st = Stream::new().unwrap();
    buf.download_async(lane * lane_elems, &mut v, &st).unwrap();
    st.synchronize().unwrap();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_makes_whole_chunks_for_every_width() {
        let g = 2 << 20;
        let r = PagedKv::padded_rows(g, 262_145, &[288, 256]);
        assert_eq!((r * 288) % g, 0);
        assert_eq!((r * 256) % g, 0);
        assert!(r >= 262_145 && r - 262_145 < 65_536);
        assert_eq!(PagedKv::padded_rows(g, 524_289, &[1024, 256]) % 8192, 0);
        // a lane of 4,096 tokens in a ratio-1 cache of 288-byte rows: 4,097 rows, one chunk
        assert_eq!(PagedKv::chunks_for(g, Spec { ratio: 1, row_bytes: 288 }, 1 << 20, 4096), 1);
        assert_eq!(PagedKv::chunks_for(g, Spec { ratio: 1, row_bytes: 288 }, 1 << 20, 7280), 1);
        assert_eq!(PagedKv::chunks_for(g, Spec { ratio: 1, row_bytes: 288 }, 1 << 20, 7281), 2);
        // never past the lane's own rows
        assert_eq!(PagedKv::chunks_for(g, Spec { ratio: 2, row_bytes: 256 }, 8192, 1 << 30), 1);
    }

    #[test]
    fn lanes_page_in_as_they_grow_and_give_their_chunks_back() {
        if !tidelm_cuda::gpu_available() {
            eprintln!("no CUDA device; skipping");
            return;
        }
        tidelm_cuda::set_device(0).unwrap();
        let g = PagedKv::gran().unwrap();
        let specs = [Spec { ratio: 1, row_bytes: 288 }, Spec { ratio: 2, row_bytes: 256 }];
        let rows = [PagedKv::padded_rows(g, 100_001, &[288]), PagedKv::padded_rows(g, 50_001, &[256])];
        let mut kv = PagedKv::new(3, &[(specs[0], rows[0]), (specs[1], rows[1])], 12).unwrap();
        assert_eq!((kv.free_chunks(), kv.need(0, 0)), (12, 2));
        kv.ensure(0, 10_000).unwrap();
        // 10,001 rows of 288 bytes: 2 chunks; 5,001 of 256: 1
        assert_eq!((kv.held(0), kv.free_chunks()), (3, 9));
        assert_eq!(kv.need(0, 10_000), 0);
        // what lane 0 writes it reads back, and lane 1 (all dummy) reads as nothing it wrote
        let st = Stream::new().unwrap();
        let b = kv.buf(0);
        let w = 144;
        let data: Vec<u16> = (0..10_001 * w).map(|i| (i % 65_521) as u16).collect();
        let src = DeviceBuffer::from_slice(&data).unwrap();
        unsafe {
            tidelm_cuda::memcpy_async(b.raw(), src.raw(), data.len() * 2, tidelm_cuda::MEMCPY_DEVICE_TO_DEVICE, &st).unwrap();
        }
        st.synchronize().unwrap();
        assert_eq!(read_rows(&b, 0, rows[0] * w, 10_001, w), data);
        let lane1 = read_rows(&b, 1, rows[0] * w, 64, w);
        assert_eq!(lane1.len(), 64 * w);
        // too big for the pool: refused, nothing taken
        assert!(kv.ensure(1, 100_000).is_err());
        assert_eq!(kv.free_chunks(), 9);
        // the probe lane keeps its floor through a release
        kv.set_floor(2, 4096).unwrap();
        kv.ensure(2, 30_000).unwrap();
        let before = kv.held(2);
        assert!(kv.release(2).unwrap() > 0);
        assert_eq!(kv.held(2), 2);
        assert!(before > 2);
        assert_eq!(kv.release(0).unwrap(), 3);
        assert_eq!(kv.free_chunks(), 12 - 2);
        // a released range reads (the dummy) without faulting
        let _ = read_rows(&b, 0, rows[0] * w, 10_001, w);
    }
}
