//! The one-shot RoCE all-gather (tools/roce/: b12x's RoCEnante via TensorFold's patch 0230):
//! each rank owns a pinned host region the ConnectX-7 RDMA-writes into and the GB10 reads in
//! place; one kernel stages this rank's shard, rings a doorbell that a busy-spinning proxy
//! thread turns into RDMA writes (payload, then a sequence flag on the same RC queue pair,
//! striped over the HCAs), waits for the peer's flag and copies every shard out in rank
//! order. The epoch lives on the device, so a replayed CUDA graph carries the sequence on.
//! Measured in Python's engine at 12.7 / 15.5 / 19.9 us a graph gather of 16 / 48 / 96 KiB,
//! against 44-70 us for NCCL's.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

use tidelm_cuda::{DeviceBuffer, Stream};

use crate::{Control, Result};

extern "C" {
    fn roce_layout(world: c_int, slot_bytes: u64, out7: *mut u64) -> c_int;
    fn roce_blob_bytes() -> c_int;
    fn roce_proxy_create(world: c_int, rank: c_int, names: *const *const c_char, gid_index: *const c_int, n_hca: c_int,
                         tc: c_int, region: *mut c_void, region_bytes: u64, slot_bytes: u64, err: *mut c_char,
                         err_len: c_int) -> *mut c_void;
    fn roce_proxy_blob(p: *mut c_void, out: *mut c_void) -> c_int;
    fn roce_proxy_connect(p: *mut c_void, blobs: *const c_void, len: u64, err: *mut c_char, err_len: c_int) -> c_int;
    fn roce_proxy_start(p: *mut c_void, cpu: c_int, err: *mut c_char, err_len: c_int) -> c_int;
    fn roce_proxy_failed(p: *mut c_void, err: *mut c_char, err_len: c_int) -> c_int;
    fn roce_proxy_destroy(p: *mut c_void);
    #[allow(clippy::too_many_arguments)]
    fn roce_gather(stream: *mut c_void, input: *const c_void, out: *mut c_void, nbytes: u64, grid: c_int, threads: c_int,
                   recv_base: u64, flag_base: u64, send_base: u64, ctrl_base: u64, slot_bytes: u64, epoch: *mut c_void,
                   stage_ctr: *mut c_void, tail_ctr: *mut c_void, poison: *mut c_void, timeout_ns: u64, world: c_int,
                   rank: c_int, n_hca: c_int, flag_stride: c_int, slots: c_int) -> c_int;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
}

const CTRL_ERR_SEQ: usize = 2;
const CTRL_ERR_PEER: usize = 3;
const CTRL_ERR_HCA: usize = 6;
const CTRL_FAILED: usize = 7;
const THREADS: i32 = 512;
const MAX_BLOCKS: usize = 8;
const PACK: usize = 16;

/// An active RoCE v2 port: device name, the lowest IPv4-mapped RoCE v2 GID index (GID
/// indices move on reboot, so they are never configured), its IPv4 address.
#[derive(Clone, Debug)]
pub struct Hca {
    pub name: String,
    pub gid_index: i32,
    pub ip: [u8; 4],
}

/// Every ACTIVE Ethernet RDMA port with an IPv4-mapped RoCE v2 GID, in subnet order
/// (tools/roce/roce.py::detect). `only`: restrict to these device names (NCCL_IB_HCA).
pub fn detect(only: Option<&[String]>) -> Vec<Hca> {
    let root = std::path::Path::new("/sys/class/infiniband");
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string());
    let mut out = Vec::new();
    let Ok(devs) = std::fs::read_dir(root) else { return out };
    for dev in devs.flatten() {
        let name = dev.file_name().to_string_lossy().to_string();
        if only.is_some_and(|o| !o.contains(&name)) {
            continue;
        }
        let port = dev.path().join("ports/1");
        if !read(port.join("state")).is_some_and(|s| s.contains("ACTIVE"))
            || read(port.join("link_layer")).as_deref() != Some("Ethernet")
        {
            continue;
        }
        let mut idx: Vec<i32> = std::fs::read_dir(port.join("gids"))
            .map(|d| d.flatten().filter_map(|e| e.file_name().to_string_lossy().parse().ok()).collect())
            .unwrap_or_default();
        idx.sort_unstable();
        for i in idx {
            if read(port.join(format!("gid_attrs/types/{i}"))).as_deref() != Some("RoCE v2") {
                continue;
            }
            let Some(gid) = read(port.join(format!("gids/{i}"))) else { continue };
            let h: String = gid.chars().filter(|c| *c != ':').collect();
            if h.len() == 32 && h[..20].chars().all(|c| c == '0') && &h[20..24] == "ffff" {
                let b = u32::from_str_radix(&h[24..], 16).unwrap_or(0).to_be_bytes();
                out.push(Hca { name: name.clone(), gid_index: i, ip: b });
                break;
            }
        }
    }
    out.sort_by_key(|h| h.ip);
    out
}

/// A rank's RoCE runtime: its pinned region, the proxy thread, the device counters.
pub struct Roce {
    proxy: *mut c_void,
    host: *mut c_void,
    dev: u64,
    lay: [u64; 7],
    slot_bytes: u64,
    counters: DeviceBuffer<u32>,
    world: usize,
    rank: usize,
    n_hca: usize,
    timeout_ns: u64,
    pub max_bytes: usize,
}

unsafe impl Send for Roce {}
unsafe impl Sync for Roce {}

fn errbuf() -> Vec<c_char> {
    vec![0; 512]
}
fn errstr(b: &[c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned()
}

impl Roce {
    /// Collective (both ranks, over `ctl`): detect and pair the HCAs by subnet, register the
    /// region, exchange connection records, start the proxy, then gather a known pattern and
    /// check it on both ranks. `max_bytes`: the largest shard a gather may carry.
    pub fn setup(rank: usize, ctl: &mut Control, max_bytes: usize, only: Option<&[String]>) -> Result<Roce> {
        let world = 2usize;
        let hcas = detect(only);
        // pair by subnet: both ranks must expose the same /24s, in the same (sorted) order
        let subnets: Vec<u8> = hcas.iter().flat_map(|h| h.ip[..3].to_vec()).collect();
        ctl.send(&subnets)?;
        let theirs = ctl.recv()?;
        if hcas.is_empty() || theirs != subnets {
            return Err(format!("rank {rank}: RoCE ports do not pair (mine {:?}, peer's subnets {theirs:?})", hcas).into());
        }
        let slot_bytes = (max_bytes.max(PACK) as u64).div_ceil(4096) * 4096;
        let mut lay = [0u64; 7];
        if unsafe { roce_layout(world as c_int, slot_bytes, lay.as_mut_ptr()) } != 0 {
            return Err("roce: unsupported geometry".into());
        }
        let total = lay[4] as usize;
        let mut host: *mut c_void = std::ptr::null_mut();
        // portable | mapped: the GPU reads what the NIC writes, in place
        let rc = unsafe { cudaHostAlloc(&mut host, total, 1 | 2) };
        if rc != 0 {
            return Err(format!("roce: cudaHostAlloc({total}) failed ({rc})").into());
        }
        unsafe { std::ptr::write_bytes(host as *mut u8, 0, total) };
        let mut dev: *mut c_void = std::ptr::null_mut();
        if unsafe { cudaHostGetDevicePointer(&mut dev, host, 0) } != 0 {
            unsafe { cudaFreeHost(host) };
            return Err("roce: cudaHostGetDevicePointer failed".into());
        }
        let names: Vec<CString> = hcas.iter().map(|h| CString::new(h.name.clone()).unwrap()).collect();
        let name_ptrs: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
        let gids: Vec<c_int> = hcas.iter().map(|h| h.gid_index).collect();
        let tc: c_int = std::env::var("NCCL_IB_TC").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut err = errbuf();
        let proxy = unsafe {
            roce_proxy_create(world as c_int, rank as c_int, name_ptrs.as_ptr(), gids.as_ptr(), hcas.len() as c_int, tc,
                              host, total as u64, slot_bytes, err.as_mut_ptr(), err.len() as c_int)
        };
        if proxy.is_null() {
            unsafe { cudaFreeHost(host) };
            return Err(format!("roce: {}", errstr(&err)).into());
        }
        // epoch, stage arrivals per grid size, tail arrivals per grid size, poison
        let counters = DeviceBuffer::<u32>::zeroed(1 + 2 * 16 + 1)?;
        let timeout_ns = (std::env::var("TIDELM_ROCE_TIMEOUT_S").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(60.0)
            * 1e9) as u64;
        let r = Roce { proxy, host, dev: dev as u64, lay, slot_bytes, counters, world, rank, n_hca: hcas.len(), timeout_ns,
                       max_bytes };
        let blob_len = unsafe { roce_blob_bytes() } as usize;
        let mut mine = vec![0u8; blob_len];
        unsafe { roce_proxy_blob(r.proxy, mine.as_mut_ptr() as *mut c_void) };
        ctl.send(&mine)?;
        let peer = ctl.recv()?;
        if peer.len() != blob_len {
            return Err("roce: the peer's connection record has another size (different builds?)".into());
        }
        let blobs: Vec<u8> = if rank == 0 { [mine, peer].concat() } else { [peer, mine].concat() };
        if unsafe { roce_proxy_connect(r.proxy, blobs.as_ptr() as *const c_void, blobs.len() as u64, err.as_mut_ptr(),
                                       err.len() as c_int) } != 0
        {
            return Err(format!("roce: {}", errstr(&err)).into());
        }
        if unsafe { roce_proxy_start(r.proxy, -1, err.as_mut_ptr(), err.len() as c_int) } != 0 {
            return Err(format!("roce: {}", errstr(&err)).into());
        }
        // both ranks must be connected before either posts a write
        ctl.send(b"roce-ready")?;
        if ctl.recv()? != b"roce-ready" {
            return Err("roce: the peer did not come up".into());
        }
        r.probe(ctl)?;
        Ok(r)
    }

    fn probe(&self, ctl: &mut Control) -> Result<()> {
        let stream = Stream::new()?;
        for &n in &[16usize, 4096, 20480, self.max_bytes.min(1 << 20)] {
            let words = n / 4;
            let mine: Vec<u32> = (0..words as u32).map(|i| i.wrapping_mul(2654435761) ^ (self.rank as u32 + 1)).collect();
            let input = DeviceBuffer::from_slice(&mine)?;
            let out = DeviceBuffer::<u32>::zeroed(words * self.world)?;
            self.gather(input.raw(), out.raw(), n, &stream)?;
            stream.synchronize()?;
            self.check()?;
            let got = out.download()?;
            for r in 0..self.world {
                let want: Vec<u32> = (0..words as u32).map(|i| i.wrapping_mul(2654435761) ^ (r as u32 + 1)).collect();
                if got[r * words..(r + 1) * words] != want[..] {
                    return Err(format!("roce: probe gather of {n} bytes returned wrong data for rank {r}").into());
                }
            }
        }
        ctl.send(b"roce-probed")?;
        if ctl.recv()? != b"roce-probed" {
            return Err("roce: the peer's probe failed".into());
        }
        Ok(())
    }

    /// out [world][nbytes] <- every rank's `nbytes` of `input`, in rank order, on `stream`
    /// (graph-capturable). nbytes <= max_bytes.
    pub fn gather(&self, input: *const c_void, out: *mut c_void, nbytes: usize, stream: &Stream) -> Result<()> {
        if nbytes == 0 || nbytes as u64 > self.slot_bytes {
            return Err(format!("roce: gather of {nbytes} bytes (slot {})", self.slot_bytes).into());
        }
        // tools/roce/roce.py::grid_blocks: a power-of-two grid, about two 16-byte packs a thread
        let need = nbytes.div_ceil(PACK).div_ceil(2 * THREADS as usize).max(1);
        let grid = need.next_power_of_two().min(MAX_BLOCKS);
        let lg = grid.trailing_zeros() as usize;
        let c = self.counters.raw() as *mut u32;
        let rc = unsafe {
            roce_gather(stream.raw(), input, out, nbytes as u64, grid as c_int, THREADS, self.dev + self.lay[0],
                        self.dev + self.lay[1], self.dev + self.lay[2], self.dev + self.lay[3], self.slot_bytes,
                        c as *mut c_void, c.add(1 + lg) as *mut c_void, c.add(17 + lg) as *mut c_void,
                        c.add(33) as *mut c_void, self.timeout_ns, self.world as c_int, self.rank as c_int,
                        self.n_hca as c_int, self.lay[5] as c_int, self.lay[6] as c_int)
        };
        if rc != 0 {
            return Err(format!("roce: gather launch failed ({rc})").into());
        }
        Ok(())
    }

    /// Fail-stop: a wait that timed out (recorded in the host-visible control record) or a
    /// proxy error poisons the runtime; report it.
    pub fn check(&self) -> Result<()> {
        let ctrl = unsafe { (self.host as *const u8).add(self.lay[3] as usize) as *const u32 };
        let word = |i: usize| unsafe { std::ptr::read_volatile(ctrl.add(i)) };
        if word(CTRL_FAILED) != 0 {
            return Err(format!("roce: rank {} timed out waiting for rank {} on HCA {} (sequence {})", self.rank,
                               word(CTRL_ERR_PEER), word(CTRL_ERR_HCA), word(CTRL_ERR_SEQ)).into());
        }
        let mut err = errbuf();
        if unsafe { roce_proxy_failed(self.proxy, err.as_mut_ptr(), err.len() as c_int) } != 0 {
            return Err(format!("roce proxy: {}", errstr(&err)).into());
        }
        Ok(())
    }
}

impl Drop for Roce {
    fn drop(&mut self) {
        unsafe {
            roce_proxy_destroy(self.proxy);
            cudaFreeHost(self.host);
        }
    }
}
