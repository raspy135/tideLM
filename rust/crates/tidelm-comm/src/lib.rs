//! The pair: NCCL for the collectives inside a step (stream-ordered, graph-capturable) and a
//! TCP control channel on which rank 0 sends rank 1 every engine call before running it
//! itself, so both ranks issue the same device work in the same order. Rank 1 decides
//! nothing: anything that could differ per rank (timings, policies) lives on rank 0 only
//! (CLAUDE.md, "The pair fails silently").

use std::ffi::{c_char, c_int, c_void, CStr};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use tidelm_cuda::Stream;

pub mod roce;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    internal: [c_char; 128],
}

type NcclComm = *mut c_void;
const NCCL_FLOAT32: c_int = 7;
const NCCL_SUM: c_int = 0;

extern "C" {
    fn ncclGetUniqueId(id: *mut NcclUniqueId) -> c_int;
    fn ncclCommInitRank(comm: *mut NcclComm, nranks: c_int, id: NcclUniqueId, rank: c_int) -> c_int;
    fn ncclAllReduce(send: *const c_void, recv: *mut c_void, count: usize, dtype: c_int, op: c_int, comm: NcclComm,
                     stream: *mut c_void) -> c_int;
    fn ncclCommDestroy(comm: NcclComm) -> c_int;
    fn ncclGetErrorString(rc: c_int) -> *const c_char;
}

fn nccl(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(ncclGetErrorString(rc)) }.to_string_lossy().into_owned();
    Err(format!("{what}: NCCL error {rc} ({msg})").into())
}

/// An NCCL communicator over the pair.
pub struct Comm {
    comm: NcclComm,
    pub rank: usize,
    pub world: usize,
}

unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    /// In-place fp32 sum over the pair, ordered on `stream` (and capturable into its graph).
    /// With two ranks every element is one a + b, the same bits on both.
    pub fn allreduce_f32(&self, buf: *mut c_void, count: usize, stream: &Stream) -> Result<()> {
        nccl(unsafe { ncclAllReduce(buf, buf, count, NCCL_FLOAT32, NCCL_SUM, self.comm, stream.raw()) }, "all-reduce")
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        unsafe { ncclCommDestroy(self.comm) };
    }
}

/// The TCP side of the pair: the unique-id exchange, the boot config check, then rank 0's
/// commands. Messages are length-prefixed.
pub struct Control {
    sock: TcpStream,
}

impl Control {
    pub fn send(&mut self, msg: &[u8]) -> Result<()> {
        self.sock.write_all(&(msg.len() as u32).to_le_bytes())?;
        self.sock.write_all(msg)?;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<Vec<u8>> {
        let mut n = [0u8; 4];
        self.sock.read_exact(&mut n)?;
        let mut msg = vec![0u8; u32::from_le_bytes(n) as usize];
        self.sock.read_exact(&mut msg)?;
        Ok(msg)
    }
}

/// Join the pair: rank 0 listens on `addr` (host:port), rank 1 connects (retrying for
/// `timeout`). Both send `config` - everything that decides numerics or control flow - and
/// refuse to start unless the other rank's matches byte for byte. Then NCCL comes up.
pub fn join(rank: usize, addr: &str, config: &[u8], timeout: Duration) -> Result<(Comm, Control)> {
    let world = 2;
    let sock = if rank == 0 {
        let listener = TcpListener::bind(addr)?;
        listener.accept()?.0
    } else {
        let t0 = Instant::now();
        loop {
            match TcpStream::connect(addr) {
                Ok(s) => break s,
                Err(e) if t0.elapsed() < timeout => {
                    let _ = e;
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => return Err(format!("rank 1: no rank 0 at {addr}: {e}").into()),
            }
        }
    };
    sock.set_nodelay(true)?;
    let mut ctl = Control { sock };
    ctl.send(config)?;
    let theirs = ctl.recv()?;
    if theirs != config {
        let first = config.iter().zip(&theirs).position(|(a, b)| a != b).unwrap_or(config.len().min(theirs.len()));
        return Err(format!("rank {rank}: the pair's boot configs differ (at byte {first} of {} / {}); refusing to \
                            start - two ranks that disagree emit different tokens without an error",
                           config.len(), theirs.len()).into());
    }
    let mut id = NcclUniqueId { internal: [0; 128] };
    if rank == 0 {
        nccl(unsafe { ncclGetUniqueId(&mut id) }, "unique id")?;
        let bytes: Vec<u8> = id.internal.iter().map(|&c| c as u8).collect();
        ctl.send(&bytes)?;
    } else {
        let bytes = ctl.recv()?;
        if bytes.len() != 128 {
            return Err("rank 1: malformed NCCL id".into());
        }
        for (d, s) in id.internal.iter_mut().zip(bytes) {
            *d = s as c_char;
        }
    }
    let mut comm: NcclComm = std::ptr::null_mut();
    nccl(unsafe { ncclCommInitRank(&mut comm, world as c_int, id, rank as c_int) }, "communicator")?;
    Ok((Comm { comm, rank, world }, ctl))
}

/// A little-endian message writer / reader for the control channel.
#[derive(Default)]
pub struct Msg(pub Vec<u8>);

impl Msg {
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32s(&mut self, v: &[u32]) -> &mut Self {
        self.u32(v.len() as u32);
        for &x in v {
            self.u32(x);
        }
        self
    }
}

pub struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, at: 0 }
    }
    pub fn u32(&mut self) -> Result<u32> {
        let b = self.buf.get(self.at..self.at + 4).ok_or("control message truncated")?;
        self.at += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64> {
        let b = self.buf.get(self.at..self.at + 8).ok_or("control message truncated")?;
        self.at += 8;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    pub fn u32s(&mut self) -> Result<Vec<u32>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.u32()).collect()
    }
}
