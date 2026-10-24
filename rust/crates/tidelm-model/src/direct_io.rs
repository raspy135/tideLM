//! O_DIRECT file access and aligned host buffers, from `engine/experts.py`
//! (`ShardFile`, `_pread_chunk`): no page-cache pollution for the 288 GB of expert
//! data, ~5.5 GB/s with enough reads in flight on this box.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

pub const ALIGN: usize = 4096;

/// A file opened twice: O_DIRECT for the bulk reads, buffered for header parsing and
/// verification.
pub struct DirectFile {
    fd: i32,
    fd_buffered: i32,
}

// pread on a shared fd is thread-safe; the struct owns nothing mutable.
unsafe impl Sync for DirectFile {}

impl DirectFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let cpath = CString::new(path.as_os_str().as_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let fd_buffered = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
        if fd_buffered < 0 {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(e);
        }
        Ok(DirectFile { fd, fd_buffered })
    }

    /// O_DIRECT-read `need` bytes at `off` into `buf` (an ALIGN-aligned slice whose
    /// length is the aligned request size). The request length is always the full
    /// slice — O_DIRECT rejects unaligned lengths — and the loop stops as soon as the
    /// bytes that actually exist have arrived, which is what makes the aligned tail of
    /// the last tensor in a shard safe.
    pub fn pread_exact(&self, buf: &mut [u8], off: u64, need: usize) -> io::Result<()> {
        debug_assert!(need <= buf.len());
        debug_assert_eq!(buf.as_ptr() as usize % ALIGN, 0, "O_DIRECT needs an aligned buffer");
        debug_assert_eq!(off as usize % ALIGN, 0, "O_DIRECT needs an aligned offset");
        let mut got = 0usize;
        while got < need {
            let r = unsafe {
                libc::pread(
                    self.fd,
                    buf[got..].as_mut_ptr() as *mut libc::c_void,
                    buf.len() - got,
                    (off + got as u64) as libc::off_t,
                )
            };
            if r <= 0 {
                let err = if r == 0 {
                    io::Error::new(io::ErrorKind::UnexpectedEof, format!("short read at {off}+{got}/{need}"))
                } else {
                    io::Error::last_os_error()
                };
                return Err(err);
            }
            got += r as usize;
        }
        Ok(())
    }

    /// Ordinary buffered read, for headers and verification; no alignment demands.
    pub fn pread_buffered(&self, buf: &mut [u8], off: u64) -> io::Result<()> {
        let mut got = 0usize;
        while got < buf.len() {
            let r = unsafe {
                libc::pread(
                    self.fd_buffered,
                    buf[got..].as_mut_ptr() as *mut libc::c_void,
                    buf.len() - got,
                    (off + got as u64) as libc::off_t,
                )
            };
            if r <= 0 {
                return Err(if r == 0 {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "short buffered read")
                } else {
                    io::Error::last_os_error()
                });
            }
            got += r as usize;
        }
        Ok(())
    }
}

impl Drop for DirectFile {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
            libc::close(self.fd_buffered);
        }
    }
}

/// 4096-aligned host memory for O_DIRECT staging. Plain malloc-backed for now; the
/// serving store swaps this for pinned (cudaMallocHost) memory so the same buffer
/// feeds the H2D arena copy — the allocation seam is this type, nothing else changes.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> io::Result<Self> {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut ptr, ALIGN, len) };
        if rc != 0 {
            return Err(io::Error::from_raw_os_error(rc));
        }
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, len) };
        Ok(AlignedBuf { ptr: ptr as *mut u8, len })
    }
}

impl std::ops::Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr as *mut libc::c_void) };
    }
}
