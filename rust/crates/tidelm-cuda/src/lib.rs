//! Thin CUDA runtime layer for the engine: errors, streams, device/pinned buffers,
//! events, and graph capture/replay.
//!
//! Deliberately minimal. The engine's performance lives in what gets captured into
//! graphs and how little the host does per step — not in this wrapper — so the wrapper
//! adds nothing on the hot path: a [`Stream`] is one pointer, a launch is one FFI call.
//! Everything returns [`CudaError`] rather than panicking; the engine's boot guard
//! decides what is fatal.

pub mod ffi;

use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::raw::c_void;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError {
    pub code: i32,
    pub what: String,
    pub op: &'static str,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: CUDA error {} ({})", self.op, self.code, self.what)
    }
}
impl std::error::Error for CudaError {}

pub type Result<T> = std::result::Result<T, CudaError>;

/// Convert a runtime return code; `op` names the call for the error message.
pub fn check(code: ffi::cudaError_t, op: &'static str) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        return Ok(());
    }
    let what = unsafe {
        let s = ffi::cudaGetErrorString(code);
        if s.is_null() { "unknown".into() } else { CStr::from_ptr(s).to_string_lossy().into_owned() }
    };
    Err(CudaError { code, what, op })
}

pub fn device_count() -> Result<i32> {
    let mut n = 0;
    check(unsafe { ffi::cudaGetDeviceCount(&mut n) }, "cudaGetDeviceCount")?;
    Ok(n)
}

pub fn set_device(device: i32) -> Result<()> {
    check(unsafe { ffi::cudaSetDevice(device) }, "cudaSetDevice")
}

pub fn device_synchronize() -> Result<()> {
    check(unsafe { ffi::cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// (free, total) device memory in bytes. On GB10 this is the unified pool and counts
/// page cache as used; the engine sizes its arena from /proc/meminfo instead (see
/// `v41_engine.py` around the arena sizing), so treat this as a lower bound only.
pub fn mem_get_info() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0usize, 0usize);
    check(unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok((free, total))
}

/// An owned, non-blocking CUDA stream.
pub struct Stream(ffi::cudaStream_t);

// Streams are externally synchronized handles; the engine's step loop owns ordering.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut s: ffi::cudaStream_t = std::ptr::null_mut();
        check(
            unsafe { ffi::cudaStreamCreateWithFlags(&mut s, ffi::STREAM_NON_BLOCKING) },
            "cudaStreamCreateWithFlags",
        )?;
        Ok(Stream(s))
    }

    /// The raw handle, for kernel launchers that take `void* stream`.
    pub fn raw(&self) -> *mut c_void {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::cudaStreamSynchronize(self.0) }, "cudaStreamSynchronize")
    }

    pub fn wait_event(&self, event: &Event) -> Result<()> {
        check(unsafe { ffi::cudaStreamWaitEvent(self.0, event.0, 0) }, "cudaStreamWaitEvent")
    }

    /// Start recording this stream's work into a graph (see `end_capture`). Split from
    /// `capture` for callers whose recording code needs `&mut` access to the stream's owner.
    pub fn begin_capture(&self) -> Result<()> {
        check(
            unsafe { ffi::cudaStreamBeginCapture(self.0, ffi::CAPTURE_MODE_THREAD_LOCAL) },
            "cudaStreamBeginCapture",
        )
    }

    /// Stop recording and instantiate. Always call after `begin_capture`, even when the
    /// recording failed, so the stream leaves capture mode.
    pub fn end_capture(&self) -> Result<GraphExec> {
        let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
        check(unsafe { ffi::cudaStreamEndCapture(self.0, &mut graph) }, "cudaStreamEndCapture")?;
        let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
        let instantiated =
            check(unsafe { ffi::cudaGraphInstantiate(&mut exec, graph, 0) }, "cudaGraphInstantiate");
        unsafe { ffi::cudaGraphDestroy(graph) };
        instantiated?;
        Ok(GraphExec(exec))
    }

    /// Capture everything queued on this stream inside `record` into an executable graph.
    pub fn capture<F: FnOnce(&Stream) -> Result<()>>(&self, record: F) -> Result<GraphExec> {
        check(
            unsafe { ffi::cudaStreamBeginCapture(self.0, ffi::CAPTURE_MODE_THREAD_LOCAL) },
            "cudaStreamBeginCapture",
        )?;
        // On a record error, end the capture anyway so the stream is reusable.
        let recorded = record(self);
        let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
        let ended = check(
            unsafe { ffi::cudaStreamEndCapture(self.0, &mut graph) },
            "cudaStreamEndCapture",
        );
        recorded?;
        ended?;
        let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
        let instantiated =
            check(unsafe { ffi::cudaGraphInstantiate(&mut exec, graph, 0) }, "cudaGraphInstantiate");
        unsafe { ffi::cudaGraphDestroy(graph) };
        instantiated?;
        Ok(GraphExec(exec))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { ffi::cudaStreamDestroy(self.0) };
    }
}

/// An instantiated, replayable CUDA graph.
pub struct GraphExec(ffi::cudaGraphExec_t);

unsafe impl Send for GraphExec {}

impl GraphExec {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaGraphLaunch(self.0, stream.raw()) }, "cudaGraphLaunch")
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe { ffi::cudaGraphExecDestroy(self.0) };
    }
}

/// An owned CUDA event (timing disabled: the engine times with host clocks and the
/// profiler, and timing events serialize).
pub struct Event(ffi::cudaEvent_t);

unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut e: ffi::cudaEvent_t = std::ptr::null_mut();
        check(
            unsafe { ffi::cudaEventCreateWithFlags(&mut e, ffi::EVENT_DISABLE_TIMING) },
            "cudaEventCreateWithFlags",
        )?;
        Ok(Event(e))
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaEventRecord(self.0, stream.raw()) }, "cudaEventRecord")
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::cudaEventSynchronize(self.0) }, "cudaEventSynchronize")
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { ffi::cudaEventDestroy(self.0) };
    }
}

/// Device memory holding `len` elements of `T`. `T` must be plain data (the crate only
/// instantiates it with primitives).
pub struct DeviceBuffer<T: Copy> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy> Send for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn new(len: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::cudaMalloc(&mut ptr, len * size_of::<T>()) }, "cudaMalloc")?;
        Ok(DeviceBuffer { ptr, len, _marker: PhantomData })
    }

    pub fn zeroed(len: usize) -> Result<Self> {
        let buf = Self::new(len)?;
        check(unsafe { ffi::cudaMemset(buf.ptr, 0, len * size_of::<T>()) }, "cudaMemset")?;
        Ok(buf)
    }

    pub fn from_slice(data: &[T]) -> Result<Self> {
        let buf = Self::new(data.len())?;
        buf.upload(data)?;
        Ok(buf)
    }

    pub fn upload(&self, data: &[T]) -> Result<()> {
        assert!(data.len() <= self.len);
        check(
            unsafe {
                ffi::cudaMemcpy(
                    self.ptr,
                    data.as_ptr() as *const c_void,
                    std::mem::size_of_val(data),
                    ffi::MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy H2D",
        )
    }

    pub fn download(&self) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(self.len);
        check(
            unsafe {
                ffi::cudaMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    self.len * size_of::<T>(),
                    ffi::MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy D2H",
        )?;
        unsafe { out.set_len(self.len) };
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for kernel launchers.
    pub fn raw(&self) -> *mut c_void {
        self.ptr
    }

    /// Device pointer to element `elem` (bounds-checked).
    pub fn ptr_at(&self, elem: usize) -> *mut c_void {
        assert!(elem <= self.len, "device offset {elem} past length {}", self.len);
        unsafe { (self.ptr as *mut u8).add(elem * size_of::<T>()) as *mut c_void }
    }

    /// Host -> device into `[elem, elem + src.len())`, ordered on `stream`. From pageable
    /// memory this returns once the source is consumed; from pinned memory it is async,
    /// and the caller keeps `src` alive until the stream passes this point.
    pub fn upload_async(&self, elem: usize, src: &[T], stream: &Stream) -> Result<()> {
        assert!(elem + src.len() <= self.len, "upload past the end of the buffer");
        unsafe {
            memcpy_async(self.ptr_at(elem), src.as_ptr() as *const c_void, std::mem::size_of_val(src),
                         ffi::MEMCPY_HOST_TO_DEVICE, stream)
        }
    }

    /// Device -> host from `[elem, elem + dst.len())`, ordered on `stream`; the data is
    /// valid once the stream is synchronized.
    pub fn download_async(&self, elem: usize, dst: &mut [T], stream: &Stream) -> Result<()> {
        assert!(elem + dst.len() <= self.len, "download past the end of the buffer");
        unsafe {
            memcpy_async(dst.as_mut_ptr() as *mut c_void, self.ptr_at(elem), std::mem::size_of_val(dst),
                         ffi::MEMCPY_DEVICE_TO_HOST, stream)
        }
    }

    /// Device -> device copy of `n` elements, ordered on `stream`.
    pub fn copy_from(&self, elem: usize, src: &DeviceBuffer<T>, src_elem: usize, n: usize, stream: &Stream) -> Result<()> {
        assert!(elem + n <= self.len && src_elem + n <= src.len, "device copy out of bounds");
        unsafe { memcpy_async(self.ptr_at(elem), src.ptr_at(src_elem), n * size_of::<T>(), ffi::MEMCPY_DEVICE_TO_DEVICE, stream) }
    }

    pub fn zero_async(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaMemsetAsync(self.ptr, 0, self.len * size_of::<T>(), stream.raw()) }, "cudaMemsetAsync")
    }

    /// Fill with a byte pattern (0xff makes every i32 -1).
    pub fn fill_bytes_async(&self, byte: u8, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaMemsetAsync(self.ptr, byte as i32, self.len * size_of::<T>(), stream.raw()) },
              "cudaMemsetAsync")
    }
}

/// Raw async copy between any two addresses (pinned host <-> device in the hot paths).
///
/// # Safety
/// `dst` and `src` must be valid for `bytes` in the memory spaces `kind` names, and stay
/// valid until `stream` passes this point.
pub unsafe fn memcpy_async(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: &Stream) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check(unsafe { ffi::cudaMemcpyAsync(dst, src, bytes, kind, stream.raw()) }, "cudaMemcpyAsync")
}

pub use ffi::{MEMCPY_DEVICE_TO_DEVICE, MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE};

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe { ffi::cudaFree(self.ptr) };
    }
}

/// Page-locked host memory, for H2D/D2H on the hot path (the engram upload, the
/// verify readback). Dereferences to a slice.
pub struct PinnedBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Copy> Send for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    pub fn zeroed(len: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::cudaMallocHost(&mut ptr, len * size_of::<T>()) }, "cudaMallocHost")?;
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, len * size_of::<T>()) };
        Ok(PinnedBuffer { ptr: ptr as *mut T, len })
    }

    pub fn raw(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}

impl PinnedBuffer<u8> {
    /// Allocate pinned memory without touching it (cudaMallocHost pages are already
    /// page-aligned, which O_DIRECT needs); contents are unspecified.
    pub fn uninit(len: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::cudaMallocHost(&mut ptr, len) }, "cudaMallocHost")?;
        Ok(PinnedBuffer { ptr: ptr as *mut u8, len })
    }
}

impl<T: Copy> std::ops::Deref for PinnedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> std::ops::DerefMut for PinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        unsafe { ffi::cudaFreeHost(self.ptr as *mut c_void) };
    }
}

/// True when at least one CUDA device is usable; GPU tests skip quietly otherwise.
pub fn gpu_available() -> bool {
    device_count().map(|n| n > 0).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_graph_replay() {
        if !gpu_available() {
            eprintln!("no CUDA device; skipping");
            return;
        }
        set_device(0).unwrap();
        let stream = Stream::new().unwrap();
        let data: Vec<u32> = (0..4096).collect();
        let buf = DeviceBuffer::from_slice(&data).unwrap();
        assert_eq!(buf.download().unwrap(), data);

        // Capture a D2D copy into a graph and replay it twice.
        let dst = DeviceBuffer::<u32>::zeroed(4096).unwrap();
        let exec = stream
            .capture(|s| {
                check(
                    unsafe {
                        ffi::cudaMemcpyAsync(
                            dst.raw(),
                            buf.raw(),
                            4096 * 4,
                            ffi::MEMCPY_DEVICE_TO_DEVICE,
                            s.raw(),
                        )
                    },
                    "cudaMemcpyAsync D2D",
                )
            })
            .unwrap();
        exec.launch(&stream).unwrap();
        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(dst.download().unwrap(), data);
    }
}
