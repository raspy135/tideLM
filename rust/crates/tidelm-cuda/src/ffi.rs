//! Raw CUDA runtime API declarations — only what the engine uses. Hand-written rather
//! than generated: the surface is ~25 functions, and owning the declarations removes
//! any binding-crate version coupling to the box's CUDA (13.0 today).

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type cudaError_t = c_int;
pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;
pub type cudaGraph_t = *mut c_void;
pub type cudaGraphExec_t = *mut c_void;

pub const CUDA_SUCCESS: cudaError_t = 0;

pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub const STREAM_NON_BLOCKING: c_uint = 1;
pub const EVENT_DISABLE_TIMING: c_uint = 2;

/// cudaStreamCaptureMode
pub const CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const CAPTURE_MODE_RELAXED: c_int = 2;

extern "C" {
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> cudaError_t;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaStreamCreateWithFlags(stream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: c_uint,
    ) -> cudaError_t;

    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;

    pub fn cudaStreamBeginCapture(stream: cudaStream_t, mode: c_int) -> cudaError_t;
    pub fn cudaStreamEndCapture(stream: cudaStream_t, graph: *mut cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphInstantiate(
        exec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: u64,
    ) -> cudaError_t;
    pub fn cudaGraphLaunch(exec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphExecDestroy(exec: cudaGraphExec_t) -> cudaError_t;
}
