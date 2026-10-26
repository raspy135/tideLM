//! CUDA virtual memory management: physical chunks of the allocation granularity (2 MB on GB10)
//! mapped into address ranges reserved up front. The driver's cuMem* calls are reached through
//! cudaGetDriverEntryPointByVersion, so nothing links libcuda.
//!
//! What it buys a cache that grows: its address range is reserved whole at boot and kernels
//! address it exactly as they would a cudaMalloc buffer, while physical memory is mapped in a
//! chunk at a time as the contents grow. One chunk may be mapped at many addresses at once (the
//! engine's dummy chunk backs every range it has not paged in, so reads past a lane's end never
//! fault). A chunk's memory is the system's on the Spark, as cudaMalloc's is.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use crate::{check, CudaError, Result};

pub type Handle = u64;
pub type DevPtr = u64;

#[repr(C)]
#[derive(Clone, Copy)]
struct Location {
    kind: c_int,
    id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AllocFlags {
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
    reserved: [u8; 4],
}

/// CUmemAllocationProp
#[repr(C)]
#[derive(Clone, Copy)]
struct AllocProp {
    kind: c_int,
    handle_types: c_int,
    location: Location,
    win32: *mut c_void,
    flags: AllocFlags,
}

/// CUmemAccessDesc
#[repr(C)]
#[derive(Clone, Copy)]
struct AccessDesc {
    location: Location,
    flags: c_int,
}

const ALLOCATION_TYPE_PINNED: c_int = 1;
const LOCATION_TYPE_DEVICE: c_int = 1;
const ACCESS_READWRITE: c_int = 3;
const GRANULARITY_MINIMUM: c_int = 0;

extern "C" {
    fn cudaGetDriverEntryPointByVersion(symbol: *const c_char, func: *mut *mut c_void, cuda_version: c_uint, flags: u64,
                                        status: *mut c_int) -> crate::ffi::cudaError_t;
}

struct Fns {
    granularity: unsafe extern "C" fn(*mut usize, *const AllocProp, c_int) -> c_int,
    create: unsafe extern "C" fn(*mut Handle, usize, *const AllocProp, u64) -> c_int,
    release: unsafe extern "C" fn(Handle) -> c_int,
    reserve: unsafe extern "C" fn(*mut DevPtr, usize, usize, DevPtr, u64) -> c_int,
    free: unsafe extern "C" fn(DevPtr, usize) -> c_int,
    map: unsafe extern "C" fn(DevPtr, usize, usize, Handle, u64) -> c_int,
    unmap: unsafe extern "C" fn(DevPtr, usize) -> c_int,
    set_access: unsafe extern "C" fn(DevPtr, usize, *const AccessDesc, usize) -> c_int,
}

/// The driver calls, the device-0 allocation properties and their granularity.
pub struct Vmm {
    fns: Fns,
    prop: AllocProp,
    access: AccessDesc,
    /// bytes a chunk (cuMemGetAllocationGranularity, minimum)
    pub gran: usize,
}

unsafe impl Send for Vmm {}
unsafe impl Sync for Vmm {}

fn cu(code: c_int, op: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError { code, what: format!("CUresult {code}"), op })
    }
}

fn entry(name: &str) -> Result<*mut c_void> {
    let sym = CString::new(name).unwrap();
    let (mut f, mut status): (*mut c_void, c_int) = (std::ptr::null_mut(), 0);
    check(unsafe { cudaGetDriverEntryPointByVersion(sym.as_ptr(), &mut f, 12000, 0, &mut status) }, "cudaGetDriverEntryPointByVersion")?;
    if f.is_null() || status != 0 {
        return Err(CudaError { code: status, what: format!("driver entry point {name} not found"), op: "cudaGetDriverEntryPointByVersion" });
    }
    Ok(f)
}

impl Vmm {
    /// The process's one instance (device 0, which the engine sets before anything else).
    pub fn get() -> Result<&'static Vmm> {
        static V: OnceLock<std::result::Result<Vmm, CudaError>> = OnceLock::new();
        V.get_or_init(Vmm::load).as_ref().map_err(|e| e.clone())
    }

    fn load() -> Result<Vmm> {
        unsafe {
            let fns = Fns {
                granularity: std::mem::transmute(entry("cuMemGetAllocationGranularity")?),
                create: std::mem::transmute(entry("cuMemCreate")?),
                release: std::mem::transmute(entry("cuMemRelease")?),
                reserve: std::mem::transmute(entry("cuMemAddressReserve")?),
                free: std::mem::transmute(entry("cuMemAddressFree")?),
                map: std::mem::transmute(entry("cuMemMap")?),
                unmap: std::mem::transmute(entry("cuMemUnmap")?),
                set_access: std::mem::transmute(entry("cuMemSetAccess")?),
            };
            let location = Location { kind: LOCATION_TYPE_DEVICE, id: 0 };
            let prop = AllocProp {
                kind: ALLOCATION_TYPE_PINNED, handle_types: 0, location, win32: std::ptr::null_mut(),
                flags: AllocFlags { compression_type: 0, gpu_direct_rdma_capable: 0, usage: 0, reserved: [0; 4] },
            };
            let mut gran = 0usize;
            cu((fns.granularity)(&mut gran, &prop, GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity")?;
            Ok(Vmm { fns, prop, access: AccessDesc { location, flags: ACCESS_READWRITE }, gran })
        }
    }

    /// One physical chunk of `gran` bytes (counted as device memory).
    #[track_caller]
    pub fn create(&self) -> Result<Handle> {
        let mut h: Handle = 0;
        cu(unsafe { (self.fns.create)(&mut h, self.gran, &self.prop, 0) }, "cuMemCreate")?;
        crate::mem::alloc("device memory (vmm)", &crate::mem::DEVICE, self.gran, std::panic::Location::caller());
        Ok(h)
    }

    /// Give a chunk back once nothing maps it.
    pub fn release(&self, h: Handle) -> Result<()> {
        crate::mem::DEVICE.fetch_sub(self.gran, std::sync::atomic::Ordering::Relaxed);
        cu(unsafe { (self.fns.release)(h) }, "cuMemRelease")
    }

    /// An address range of `bytes` (a multiple of `gran`), nothing mapped.
    pub fn reserve(&self, bytes: usize) -> Result<DevPtr> {
        let mut p: DevPtr = 0;
        cu(unsafe { (self.fns.reserve)(&mut p, bytes, self.gran, 0, 0) }, "cuMemAddressReserve")?;
        Ok(p)
    }

    pub fn free(&self, va: DevPtr, bytes: usize) -> Result<()> {
        cu(unsafe { (self.fns.free)(va, bytes) }, "cuMemAddressFree")
    }

    /// Map chunk `h` at `va` (gran-aligned), read-write for the device.
    pub fn map(&self, va: DevPtr, h: Handle) -> Result<()> {
        cu(unsafe { (self.fns.map)(va, self.gran, 0, h, 0) }, "cuMemMap")?;
        cu(unsafe { (self.fns.set_access)(va, self.gran, &self.access, 1) }, "cuMemSetAccess")
    }

    /// Unmap the chunk at `va`. Nothing may be reading or writing it.
    pub fn unmap(&self, va: DevPtr) -> Result<()> {
        cu(unsafe { (self.fns.unmap)(va, self.gran) }, "cuMemUnmap")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gpu_available, set_device, DeviceBuffer};

    #[test]
    fn a_chunk_maps_at_many_addresses_and_a_remap_isolates_one() {
        if !gpu_available() {
            eprintln!("no CUDA device; skipping");
            return;
        }
        set_device(0).unwrap();
        let v = Vmm::get().unwrap();
        let g = v.gran;
        let (dummy, real) = (v.create().unwrap(), v.create().unwrap());
        let va = v.reserve(4 * g).unwrap();
        for i in 0..4 {
            v.map(va + (i * g) as u64, dummy).unwrap();
        }
        // one chunk behind four addresses: a write through the first reads back through the last
        let src = DeviceBuffer::from_slice(&[7u32, 8, 9]).unwrap();
        let copy = |dst: u64, src: *const c_void, n: usize| unsafe {
            check(crate::ffi::cudaMemcpy(dst as *mut c_void, src, n, crate::ffi::MEMCPY_DEVICE_TO_DEVICE), "copy").unwrap()
        };
        copy(va, src.raw(), 12);
        let back = DeviceBuffer::<u32>::zeroed(3).unwrap();
        copy(back.raw() as u64, (va + 3 * g as u64) as *const c_void, 12);
        assert_eq!(back.download().unwrap(), vec![7, 8, 9]);
        // the second address gets a chunk of its own: writes there leave the dummy alone
        v.unmap(va + g as u64).unwrap();
        v.map(va + g as u64, real).unwrap();
        let other = DeviceBuffer::from_slice(&[1u32, 2, 3]).unwrap();
        copy(va + g as u64, other.raw(), 12);
        copy(back.raw() as u64, (va + 2 * g as u64) as *const c_void, 12);
        assert_eq!(back.download().unwrap(), vec![7, 8, 9]);
        copy(back.raw() as u64, (va + g as u64) as *const c_void, 12);
        assert_eq!(back.download().unwrap(), vec![1, 2, 3]);
        for i in 0..4 {
            v.unmap(va + (i * g) as u64).unwrap();
        }
        v.free(va, 4 * g).unwrap();
        v.release(dummy).unwrap();
        v.release(real).unwrap();
    }
}
