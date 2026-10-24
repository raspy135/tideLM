//! Typed FFI to the engine's native kernels, statically linked from the same sources
//! the Python engine JIT-compiles.
//!
//! - `fp4`: the decode MoE launchers from `tools/fp4_moe_cuda.cu`. Plain CUDA C++ with a
//!   C ABI: every launcher takes the CUDA stream first, returns `cudaGetLastError()` as
//!   an `int`, and is CUDA-graph capturable. The pointers are raw device pointers; the
//!   layouts they must obey are the arena layouts documented in `tools/fp4_moe.py`
//!   (w1/w3 `[slots, 1152, 2560]` u8, w2 `[slots, 2560, 1152]` u8, native or packed
//!   scale planes).
//! - `engram`: the CPU row gather from `engine/engram_gather.cpp` (memmap or O_DIRECT),
//!   used for the per-step Engram table reads. Thread-pool based; `run` blocks and may
//!   be called from a blocking executor.
//!
//! This crate deliberately stays `unsafe` and thin: ownership of device memory, streams
//! and graphs lives in the engine crate above it. Keeping the boundary at the exact C
//! ABI means a kernel change on the Python side is a build change here, never a silent
//! semantic drift.

use std::os::raw::{c_int, c_void};

/// A raw `cudaStream_t`.
pub type CudaStream = *mut c_void;

/// The decode MoE launchers. Signatures mirror `tools/fp4_moe_cuda.py::_library`
/// argtypes one for one; see `tools/fp4_moe_cuda.cu` lines 987+ for the definitions.
pub mod fp4 {
    use super::*;

    extern "C" {
        /// v1 up-projection (gate+up, SwiGLU input `h`): BM=16 grouped-by-block pairs.
        pub fn fp4_moe_cuda_up(
            stream: CudaStream,
            x: *const c_void,
            w1: *const c_void,
            s1: *const c_void,
            w3: *const c_void,
            s3: *const c_void,
            h: *mut c_void,
            route_weight: *const c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_x: i64,
            stride_h: i64,
            limit: f32,
            topk: c_int,
            n: c_int,
            k: c_int,
            nb: c_int,
            null_slot: c_int,
        ) -> c_int;

        /// v2 up (16-byte group loads, row kept in registers): same bits as v1 relaxed.
        pub fn fp4_moe_cuda_up_v2(
            stream: CudaStream,
            x: *const c_void,
            w1: *const c_void,
            s1: *const c_void,
            w3: *const c_void,
            s3: *const c_void,
            h: *mut c_void,
            route_weight: *const c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_x: i64,
            stride_h: i64,
            limit: f32,
            topk: c_int,
            n: c_int,
            k: c_int,
            nb: c_int,
            null_slot: c_int,
        ) -> c_int;

        /// v2 up reading packed lossless scale records (`scale_stride` > 0).
        pub fn fp4_moe_cuda_up_scales(
            stream: CudaStream,
            x: *const c_void,
            w1: *const c_void,
            s1: *const c_void,
            w3: *const c_void,
            s3: *const c_void,
            h: *mut c_void,
            route_weight: *const c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_x: i64,
            stride_h: i64,
            limit: f32,
            topk: c_int,
            n: c_int,
            k: c_int,
            nb: c_int,
            null_slot: c_int,
            scale_stride: c_int,
        ) -> c_int;

        pub fn fp4_moe_cuda_down(
            stream: CudaStream,
            h: *const c_void,
            w2: *const c_void,
            s2: *const c_void,
            y: *mut c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_h: i64,
            stride_y: i64,
            topk: c_int,
            n: c_int,
            k: c_int,
            ntok: c_int,
            nb: c_int,
            null_slot: c_int,
            partial: c_int,
            parts_world: c_int,
        ) -> c_int;

        pub fn fp4_moe_cuda_down_v2(
            stream: CudaStream,
            h: *const c_void,
            w2: *const c_void,
            s2: *const c_void,
            y: *mut c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_h: i64,
            stride_y: i64,
            topk: c_int,
            n: c_int,
            k: c_int,
            ntok: c_int,
            nb: c_int,
            null_slot: c_int,
            partial: c_int,
            parts_world: c_int,
        ) -> c_int;

        /// v3 down: `rows_per_warp` in {1, 2, 4}; same bits as v2.
        pub fn fp4_moe_cuda_down_v3(
            stream: CudaStream,
            h: *const c_void,
            w2: *const c_void,
            s2: *const c_void,
            y: *mut c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_h: i64,
            stride_y: i64,
            topk: c_int,
            n: c_int,
            k: c_int,
            ntok: c_int,
            nb: c_int,
            null_slot: c_int,
            partial: c_int,
            parts_world: c_int,
            rows_per_warp: c_int,
        ) -> c_int;

        /// v3 down reading packed lossless scale records.
        pub fn fp4_moe_cuda_down_scales(
            stream: CudaStream,
            h: *const c_void,
            w2: *const c_void,
            s2: *const c_void,
            y: *mut c_void,
            block_slot: *const c_void,
            block_pair: *const c_void,
            stride_h: i64,
            stride_y: i64,
            topk: c_int,
            n: c_int,
            k: c_int,
            ntok: c_int,
            nb: c_int,
            null_slot: c_int,
            partial: c_int,
            parts_world: c_int,
            rows_per_warp: c_int,
            scale_stride: c_int,
        ) -> c_int;

        /// Build BM=16 routing blocks on device for 1..512 (slot, token) pairs.
        pub fn fp4_moe_cuda_route_small(
            stream: CudaStream,
            slots: *const c_void,
            block_slot: *mut c_void,
            block_pair: *mut c_void,
            block_route: *mut c_void,
            pairs: c_int,
            topk: c_int,
        ) -> c_int;

        /// Reduce per-rank partial k-sums (EP parity mode's round combine).
        pub fn fp4_moe_cuda_round_reduce(
            stream: CudaStream,
            parts: *const c_void,
            out: *mut c_void,
            td: c_int,
            topk: c_int,
            out_bf16: c_int,
        ) -> c_int;

        /// Expand a packed lossless scale record into a native scale plane.
        pub fn fp4_moe_cuda_scale_expand(
            stream: CudaStream,
            packed: *const c_void,
            out: *mut c_void,
            a: c_int,
            b: c_int,
            c: c_int,
        ) -> c_int;
    }
}

/// The Engram CPU row gather (`engine/engram_gather.cpp`). Rows are 264 bytes:
/// 256 FP8 values + 8 scale bytes, gathered by absolute row id from either a mapped
/// table (`create`) or an O_DIRECT fd (`create_direct`).
pub mod engram {
    use super::*;

    extern "C" {
        /// Returns an opaque gather context over mapped weight/scale planes, or null on
        /// invalid arguments. `threads` in 1..=128.
        pub fn engram_gather_create(
            w: *const u8,
            s: *const u8,
            rows: i64,
            threads: c_int,
        ) -> *mut c_void;

        /// O_DIRECT variant: reads 512-byte sectors from `fd` at the given plane offsets.
        pub fn engram_gather_create_direct(
            fd: c_int,
            w_off: i64,
            s_off: i64,
            rows: i64,
            threads: c_int,
        ) -> *mut c_void;

        /// Gather `count` rows into `output` (`count * 264` bytes). Returns 0 on
        /// success; blocks until every row landed.
        pub fn engram_gather_run(
            ctx: *mut c_void,
            ids: *const i64,
            count: usize,
            output: *mut u8,
        ) -> c_int;

        pub fn engram_gather_destroy(ctx: *mut c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Link-time proof that every symbol the engine needs resolves from the static
    // archive. Runtime behavior is covered by parity tests against the Python engine
    // (the kernels are the same object code either way).
    #[test]
    fn symbols_link() {
        let fns: [*const (); 12] = [
            fp4::fp4_moe_cuda_up as *const (),
            fp4::fp4_moe_cuda_up_v2 as *const (),
            fp4::fp4_moe_cuda_up_scales as *const (),
            fp4::fp4_moe_cuda_down as *const (),
            fp4::fp4_moe_cuda_down_v2 as *const (),
            fp4::fp4_moe_cuda_down_v3 as *const (),
            fp4::fp4_moe_cuda_down_scales as *const (),
            fp4::fp4_moe_cuda_route_small as *const (),
            fp4::fp4_moe_cuda_round_reduce as *const (),
            fp4::fp4_moe_cuda_scale_expand as *const (),
            engram::engram_gather_create as *const (),
            engram::engram_gather_run as *const (),
        ];
        assert!(fns.iter().all(|f| !f.is_null()));
    }

    // The CPU gather is fully exercisable without a GPU: build a tiny in-memory table
    // and check the gathered bytes land row for row.
    #[test]
    fn engram_gather_reads_rows() {
        const ROWS: usize = 64;
        let w: Vec<u8> = (0..ROWS * 256).map(|i| (i % 251) as u8).collect();
        let s: Vec<u8> = (0..ROWS * 8).map(|i| (i % 13) as u8).collect();
        unsafe {
            let ctx = engram::engram_gather_create(w.as_ptr(), s.as_ptr(), ROWS as i64, 4);
            assert!(!ctx.is_null());
            let ids: Vec<i64> = vec![0, 7, 63, 7];
            let mut out = vec![0u8; ids.len() * 264];
            let rc = engram::engram_gather_run(ctx, ids.as_ptr(), ids.len(), out.as_mut_ptr());
            assert_eq!(rc, 0);
            for (j, &id) in ids.iter().enumerate() {
                let row = &out[j * 264..(j + 1) * 264];
                assert_eq!(&row[..256], &w[id as usize * 256..(id as usize + 1) * 256]);
                assert_eq!(&row[256..], &s[id as usize * 8..(id as usize + 1) * 8]);
            }
            engram::engram_gather_destroy(ctx);
        }
    }
}
