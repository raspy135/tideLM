//! End-to-end GPU parity: the engine's real `fp4_moe_cuda_scale_expand` kernel,
//! launched from Rust, must reproduce the scale planes that `tidelm_core::scales`
//! (the port of engine/scale_layout.py) packed — for every byte value, partial
//! blocks, and records padded to an audited capacity larger than their own.
//!
//! This is the first Rust-launched execution of engine object code; it proves the
//! whole no-Python chain: codec -> pinned/device memory -> kernel launch -> readback.

use tidelm_core::scales::{encode, layout};
use tidelm_cuda::{gpu_available, set_device, DeviceBuffer, Stream};
use tidelm_kernels::fp4::fp4_moe_cuda_scale_expand;

#[test]
fn gpu_expand_matches_the_ported_codec() {
    if !gpu_available() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    set_device(0).unwrap();
    let stream = Stream::new().unwrap();

    // Three slots of one plane shape, mixed content, capacity audited to the max.
    let size = 360usize; // 5 full blocks + one partial block of 40
    let mut planes: Vec<Vec<u8>> = vec![
        (0..size).map(|i| (i % 256) as u8).collect(),
        vec![7u8; size],
        {
            let mut state = 17u64;
            (0..size)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (state >> 33) as u8
                })
                .collect()
        },
    ];
    // and one plane whose values straddle base/base+1 so code bits dominate
    planes.push((0..size).map(|i| if i % 3 == 0 { 8 } else { 7 }).collect());

    let encoded: Vec<_> = planes.iter().map(|p| encode(p).unwrap()).collect();
    let capacity = encoded.iter().map(|e| e.exceptions.len()).max().unwrap();
    let spec = layout(size, capacity).unwrap();

    let mut packed = Vec::with_capacity(planes.len() * spec.stride);
    for e in &encoded {
        packed.extend(e.pack(capacity).unwrap());
    }

    let d_packed = DeviceBuffer::from_slice(&packed).unwrap();
    let d_out = DeviceBuffer::<u8>::zeroed(planes.len() * size).unwrap();
    let rc = unsafe {
        fp4_moe_cuda_scale_expand(
            stream.raw(),
            d_packed.raw(),
            d_out.raw(),
            size as i32,
            spec.stride as i32,
            planes.len() as i32,
        )
    };
    assert_eq!(rc, 0, "kernel launch failed");
    stream.synchronize().unwrap();

    let out = d_out.download().unwrap();
    for (slot, plane) in planes.iter().enumerate() {
        assert_eq!(
            &out[slot * size..(slot + 1) * size],
            plane.as_slice(),
            "slot {slot} diverged from the CPU codec"
        );
    }
}
