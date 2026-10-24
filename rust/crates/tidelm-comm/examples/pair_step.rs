//! What the pair's per-layer all-reduce costs inside a step graph: 40 layers of
//! memory-bound work (a D2D copy) each followed by the MoE combine's all-reduce, captured
//! as one CUDA graph like the engine's step, timed against the same work without the
//! collective and with one rank given more work per layer (expert imbalance).
//!
//!   pair_step --rank 0|1 [--addr 10.0.0.1:29720] [--rows 1] [--layer-mb 90]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use tidelm_comm::join;
use tidelm_cuda::{ffi, set_device, DeviceBuffer, Stream};

fn main() -> tidelm_comm::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1).cloned());
    let rank: usize = arg("--rank").unwrap().parse()?;
    let addr = arg("--addr").unwrap_or_else(|| "10.0.0.1:29720".into());
    let rows: usize = arg("--rows").map(|v| v.parse().unwrap()).unwrap_or(1);
    let layer_mb: usize = arg("--layer-mb").map(|v| v.parse().unwrap()).unwrap_or(90);
    set_device(0)?;
    let stream = Stream::new()?;
    let (comm, _ctl) = join(rank, &addr, format!("pair-step-v1 rows {rows} mb {layer_mb}").as_bytes(), Duration::from_secs(60))?;
    let layers = 40;
    let bytes = layer_mb << 20;
    // one source per layer so every copy reads cold memory, like the step's weights
    let src: Vec<DeviceBuffer<u8>> = (0..layers).map(|_| DeviceBuffer::<u8>::zeroed(bytes)).collect::<Result<_, _>>()?;
    let dst = DeviceBuffer::<u8>::zeroed(bytes)?;
    let red = DeviceBuffer::<f32>::zeroed(rows * 5120)?;
    let step = |s: &Stream, comm_on: bool, extra: f64| -> tidelm_comm::Result<()> {
        for l in 0..layers {
            let n = (bytes as f64 * (1.0 + extra)) as usize;
            let n = n.min(bytes);
            let rc = unsafe { ffi::cudaMemcpyAsync(dst.raw(), src[l].raw() as *const c_void, n, ffi::MEMCPY_DEVICE_TO_DEVICE, s.raw()) };
            tidelm_cuda::check(rc, "copy")?;
            if comm_on {
                comm.allreduce_f32(red.raw(), rows * 5120, s)?;
            }
        }
        Ok(())
    };
    // (comm, extra work on rank 1 as a fraction of the layer)
    for (name, comm_on, extra) in [("work only", false, 0.0), ("work + all-reduce", true, 0.0),
                                   ("+ rank 1 does 1/3 less", true, -0.33), ("+ rank 1 does 1/3 more", true, 0.33)] {
        let mine = if rank == 1 { extra } else { 0.0 };
        stream.begin_capture()?;
        let rec = step(&stream, comm_on, mine);
        let g = stream.end_capture();
        rec?;
        let g = g?;
        for _ in 0..3 {
            g.launch(&stream)?;
        }
        stream.synchronize()?;
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            g.launch(&stream)?;
        }
        stream.synchronize()?;
        println!("rank {rank}: {name:26} {:.2} ms/step ({rows} rows, {layers} x {layer_mb} MB)", t.elapsed().as_secs_f64() / reps as f64 * 1e3);
    }
    Ok(())
}
