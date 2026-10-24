//! Two-box check: an all-reduce with known values, then the latency of the all-reduces a
//! step issues (the MoE combine at 1, 6 and 48 rows of 5120 fp32).
//!
//!   pair_smoke --rank 0 --addr 10.0.0.1:29700      (this box)
//!   pair_smoke --rank 1 --addr 10.0.0.1:29700      (the peer)

use std::time::{Duration, Instant};

use tidelm_comm::join;
use tidelm_cuda::{set_device, DeviceBuffer, Stream};

fn main() -> tidelm_comm::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1).cloned());
    let rank: usize = arg("--rank").unwrap().parse()?;
    let addr = arg("--addr").unwrap_or_else(|| "10.0.0.1:29700".into());
    set_device(0)?;
    let stream = Stream::new()?;
    let (comm, _ctl) = join(rank, &addr, b"pair-smoke-v1", Duration::from_secs(60))?;
    let n = 48 * 5120;
    let v: Vec<f32> = (0..n).map(|i| (rank + 1) as f32 + (i % 1000) as f32 * 0.5).collect();
    let buf = DeviceBuffer::from_slice(&v)?;
    comm.allreduce_f32(buf.raw(), n, &stream)?;
    stream.synchronize()?;
    let got = buf.download()?;
    let bad = got.iter().enumerate().filter(|(i, &x)| x != 3.0 + (i % 1000) as f32 * 1.0).count();
    println!("rank {rank}: all-reduce of {n} floats, {bad} wrong");
    for rows in [1usize, 6, 48] {
        let count = rows * 5120;
        let reps = 400;
        stream.synchronize()?;
        let t = Instant::now();
        for _ in 0..reps {
            comm.allreduce_f32(buf.raw(), count, &stream)?;
        }
        stream.synchronize()?;
        println!("rank {rank}: {rows:2} rows ({:6} KB): {:.1} us per all-reduce", count * 4 / 1024,
                 t.elapsed().as_secs_f64() / reps as f64 * 1e6);
    }
    Ok(())
}
