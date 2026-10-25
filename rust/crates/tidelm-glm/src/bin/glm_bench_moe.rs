//! Routed-expert decode GEMV sweep (exl3m_grouped) on one GPU at the GLM-5.3 rank shapes: gate+up
//! K 6144 x N 1024 a matrix, down K 2048 x N 3072, 3-bit trellis. The weights are random words
//! (the decode's time does not depend on their values) in a pool of 128 experts a matrix, and each
//! timed call routes to experts no recent call touched, so every read is DRAM-cold as in decode.
//! Rows R with U distinct experts as TensorFold's kit tabulates them (R 1 / U 8 ... R 8 / U 40).
//!
//!   glm_bench_moe

use tidelm_cuda::{DeviceBuffer, Event, Stream};
use tidelm_glm::Result;
use tidelm_kernels::exl3::*;

const POOL: usize = 128;
const BITS: usize = 3;
const TOPK: usize = 8;

fn main() -> Result<()> {
    tidelm_cuda::set_device(0)?;
    let st = Stream::new()?;
    let (h, il, kd, dn) = (6144usize, 1024usize, 2048usize, 3072usize);
    let words = |k: usize, n: usize| (k / 16) * (n / 16) * 8 * BITS;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = |n: usize| -> Vec<u32> {
        (0..n).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u32 }).collect()
    };
    let (wgu, wdn) = (words(h, il), words(kd, dn));
    let t1 = DeviceBuffer::<u32>::from_slice(&rnd(POOL * wgu))?;
    let t3 = DeviceBuffer::<u32>::from_slice(&rnd(POOL * wgu))?;
    let t2 = DeviceBuffer::<u32>::from_slice(&rnd(POOL * wdn))?;
    let ptrs = |b: &DeviceBuffer<u32>, w: usize| -> Result<DeviceBuffer<i64>> {
        Ok(DeviceBuffer::from_slice(&(0..POOL).map(|e| b.ptr_at(e * w) as i64).collect::<Vec<_>>())?)
    };
    let (t1p, t3p, t2p) = (ptrs(&t1, wgu)?, ptrs(&t3, wgu)?, ptrs(&t2, wdn)?);
    let k2 = DeviceBuffer::from_slice(&vec![2 * BITS as i32; POOL])?;
    let pmax = 8 * TOPK;
    let x0 = DeviceBuffer::<u16>::from_slice(&vec![0x3c00u16; pmax * h])?;
    let x1 = DeviceBuffer::<u16>::from_slice(&vec![0x3c00u16; pmax * h])?;
    let z = DeviceBuffer::<f32>::new(2 * 32 * pmax * dn)?;
    let pick = DeviceBuffer::<i32>::new(pmax)?;
    let (uids, ucount, members) = (DeviceBuffer::<i32>::new(pmax)?, DeviceBuffer::<i32>::new(1)?,
                                   DeviceBuffer::<i32>::new(pmax * MAX_MEMBERS as usize)?);
    let (e0, e1) = (Event::timing()?, Event::timing()?);
    let mut call = 0usize;
    println!("GB/s of trellis bytes read (median of 12 calls); nt x warps x pf, then split-K");
    for &(r, u) in &[(1usize, 8usize), (2, 12), (4, 21), (8, 40)] {
        let p = r * TOPK;
        for (name, mats, k, n, wq, tp0, tp1) in [("gate+up", 2, h, il, wgu, &t1p, &t3p), ("down", 1, kd, dn, wdn, &t2p, &t2p)] {
            let mut line = format!("R{r} U{u:>2} {name:>7}:");
            for &(nt, pf) in &[(8usize, 1usize), (8, 2), (4, 2)] {
                line += &format!("  nt{nt}pf{pf}");
                for sk in [1usize, 2, 4, 8] {
                    if (k / 16) % (sk * 4) != 0 {
                        continue;
                    }
                    let mut ts = vec![];
                    for i in 0..14 {
                        // U experts no call in the last POOL / U calls used; row j's picks overlap
                        let base = call * u;
                        call += 1;
                        let picks: Vec<i32> = (0..p).map(|q| ((base + q % u) % POOL) as i32).collect();
                        pick.upload(&picks)?;
                        unsafe {
                            exl3m_group(pick.raw(), uids.raw(), ucount.raw(), members.raw(), r as i32, TOPK as i32,
                                        POOL as i32, MAX_MEMBERS, st.raw());
                        }
                        e0.record(&st)?;
                        unsafe {
                            exl3m_grouped(x0.raw(), x1.raw(), tp0.raw(), tp1.raw(), k2.raw(), k2.raw(), uids.raw(),
                                          ucount.raw(), members.raw(), z.raw(), k as i32, n as i32, p as i32, sk as i32,
                                          MAX_MEMBERS, TOPK as i32, p as i32, mats, nt as i32, 4, pf as i32, 2, 10,
                                          CB_MUL1, st.raw());
                        }
                        e1.record(&st)?;
                        e1.synchronize()?;
                        if i >= 2 {
                            ts.push(e1.elapsed_ms(&e0)? as f64 * 1e3);
                        }
                    }
                    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let us = ts[ts.len() / 2];
                    let bytes = (u * mats as usize * wq * 4) as f64;
                    line += &format!(" s{sk} {us:.0}us/{:.0}", bytes / us / 1e3);
                }
            }
            println!("{line}");
        }
    }
    Ok(())
}
