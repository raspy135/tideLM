//! Routed-expert decode GEMV sweep (exl3m_grouped) on one GPU at the GLM-5.3 rank shapes: gate+up
//! K 6144 x N 1024 a matrix, down K 2048 x N 3072, 3-bit trellis. The weights are random words
//! (the decode's time does not depend on their values) in a pool of 128 experts a matrix, and each
//! timed call routes to experts no recent call touched, so every read is DRAM-cold as in decode.
//! Rows R with U distinct experts as TensorFold's kit tabulates them (R 1 / U 8 ... R 8 / U 40).
//!
//!
//! --prefill: the prefill expert GEMM instead (exl3m_prefill_smem_mode at the engine's tile and
//! cp.async depth, 4x1x8 mode -4), one MoE sub-chunk of 2,048 rows x top-8 spread uniformly over
//! the pool, gate+up then down; ms a call and GB/s of trellis bytes (the median of 8).
//!
//!   glm_bench_moe [--prefill]

use tidelm_cuda::{DeviceBuffer, Event, Stream};
use tidelm_glm::Result;
use tidelm_kernels::exl3::*;

const POOL: usize = 128;
const TOPK: usize = 8;

fn main() -> Result<()> {
    tidelm_cuda::set_device(0)?;
    let st = Stream::new()?;
    // BITS=2..4: the trellis width (default 3): equal times at every width mean the decode math, not
    // the bytes, bounds the call
    let bits: usize = std::env::var("BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let (h, il, kd, dn) = (6144usize, 1024usize, 2048usize, 3072usize);
    let words = |k: usize, n: usize| (k / 16) * (n / 16) * 8 * bits;
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
    let k2 = DeviceBuffer::from_slice(&vec![2 * bits as i32; POOL])?;
    let pmax = 8 * TOPK;
    let x0 = DeviceBuffer::<u16>::from_slice(&vec![0x3c00u16; pmax * h])?;
    let x1 = DeviceBuffer::<u16>::from_slice(&vec![0x3c00u16; pmax * h])?;
    let z = DeviceBuffer::<f32>::new(2 * 32 * pmax * dn)?;
    let pick = DeviceBuffer::<i32>::new(pmax)?;
    let (uids, ucount, members) = (DeviceBuffer::<i32>::new(pmax)?, DeviceBuffer::<i32>::new(1)?,
                                   DeviceBuffer::<i32>::new(pmax * MAX_MEMBERS as usize)?);
    let (e0, e1) = (Event::timing()?, Event::timing()?);
    if std::env::args().any(|a| a == "--prefill") {
        return prefill(&st, &t1p, &t3p, &t2p, &k2, (h, il, kd, dn), (wgu, wdn), &e0, &e1);
    }
    let mut call = 0usize;
    println!("GB/s of trellis bytes read (median of 12 calls); nt x warps x pf, then split-K");
    for &(r, u) in &[(1usize, 8usize), (2, 12), (4, 21), (8, 40)] {
        let p = r * TOPK;
        for (name, mats, k, n, wq, tp0, tp1) in [("gate+up", 2, h, il, wgu, &t1p, &t3p), ("down", 1, kd, dn, wdn, &t2p, &t2p)] {
            let mut line = format!("R{r} U{u:>2} {name:>7}:");
            for &(nt, pf) in &[(8usize, 1i32), (8, -2), (8, -3)] {
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
                                          MAX_MEMBERS, TOPK as i32, p as i32, mats, nt as i32, 4, pf, 2, 10,
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

/// The prefill expert GEMM over one MoE sub-chunk (forward.rs moe_rows' prefill branch).
#[allow(clippy::too_many_arguments)]
fn prefill(st: &Stream, t1p: &DeviceBuffer<i64>, t3p: &DeviceBuffer<i64>, t2p: &DeviceBuffer<i64>, k2: &DeviceBuffer<i32>,
           (h, il, kd, dn): (usize, usize, usize, usize), (wgu, wdn): (usize, usize), e0: &Event, e1: &Event) -> Result<()> {
    use tidelm_kernels::ds41::{ds_route_blocks, ds_route_large};
    use tidelm_kernels::glm::{glm_count_live, glm_member_codes};
    let rows = 2048usize;
    let p = rows * TOPK;
    let mut seed = 0x2545F4914F6CDD1Du64;
    let picks: Vec<i32> = (0..rows).flat_map(|_| {
        let mut row: Vec<i32> = vec![];
        while row.len() < TOPK {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let e = (seed % POOL as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        row
    }).collect();
    let pick = DeviceBuffer::<i32>::from_slice(&picks)?;
    let nb = unsafe { ds_route_blocks(p as i32, POOL as i32, 64) } as usize;
    let work = DeviceBuffer::<i32>::new(3 * POOL + 64)?;
    let (bslot, bpair, broute) = (DeviceBuffer::<i32>::new(nb)?, DeviceBuffer::<i32>::new(nb * 64)?, DeviceBuffer::<i32>::new(nb * 64)?);
    let members = DeviceBuffer::<i32>::new(nb * 64)?;
    let ucount = DeviceBuffer::<i32>::new(1)?;
    let ok = |rc: i32, what: &str| if rc == 0 { Ok(()) } else { Err(format!("{what}: {rc}")) };
    ok(unsafe { ds_route_large(st.raw(), pick.raw(), p as i32, TOPK as i32, POOL as i32, work.raw(), bslot.raw(), bpair.raw(),
                               broute.raw(), 64) }, "route")?;
    ok(unsafe { glm_member_codes(st.raw(), bpair.raw(), (nb * 64) as i32, TOPK as i32, members.raw()) }, "members")?;
    ok(unsafe { glm_count_live(st.raw(), bslot.raw(), nb as i32, ucount.raw()) }, "live")?;
    // random small fp16 activations, so a row mix-up shows up in the check
    let mut hv = |n: usize| -> Vec<u16> {
        (0..n).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; 0x3000u16 | (seed & 0x83ff) as u16 }).collect()
    };
    let x0 = DeviceBuffer::<u16>::from_slice(&hv(p * h))?;
    let x1 = DeviceBuffer::<u16>::from_slice(&hv(p * h))?;
    let xd = DeviceBuffer::<u16>::from_slice(&hv(p * kd))?;
    let zup = DeviceBuffer::<f32>::new(2 * p * il)?;
    let zd = DeviceBuffer::<f32>::new(p * dn)?;
    println!("prefill expert GEMM, {rows} rows x top-{TOPK} over {POOL} experts (median of 8, ms / GB/s of trellis bytes)");
    // the shipped mul1 2-3-bit tile, then exl3m_prefill_smem_cb (mixed widths, up to 4 bits) on the same
    // 3-bit experts in both codebooks
    // the fat tile (exl3m_prefill_fat) against smem_cb mul1 on the same words: CHECK=1 compares every byte
    let check = std::env::var("CHECK").is_ok();
    let zref_up = DeviceBuffer::<f32>::new(2 * p * il)?;
    let zref_dn = DeviceBuffer::<f32>::new(p * dn)?;
    // the fused gate/up epilogue (exl3m_prefill_fat_gateup) against smem_cb + exl3m_gateup, on random scales
    let svh_g = DeviceBuffer::<u16>::from_slice(&hv(POOL * il))?;
    let svh_u = DeviceBuffer::<u16>::from_slice(&hv(POOL * il))?;
    let suh_d = DeviceBuffer::<u16>::from_slice(&hv(POOL * kd))?;
    let xd_ref = DeviceBuffer::<u16>::new(p * il)?;
    let xd_fat = DeviceBuffer::<u16>::new(p * il)?;
    {
        let mut tref = vec![];
        let mut tfat = vec![];
        for i in 0..10 {
            e0.record(st)?;
            unsafe {
                exl3m_prefill_smem_cb(x0.raw(), x1.raw(), t1p.raw(), t3p.raw(), k2.raw(), k2.raw(), bslot.raw(), ucount.raw(),
                                      members.raw(), zup.raw(), h as i32, il as i32, p as i32, TOPK as i32, nb as i32, 2, CB_MUL1, 8,
                                      st.raw());
                exl3m_gateup(zup.raw(), pick.raw(), svh_g.raw(), svh_u.raw(), suh_d.raw(), xd_ref.raw(), rows as i32, p as i32,
                             il as i32, 1, TOPK as i32, POOL as i32, kd as i32, f32::INFINITY, 1, st.raw());
            }
            e1.record(st)?;
            e1.synchronize()?;
            let a = e1.elapsed_ms(e0)? as f64;
            e0.record(st)?;
            unsafe {
                exl3m_prefill_fat_gateup(x0.raw(), x1.raw(), t1p.raw(), t3p.raw(), k2.raw(), k2.raw(), bslot.raw(), ucount.raw(),
                                         members.raw(), h as i32, il as i32, p as i32, TOPK as i32, nb as i32, CB_MUL1, 8,
                                         svh_g.raw(), svh_u.raw(), suh_d.raw(), kd as i32, f32::INFINITY, xd_fat.raw(), st.raw());
            }
            e1.record(st)?;
            e1.synchronize()?;
            ok(unsafe { tidelm_cuda::ffi::cudaGetLastError() }, "fat gateup")?;
            if i >= 2 {
                tref.push(a);
                tfat.push(e1.elapsed_ms(e0)? as f64);
            }
        }
        tref.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tfat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (a, b) = (xd_ref.download()?, xd_fat.download()?);
        let bad = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!("gate+up with the epilogue: smem_cb + gateup {:.2} ms, fat fused {:.2} ms; {bad} of {} xd values differ",
                 tref[tref.len() / 2], tfat[tfat.len() / 2], a.len());
        let h2f = |v: u16| -> f32 { let s = ((v >> 15) & 1) as i32; let e = ((v >> 10) & 0x1f) as i32; let m = (v & 0x3ff) as f32;
                                    let f = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
                                    if s == 1 { -f } else { f } };
        let mut shown = 0;
        let mut rows = std::collections::BTreeMap::new();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            if x != y {
                *rows.entry(i / il).or_insert(0) += 1;
                if shown < 8 {
                    println!("  xd[{}][{}]: ref {:#06x} ({}) fat {:#06x} ({})", i / il, i % il, x, h2f(*x), y, h2f(*y));
                    shown += 1;
                }
            }
        }
        let cols: std::collections::BTreeSet<usize> = a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| (i % il) / 128).collect();
        println!("  differing rows {} (first {:?}), 128-col blocks {:?}", rows.len(), rows.iter().take(6).collect::<Vec<_>>(), cols);
    }
    for kernel in ["smem_mode (3-bit mul1)", "smem_cb mul1", "smem_cb mcg", "fat mul1", "fat mcg"] {
        for (name, mats, x0p, x1p, tp0, tp1, z, k_, n, wq) in [("gate+up", 2, &x0, &x1, t1p, t3p, &zup, h, il, wgu),
                                                              ("down", 1, &xd, &xd, t2p, t2p, &zd, kd, dn, wdn)] {
            let mut ts = vec![];
            for i in 0..10 {
                e0.record(st)?;
                unsafe {
                    match kernel {
                        "smem_mode (3-bit mul1)" => exl3m_prefill_smem_mode(x0p.raw(), x1p.raw(), tp0.raw(), tp1.raw(), k2.raw(),
                            k2.raw(), bslot.raw(), ucount.raw(), members.raw(), z.raw(), k_ as i32, n as i32, p as i32, TOPK as i32,
                            nb as i32, mats, 4, 1, 8, -4, st.raw()),
                        k if k.starts_with("fat") => exl3m_prefill_fat(x0p.raw(), x1p.raw(), tp0.raw(), tp1.raw(), k2.raw(), k2.raw(),
                            bslot.raw(), ucount.raw(), members.raw(), z.raw(), k_ as i32, n as i32, p as i32, TOPK as i32, nb as i32,
                            mats, if kernel.ends_with("mcg") { CB_MCG } else { CB_MUL1 }, 8, st.raw()),
                        _ => exl3m_prefill_smem_cb(x0p.raw(), x1p.raw(), tp0.raw(), tp1.raw(), k2.raw(), k2.raw(), bslot.raw(),
                            ucount.raw(), members.raw(), z.raw(), k_ as i32, n as i32, p as i32, TOPK as i32, nb as i32, mats,
                            if kernel.ends_with("mcg") { CB_MCG } else { CB_MUL1 }, 8, st.raw()),
                    }
                }
                if i == 0 {
                    st.synchronize()?;
                    ok(unsafe { tidelm_cuda::ffi::cudaGetLastError() }, kernel)?;
                    let zref = if mats == 2 { &zref_up } else { &zref_dn };
                    if check && kernel == "smem_cb mul1" {
                        zref.copy_from(0, z, 0, z.len(), st)?;
                    } else if check && kernel == "fat mul1" {
                        let (a, b) = (zref.download()?, z.download()?);
                        let bad = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                        println!("{kernel:>24} {name:>7}: {bad} of {} values differ from smem_cb mul1", a.len());
                    }
                }
                e1.record(st)?;
                e1.synchronize()?;
                if i >= 2 {
                    ts.push(e1.elapsed_ms(e0)? as f64);
                }
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = ts[ts.len() / 2];
            let bytes = (POOL * mats as usize * wq * 4) as f64;
            println!("{kernel:>24} {name:>7}: {ms:.2} ms, {:.0} GB/s", bytes / ms / 1e6);
        }
    }
    Ok(())
}
