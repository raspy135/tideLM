//! The adaptive boot selection from a demand history and a routing trace, as a [40][384] u8
//! mask (for comparing against the Python engine's: the same files must give the same set).
//!
//!   cargo run --release -p tidelm-engine --example adapt_boot -- TRACE.json DB.npz BUDGET OUT.bin [REQ.npz PLAN.txt]
//!
//! With REQ.npz / PLAN.txt: a synthetic request demand (written to REQ.npz) previewed into the
//! history and the plan it gives, one "from_layer from_expert to_layer to_expert" a line.

use std::collections::BTreeMap;

use tidelm_core::npz;
use tidelm_engine::adapt::{load_trace, Adapt, AdaptConfig};
use tidelm_engine::forward::LaneDemand;

fn main() -> tidelm_engine::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (trace, db, budget, out) = (&a[1], &a[2], a[3].parse::<usize>()?, &a[4]);
    let cfg = AdaptConfig::new("high", 4.0, 600, 6)?.ok_or("adaptation off")?;
    let adapt = Adapt::new(cfg, load_trace(std::path::Path::new(trace), 40, 384)?, 40, 384, Some(db.into()))?;
    let keep = adapt.boot_keep(budget)?;
    let bytes: Vec<u8> = keep.iter().flatten().map(|&b| b as u8).collect();
    std::fs::write(out, &bytes)?;
    println!("rust: residents {}", bytes.iter().map(|&b| b as usize).sum::<usize>());
    if let (Some(req), Some(plan)) = (a.get(5), a.get(6)) {
        // a request that leans on absent experts: a deterministic spread over every expert,
        // heavier on every third non-resident one
        let mut x = 12345u64;
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as f64 / (1u64 << 31) as f64
        };
        let (mut counts, mut mass) = (vec![0f64; 40 * 384], vec![0f64; 40 * 384]);
        for l in 0..40 {
            for e in 0..384 {
                let w = if !keep[l][e] && e % 3 == 0 { 40.0 } else { 1.0 };
                let c = (next() * 10.0 * w).floor();
                counts[l * 384 + e] = c;
                mass[l * 384 + e] = c * (0.5 + next());
            }
        }
        let mut m = BTreeMap::new();
        m.insert("counts".to_string(), npz::Array::f64s(vec![40, 384], &counts));
        m.insert("mass".to_string(), npz::Array::f64s(vec![40, 384], &mass));
        npz::write(std::path::Path::new(req), &m)?;
        let moves = adapt.plan(&keep, Some(&LaneDemand { counts, mass, miss: [0.0; 8] }))?;
        let text: String = moves.iter().map(|m| format!("{} {} {} {}\n", m[0], m[1], m[2], m[3])).collect();
        std::fs::write(plan, text)?;
        println!("rust: {} moves", moves.len());
    }
    Ok(())
}
