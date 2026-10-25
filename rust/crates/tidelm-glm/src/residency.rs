//! Adaptive expert residency's state on the host: the routing history (tp.py's GLM_STATS_DB,
//! a torch.save'd {"version": 1, "mass": float64 [layers + 1, experts], "tokens": int}; the Rust
//! engine keeps its own copy as .npz), and the boot placement from it (tp._db_resident).

use std::collections::BTreeMap;
use std::path::Path;

use crate::Result;

/// The stored members of a zip archive (torch.save and np.savez both write stored entries).
fn zip_members(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    let eocd = (0..bytes.len().saturating_sub(21)).rev().find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or("zip: no end of central directory")?;
    let u16at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;
    let u32at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
    let (n, mut at) = (u16at(eocd + 10), u32at(eocd + 16));
    let mut out = BTreeMap::new();
    for _ in 0..n {
        if at + 46 > bytes.len() || u32at(at) != 0x0201_4b50 {
            return Err("zip: bad central directory entry".into());
        }
        let method = u16at(at + 10);
        let csize = u32at(at + 20);
        let (nlen, xlen, clen) = (u16at(at + 28), u16at(at + 30), u16at(at + 32));
        let local = u32at(at + 42);
        let name = String::from_utf8_lossy(&bytes[at + 46..at + 46 + nlen]).to_string();
        at += 46 + nlen + xlen + clen;
        if method != 0 {
            return Err(format!("zip: {name} is compressed").into());
        }
        let data_at = local + 30 + u16at(local + 26) + u16at(local + 28);
        out.insert(name, bytes.get(data_at..data_at + csize).ok_or("zip: truncated")?.to_vec());
    }
    Ok(out)
}

/// The routing history's mass [layers][experts] (f64), from the Python engine's .pt or this
/// engine's .npz.
pub fn read_mass(path: &Path, layers: usize, experts: usize) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let want = layers * experts;
    if path.extension().and_then(|e| e.to_str()) == Some("npz") {
        let m = tidelm_core::npz::parse(&bytes).map_err(|e| e.0)?;
        let a = m.get("mass").ok_or("history: no mass")?;
        let v = a.to_f64().map_err(|e| e.0)?;
        if v.len() != want {
            return Err(format!("history: mass of {} values, want {want}", v.len()).into());
        }
        return Ok(v);
    }
    // torch.save: the one float64 storage of layers x experts values
    let members = zip_members(&bytes)?;
    let data = members.iter().find(|(k, v)| k.contains("/data/") && v.len() == want * 8)
        .map(|(_, v)| v).ok_or_else(|| format!("{}: no float64 [{layers}, {experts}] storage", path.display()))?;
    let v: Vec<f64> = data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{}: non-finite mass", path.display()).into());
    }
    Ok(v)
}

pub fn write_mass(path: &Path, mass: &[f64], layers: usize, experts: usize, tokens: u64) -> Result<()> {
    let mut m = BTreeMap::new();
    m.insert("mass".to_string(), tidelm_core::npz::Array::f64s(vec![layers, experts], mass));
    m.insert("tokens".to_string(), tidelm_core::npz::Array::i64s(vec![1], &[tokens as i64]));
    let tmp = path.with_extension("tmp.npz");
    tidelm_core::npz::write(&tmp, &m).map_err(|e| e.0)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// tp._db_resident: per swap layer its `floor` heaviest experts by mass, then the remaining
/// slots by global mass (ties to the lower layer, then expert).
pub fn placement(mass: &[f64], experts: usize, swap_layers: &[usize], slots: usize, floor: usize) -> Vec<(usize, usize)> {
    let mut chosen = vec![false; mass.len()];
    let mut out = vec![];
    for &l in swap_layers {
        let mut es: Vec<usize> = (0..experts).collect();
        es.sort_by(|&a, &b| mass[l * experts + b].partial_cmp(&mass[l * experts + a]).unwrap().then(a.cmp(&b)));
        for &e in es.iter().take(floor) {
            if out.len() < slots {
                chosen[l * experts + e] = true;
                out.push((l, e));
            }
        }
    }
    let mut rest: Vec<(usize, usize)> = swap_layers.iter().flat_map(|&l| (0..experts).map(move |e| (l, e)))
        .filter(|&(l, e)| !chosen[l * experts + e]).collect();
    rest.sort_by(|a, b| mass[b.0 * experts + b.1].partial_cmp(&mass[a.0 * experts + a.1]).unwrap().then(a.cmp(b)));
    for x in rest {
        if out.len() >= slots {
            break;
        }
        out.push(x);
    }
    out
}
