//! TIDELM_GLM_ABLIT (a profile's GLM_ABLIT): serve GLM-5.3 abliterated, as a rank-1 projection of
//! each edited layer's residual writes instead of edited weights.
//!
//! Blackfrost-Research/GLM-5.3-DERISKED-BF16 is the BF16 checkpoint with W' = W - alpha r (r^T W)
//! on every residual-stream writer of layers 2-49 - the attention's o_proj and every down_proj,
//! dense, shared and routed - one unit direction r a layer, alpha 3. A layer's writers' outputs
//! are summed into the residual, so editing all of them is the same as projecting their sum:
//! y <- y - alpha r (r . y) on the attention's output and on the MLP's, the experts we serve
//! (EXL3 packs of the stock weights) untouched and nothing re-quantized. The directions were
//! recovered from the two BF16 checkpoints by range requests, layer by layer
//! (tools/glm_ablit_fit.py, 2026-10-29).
//!
//! The file: safetensors with `glm_ablit.layers` I32 [n], `glm_ablit.alpha` F32 [n] and
//! `glm_ablit.r` F32 [n, hidden]. It changes the model by design, so it is off unless set; its
//! sha256 is in the pair's boot config (two ranks cannot serve different directions) and in the
//! disk prefix store's namespace (rows cached under the other model are never restored).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use tidelm_cuda::DeviceBuffer;

use crate::Result;

pub const ENV: &str = "TIDELM_GLM_ABLIT";

/// The directions file, when set.
pub fn path() -> Option<PathBuf> {
    std::env::var_os(ENV).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The file's sha256 in hex ("" when unset), hashed once per process.
pub fn digest() -> Result<String> {
    static DIGEST: OnceLock<String> = OnceLock::new();
    if let Some(d) = DIGEST.get() {
        return Ok(d.clone());
    }
    let d = match path() {
        Some(p) => sha256(&p).map_err(|e| format!("{ENV}={}: {e}", p.display()))?,
        None => String::new(),
    };
    Ok(DIGEST.get_or_init(|| d).clone())
}

fn sha256(p: &Path) -> std::io::Result<String> {
    let (mut f, mut h, mut buf) = (std::fs::File::open(p)?, Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        match f.read(&mut buf)? {
            0 => break,
            n => h.update(&buf[..n]),
        }
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The directions on the device: row i of `r` is layer `layers[i]`'s.
pub struct Ablit {
    pub layers: Vec<usize>,
    pub alpha: Vec<f32>,
    pub r: DeviceBuffer<f32>,
    hidden: usize,
}

impl Ablit {
    /// Layer l's (direction, alpha), when it is edited.
    pub fn of(&self, l: usize) -> Option<(*const std::os::raw::c_void, f32)> {
        let i = self.layers.iter().position(|&x| x == l)?;
        Some((self.r.ptr_at(i * self.hidden) as *const _, self.alpha[i]))
    }
}

/// The file's directions for a model of `n_layers` decoder layers and `hidden` wide, when set:
/// refused unless every row is a unit vector of that width and every layer one of the decoder's.
pub fn load(n_layers: usize, hidden: usize) -> Result<Option<Ablit>> {
    let Some(p) = path() else { return Ok(None) };
    // the file is small (a few MB) and its names are not the checkpoint's: read it whole
    let bytes = std::fs::read(&p).map_err(|e| format!("{ENV}={}: {e}", p.display()))?;
    let bad = |what: String| -> crate::Error { format!("{ENV}={}: {what}", p.display()).into() };
    let n_hdr = u64::from_le_bytes(bytes.get(..8).ok_or_else(|| bad("not safetensors".into()))?.try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + n_hdr).ok_or_else(|| bad("short header".into()))?)
        .map_err(|e| bad(format!("header: {e}")))?;
    let data = &bytes[8 + n_hdr..];
    let get = |name: &str, dtype: &str| -> Result<(Vec<u8>, Vec<usize>)> {
        let m = &hdr[name];
        if m["dtype"].as_str() != Some(dtype) {
            return Err(bad(format!("{name}: dtype {}, not {dtype}", m["dtype"])));
        }
        let shape: Vec<usize> = m["shape"].as_array().map(|v| v.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            .unwrap_or_default();
        let off: Vec<usize> = m["data_offsets"].as_array().map(|v| v.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            .unwrap_or_default();
        match (off.as_slice(), data.get(off.first().copied().unwrap_or(0)..off.get(1).copied().unwrap_or(0))) {
            ([a, b], Some(d)) if b - a == 4 * shape.iter().product::<usize>() => Ok((d.to_vec(), shape)),
            _ => Err(bad(format!("{name}: offsets {off:?} for shape {shape:?}"))),
        }
    };
    let (lb, ls) = get("glm_ablit.layers", "I32")?;
    let (ab, as_) = get("glm_ablit.alpha", "F32")?;
    let (rb, rs) = get("glm_ablit.r", "F32")?;
    let n = ls.first().copied().unwrap_or(0);
    if ls.len() != 1 || as_ != [n] || rs != [n, hidden] || n == 0 {
        return Err(format!("{ENV}={}: shapes {ls:?} / {as_:?} / {rs:?} for {hidden}-wide rows", p.display()).into());
    }
    let layers: Vec<usize> = lb.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
    let alpha: Vec<f32> = ab.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let r: Vec<f32> = rb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    for (i, &l) in layers.iter().enumerate() {
        let norm: f64 = r[i * hidden..(i + 1) * hidden].iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        if l >= n_layers || (norm - 1.0).abs() > 1e-3 || !alpha[i].is_finite() {
            return Err(format!("{ENV}={}: row {i}: layer {l} of {n_layers}, |r| {norm:.5}, alpha {}", p.display(), alpha[i]).into());
        }
    }
    Ok(Some(Ablit { layers, alpha, r: DeviceBuffer::from_slice(&r)?, hidden }))
}
