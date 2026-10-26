//! TIDELM_ABLIT_WOB (a profile's DSV41_ABLIT_WOB): serve layers 10-35's attention output
//! projection from an abliteration overlay, as engine/ablit.py does for the Python engine.
//!
//! The overlay (README: `drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`) is 52 tensors,
//! `layers.{10..35}.attn.wo_b.{weight,scale}` in the checkpoint's own storage (FP8 E4M3 + UE8M0);
//! replacing them is the whole abliteration. Experts, Engram, the DSpark heads, vision and every
//! other tensor stay the checkpoint's. It is applied where the checkpoint resolves a tensor to
//! its file (tidelm_model::Checkpoint::overlay), so the boot load, the TP shards and every later
//! read see it, and the stock tree is never written.
//!
//! It changes the model by design, so it is off unless set. Its sha256 is in the pair's boot
//! config (two ranks cannot serve different overlays) and in the disk prefix cache's namespace
//! (a prefix cached under other weights is never restored).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use tidelm_model::Checkpoint;

use crate::Result;

pub const ENV: &str = "TIDELM_ABLIT_WOB";
pub const FIRST: usize = 10;
pub const LAST: usize = 35;

/// The overlay file, when set.
pub fn path() -> Option<PathBuf> {
    std::env::var_os(ENV).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The tensors an overlay must hold, sorted: exactly these, under the checkpoint's names.
pub fn expected() -> Vec<String> {
    let mut v: Vec<String> =
        (FIRST..=LAST).flat_map(|l| ["weight", "scale"].map(|k| format!("layers.{l}.attn.wo_b.{k}"))).collect();
    v.sort();
    v
}

/// The overlay file's sha256 in hex ("" when unset), hashed once per process.
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
    let (mut f, mut h, mut buf) = (std::fs::File::open(p)?, Sha256::new(), vec![0u8; 8 << 20]);
    loop {
        match f.read(&mut buf)? {
            0 => break,
            n => h.update(&buf[..n]),
        }
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The checkpoint, with the overlay applied when TIDELM_ABLIT_WOB is set.
pub fn open(model_dir: &Path) -> Result<Checkpoint> {
    let mut ckpt = Checkpoint::open(model_dir)?;
    if let Some(p) = path() {
        apply(&mut ckpt, &p)?;
    }
    Ok(ckpt)
}

/// Serve `p`'s tensors over the checkpoint's: refused unless they are exactly `expected()`, each
/// of the checkpoint's dtype and shape, rather than serving a half-abliterated model.
pub fn apply(ckpt: &mut Checkpoint, p: &Path) -> Result<()> {
    let names = ckpt.overlay(p).map_err(|e| format!("{ENV}={}: {e}", p.display()))?;
    let want = expected();
    if names != want {
        let missing: Vec<&String> = want.iter().filter(|n| !names.contains(n)).collect();
        let extra: Vec<&String> = names.iter().filter(|n| !want.contains(n)).collect();
        return Err(format!("{ENV}={} is not a wo_b overlay: missing {} (e.g. {:?}), unexpected {} (e.g. {:?}); it must be \
                            layers.{{{FIRST}..{LAST}}}.attn.wo_b.{{weight,scale}} in the checkpoint's names",
                           p.display(), missing.len(), missing.first(), extra.len(), extra.first()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_overlay_is_layers_10_to_35_wo_b() {
        let e = expected();
        assert_eq!(e.len(), 52);
        assert!(e.contains(&"layers.10.attn.wo_b.weight".to_string()) && e.contains(&"layers.35.attn.wo_b.scale".to_string()));
        assert!(!e.iter().any(|n| n.starts_with("layers.9.") || n.starts_with("layers.36.")));
    }

    /// The published overlay on the real checkpoint (skipped when either is absent): the 52
    /// tensors read back as the overlay file's bytes, and differ from the stock ones.
    #[test]
    fn the_published_overlay_replaces_wo_b_on_the_real_checkpoint() {
        let home = PathBuf::from(std::env::var("HOME").unwrap());
        let dir = home.join("models/DeepSeek-V4.1-Flash-1017");
        let ov = home.join("models/dsv41-1017-wo-b-ablit/wo_b_l10_35_1017.safetensors");
        if !dir.join("model.safetensors.index.json").exists() || !ov.exists() {
            eprintln!("checkpoint or overlay not present; skipping");
            return;
        }
        let stock = Checkpoint::open(&dir).unwrap();
        let mut ckpt = Checkpoint::open(&dir).unwrap();
        apply(&mut ckpt, &ov).unwrap();
        let file = tidelm_model::safetensors::Shard::open(&ov).unwrap();
        for name in ["layers.10.attn.wo_b.weight", "layers.35.attn.wo_b.scale", "layers.22.attn.wo_b.weight"] {
            let (got, meta) = ckpt.read_tensor_vec(name).unwrap();
            let mut want = vec![0u8; meta.nbytes()];
            file.file.pread_buffered(&mut want, file.meta(name).unwrap().lo).unwrap();
            assert_eq!(got, want, "{name}");
            if name.ends_with("weight") {
                assert_ne!(got, stock.read_tensor_vec(name).unwrap().0, "{name} is the stock tensor");
            }
        }
        for name in ["layers.9.attn.wo_b.weight", "layers.36.attn.wo_b.weight", "mtp.0.attn.wo_b.weight"] {
            assert_eq!(ckpt.read_tensor_vec(name).unwrap().0, stock.read_tensor_vec(name).unwrap().0, "{name}");
        }
    }
}
