//! The GLM-5.3 engine on the Rust stack: engine/glm/tp.py's TP2 forward (EXL3 weights, absorbed
//! MLA with the DSA indexer, the KV cache split by position across the pair, routed experts from
//! a per-rank arena), its MTP layer, and the serving loop around them. No Python anywhere.

pub mod ablit;
pub mod adapt;
pub mod arena;
pub mod config;
pub mod forward;
pub mod prefix;
pub mod residency;
pub mod serve;
pub mod weights;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

/// A kernel launcher's return code as a Result.
pub fn ck(rc: i32, what: &str) -> Result<()> {
    if rc == 0 { Ok(()) } else { Err(format!("{what}: CUDA error {rc}").into()) }
}
