//! The DS v4.1 engine in Rust.
//!
//! - `config`: inference/config.json
//! - `rope`: the RoPE / YaRN tables
//! - `weights`: device weights in their stored formats, loaded from the checkpoint
//! - `forward`: the forward pass on the ds41 and native FP4 kernels
//! - `adapt`: adaptive expert residency (the demand history and the swap planner)
//! - `ablit`: the abliteration overlay (layers 10-35 attn.wo_b from a sidecar file)

pub mod ablit;
pub mod adapt;
pub mod config;
pub mod copy;
pub mod decode;
pub mod draft_vocab;
pub mod forward;
pub mod hot;
pub mod keep;
pub mod pair;
pub mod prefix_disk;
pub mod rope;
pub mod vision;
pub mod weights;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
