//! The DS v4.1 engine in Rust.
//!
//! - `config`: inference/config.json
//! - `rope`: the RoPE / YaRN tables
//! - `weights`: device weights in their stored formats, loaded from the checkpoint
//! - `forward`: the forward pass on the ds41 and native FP4 kernels

pub mod config;
pub mod decode;
pub mod forward;
pub mod keep;
pub mod pair;
pub mod rope;
pub mod weights;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
