//! GPU-free core of the DS v4.1 engine, ported from `engine/` (Python).
//!
//! Everything in this crate is host logic with no CUDA dependency, so it builds and
//! tests anywhere. Each module names the Python file it was ported from; the port is
//! semantic, not line-by-line, but any intentional behavior difference is called out
//! at the site. The EP2 rule from the repo notes applies unchanged: anything here that
//! changes numerics or control flow per rank must surface in the boot-time config
//! guard, never behind a rank-local condition.

pub mod residency;
pub mod sampling;
pub mod scales;
pub mod spec;
