//! Bringing up a two-box pair from a CLI: `--world 2 --rank R --addr HOST:PORT`.

use std::time::Duration;

use crate::forward::{Options, Pair};
use crate::keep::owned_split;
use crate::Result;

pub struct PairArgs {
    pub rank: usize,
    pub addr: String,
}

/// The pair flags, if any (`--world 2` is the only world besides 1).
pub fn pair_args(args: &[String]) -> Result<Option<PairArgs>> {
    let arg = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1).cloned());
    match arg("--world").as_deref() {
        None | Some("1") => Ok(None),
        Some("2") => Ok(Some(PairArgs {
            rank: arg("--rank").ok_or("--world 2 needs --rank 0|1")?.parse()?,
            addr: arg("--addr").unwrap_or_else(|| "10.0.0.1:29711".into()),
        })),
        Some(w) => Err(format!("--world {w}: only 1 or 2").into()),
    }
}

/// Agree on the boot config with the peer, then take this rank's share of the routable
/// experts (`scores`: the demand ranking that alternates them between the ranks).
pub fn join(opts: &mut Options, scores: Option<&[Vec<f64>]>, model_dir: &std::path::Path, pa: &PairArgs) -> Result<Pair> {
    let keep = opts.keep.as_ref().ok_or("expert parallel needs a keep set (--keep-db/--budget or --keep-first)")?;
    let boot = opts.boot_config(model_dir)?;
    let (comm, ctl) = tidelm_comm::join(pa.rank, &pa.addr, &boot, Duration::from_secs(300))?;
    opts.owned = Some(owned_split(keep, scores, 2, pa.rank));
    Ok(Pair { comm, ctl })
}
