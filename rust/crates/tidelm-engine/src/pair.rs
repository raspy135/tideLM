//! Bringing up a two-box tensor-parallel pair from a CLI: `--world 2 --rank R --addr HOST:PORT`.

use std::time::Duration;

use crate::forward::{tp_max_gather, Options, Pair};
use crate::weights::Tp;
use crate::Result;

#[derive(Clone)]
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

/// Agree on the boot config with the peer, bring up RoCE (on the HCAs NCCL_IB_HCA names, all
/// active ones otherwise), and make `opts` this rank's tensor-parallel shard.
/// `join` on its own thread, so the engine loads while the pair handshakes (boot config, NCCL,
/// RoCE: ~1-2 s, before the engine load began): `opts.tp` is set at once, the Pair comes from
/// the handle. A boot-config mismatch then surfaces when the handle is joined, after the load.
pub fn join_in_background(opts: &mut Options, model_dir: &std::path::Path, pa: &PairArgs)
                          -> Result<std::thread::JoinHandle<Result<Pair>>> {
    let mut o = opts.clone();
    let (dir, pa) = (model_dir.to_path_buf(), pa.clone());
    opts.tp = Some(Tp { rank: pa.rank, world: 2 });
    Ok(std::thread::spawn(move || {
        tidelm_cuda::set_device(0)?;
        join(&mut o, &dir, &pa)
    }))
}

pub fn join(opts: &mut Options, model_dir: &std::path::Path, pa: &PairArgs) -> Result<Pair> {
    let boot = opts.boot_config(model_dir)?;
    let (comm, mut ctl) = tidelm_comm::join(pa.rank, &pa.addr, &boot, Duration::from_secs(300))?;
    let hcas: Option<Vec<String>> = std::env::var("NCCL_IB_HCA").ok().map(|v| v.split(',').map(|s| s.to_string()).collect());
    let roce = tidelm_comm::roce::Roce::setup(pa.rank, &mut ctl, tp_max_gather(), hcas.as_deref())?;
    opts.tp = Some(Tp { rank: pa.rank, world: 2 });
    Ok(Pair { comm, ctl, roce })
}
