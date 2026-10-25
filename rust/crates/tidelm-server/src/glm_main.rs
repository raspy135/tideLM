//! tidelm-server for a GLM-5.3 checkpoint (crates/tidelm-glm): the same HTTP front, the GLM
//! engine behind it, one request at a time (server/muse_app.py's queue). main() comes here when
//! --model-dir's config.json has model_type glm_moe_dsa.
//!
//!   tidelm-server --model-dir DIR --pack PACK_r{rank}of2.bin --db HISTORY [--world 2 --rank R
//!                 --addr H:P] [--slots 12500] [--max-seq 262144] [--chunk 4096] [--graphs]
//!                 [--hot 1] [--hot-keep 32] [--depth 7] [--cal 0.36 | --dyn 0.9] [--lookup 8]
//!                 [--adapt 1] [--adapt-every 200] [--adapt-prefill 512] [--adapt-decode 256]
//!                 [--bg-every 2] [--loop-break 12] [--two-pass 512] [--two-pass-miss 0.15]
//!                 [--state-db FILE] [--host 0.0.0.0] [--port 8000] [--served-model-name glm-5.3]
//!
//! The routing history: --db is read at boot (the Python engine's torch.save .pt or this
//! engine's .npz); adaptation writes its own .npz at --state-db (default ~/tidelm-rust-state/
//! glm-stats-rust.npz), never the Python server's file.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Arc};
use std::time::Instant;

use tidelm_engine::decode::{Event, Finish, Generation};
use tidelm_glm::forward::{Engine, Options, Pair, ROCE_BYTES};
use tidelm_glm::serve::{ServeOptions, Server};
use tidelm_text::Text;

use crate::{anthropic, arg, http, Cmd, Ctx, Family};

pub fn is_glm(model_dir: &std::path::Path) -> bool {
    std::fs::read(model_dir.join("config.json")).ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|c| c["model_type"].as_str() == Some("glm_moe_dsa"))
}

pub fn main(args: &[String], model_dir: PathBuf) -> tidelm_engine::Result<()> {
    const VALUED: [&str; 34] = ["--model-dir", "--pack", "--db", "--state-db", "--world", "--rank", "--addr", "--slots",
                                "--max-seq", "--chunk", "--hot", "--hot-keep", "--depth", "--cal", "--dyn", "--lookup",
                                "--adapt", "--adapt-every", "--adapt-prefill", "--adapt-decode", "--bg-every", "--loop-break",
                                "--two-pass", "--two-pass-miss", "--host", "--port", "--served-model-name", "--moe-rows",
                                "--kv-split", "--reduce", "--warmup", "--prefix-disk", "--prefix-disk-gb",
                                "--prefix-min"];
    const FLAGS: [&str; 1] = ["--graphs"];
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        if VALUED.contains(&a) {
            i += 2;
        } else if FLAGS.contains(&a) {
            i += 1;
        } else {
            return Err(format!("tidelm-server (GLM): unknown argument {a}").into());
        }
    }
    let num = |name: &str, d: f64| -> tidelm_engine::Result<f64> {
        Ok(arg(args, name).map(|v| v.parse::<f64>()).transpose().map_err(|e| format!("{name}: {e}"))?.unwrap_or(d))
    };
    let t0 = Instant::now();
    let world = num("--world", 1.0)? as usize;
    let rank = num("--rank", 0.0)? as usize;
    let pack = arg(args, "--pack").ok_or("--pack PACK_r{rank}of2.bin")?.replace("{rank}", &rank.to_string());
    let opt = Options {
        max_seq: num("--max-seq", 262144.0)? as usize,
        chunk: num("--chunk", 4096.0)? as usize,
        slots: num("--slots", 12500.0)? as usize,
        kv_split: world > 1 && arg(args, "--kv-split").as_deref() != Some("0"),
        prefill_reduce_bf16: arg(args, "--reduce").as_deref() != Some("fp32"),
        moe_rows: num("--moe-rows", 2048.0)? as usize,
        pack: PathBuf::from(&pack),
        guard_extra: format!("graphs {} hot {} prefix {} gb {}", args.iter().any(|a| a == "--graphs"),
                             arg(args, "--hot").unwrap_or_else(|| "1".into()), arg(args, "--prefix-disk").is_some(),
                             arg(args, "--prefix-disk-gb").unwrap_or_default()),
    };
    tidelm_cuda::set_device(0)?;
    let cfg = tidelm_glm::config::GlmConfig::load(&model_dir)?;
    let e = cfg.experts;
    let mut pair = None;
    if world == 2 {
        let p = tidelm_glm::arena::Pack::open(&opt.pack, cfg.hidden, cfg.moe_inter)?;
        let boot = opt.boot_config(&model_dir, &p)?;
        let addr = arg(args, "--addr").unwrap_or_else(|| "10.0.0.1:29711".into());
        let (comm, mut ctl) = tidelm_comm::join(rank, &addr, &boot, std::time::Duration::from_secs(1800))?;
        let hcas: Option<Vec<String>> = std::env::var("NCCL_IB_HCA").ok().map(|v| v.split(',').map(|s| s.to_string()).collect());
        let roce = tidelm_comm::roce::Roce::setup(rank, &mut ctl, ROCE_BYTES, hcas.as_deref())?;
        pair = Some(Pair { comm, ctl, roce });
    } else if world != 1 {
        return Err("--world 1 or 2".into());
    }
    // the boot residents: rank 0 places them from the history (tp._db_resident), rank 1 takes them
    let swap: Vec<usize> = (0..cfg.layers).filter(|&l| !cfg.dense[l]).collect();
    let state_db = arg(args, "--state-db").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("tidelm-rust-state/glm-stats-rust.npz")
    });
    let mut mass = None;
    let residents: Vec<(usize, usize)> = if rank == 0 {
        let db = if state_db.exists() { state_db.clone() } else { PathBuf::from(arg(args, "--db").ok_or("--db HISTORY")?) };
        let m = tidelm_glm::residency::read_mass(&db, cfg.n_layers(), e)?;
        eprintln!("boot: routing history {} ", db.display());
        let r = tidelm_glm::residency::placement(&m, e, &swap, opt.slots, 2 * cfg.topk);
        if let Some(p) = pair.as_mut() {
            let mut msg = tidelm_comm::Msg::default();
            msg.u32s(&r.iter().map(|&(l, x)| (l * e + x) as u32).collect::<Vec<_>>());
            p.ctl.send(&msg.0)?;
        }
        mass = Some(m);
        r
    } else {
        let msg = pair.as_mut().ok_or("rank 1 needs the pair")?.ctl.recv()?;
        tidelm_comm::Reader::new(&msg).u32s()?.into_iter().map(|v| (v as usize / e, v as usize % e)).collect()
    };
    let log = move |s: &str| {
        if rank == 0 {
            eprintln!("{s}");
        }
    };
    let mut eng = Engine::load(&model_dir, opt, rank, world, pair, &residents, &log)?;
    eng.graphs = arg(args, "--graphs").is_some() || args.iter().any(|a| a == "--graphs");
    if num("--hot", 1.0)? > 0.0 {
        eng.enable_hot(true)?;
    }
    if rank == 0 {
        if let Some(p) = state_db.parent() {
            std::fs::create_dir_all(p)?;
        }
        eng.res.db_mass = mass;
        eng.res.db_path = Some(state_db);
    }
    let sopt = ServeOptions {
        depth: num("--depth", 7.0)? as usize,
        cal: arg(args, "--cal").map(|v| v.parse()).transpose().map_err(|e| format!("--cal: {e}"))?,
        dyn_gate: num("--dyn", 0.9)?,
        lookup: num("--lookup", 8.0)? as usize,
        adapt: num("--adapt", 1.0)? > 0.0,
        adapt_every: num("--adapt-every", 200.0)? as usize,
        adapt_prefill: num("--adapt-prefill", 512.0)? as usize,
        adapt_decode: num("--adapt-decode", 256.0)? as usize,
        bg_every: num("--bg-every", 2.0)? as usize,
        hot_keep: num("--hot-keep", 32.0)? as usize,
        loop_break: num("--loop-break", 12.0)? as usize,
        two_pass: num("--two-pass", 512.0)? as usize,
        two_pass_miss: num("--two-pass-miss", 0.15)?,
    };
    // the disk prefix store: its namespace is everything that changes what a cached row means
    let mut store = None;
    if let Some(root) = arg(args, "--prefix-disk").map(PathBuf::from) {
        use sha2::Digest;
        let cfg_bytes = std::fs::read(model_dir.join("config.json"))?;
        let key = format!("glm-rust-kv v1 {:x} {} {} {} {}", sha2::Sha256::digest(&cfg_bytes), eng.pack.source_sha256,
                          eng.opt.max_seq, eng.opt.kv_split, world);
        let ns = format!("{:x}", sha2::Sha256::digest(key.as_bytes()))[..16].to_string();
        let gb = num("--prefix-disk-gb", 20.0)?;
        let st = tidelm_glm::prefix::Store::open(&root, &ns, rank, (gb * 1e9) as u64, num("--prefix-min", 256.0)? as usize)?;
        eng.prefix_dir = Some(st.dir.clone());
        if rank == 0 {
            eprintln!("boot: prefix store {} ({} segments, {gb} GB)", st.dir.display(), st.index.segs.len());
            store = Some(st);
        }
    }
    if rank == 1 {
        eng.follow()?;
        return Ok(());
    }
    let mut srv = Server::new(eng, sopt, Box::new(|s: &str| eprintln!("{s}")));
    srv.store = store;
    if num("--warmup", 1.0)? > 0.0 {
        let tw = Instant::now();
        srv.warmup()?;
        eprintln!("boot: warm-up {:.1} s", tw.elapsed().as_secs_f64());
    }
    let text = Text::load(&model_dir)?;
    let family = Family::load(&model_dir, &text)?;
    let (eos, think_end) = match &family {
        Family::Glm(g) => (g.ids.stop[0], Some(g.ids.think_end)),
        Family::Ds => return Err("not a GLM checkpoint".into()),
    };
    let max_seq = srv.eng.opt.max_seq;
    let (tx, rx) = mpsc::channel::<Cmd>();
    let ctx = Arc::new(Ctx {
        text,
        model: arg(args, "--served-model-name").unwrap_or_else(|| "glm-5.3".into()),
        max_seq,
        default_thinking: true,
        default_effort: 75,
        eos,
        think_end,
        vision: None,
        engine: tx,
        next_id: AtomicU64::new(1),
        family,
        memo: anthropic::ReasoningMemo::default(),
    });
    let addr = format!("{}:{}", arg(args, "--host").unwrap_or_else(|| "0.0.0.0".into()),
                       arg(args, "--port").unwrap_or_else(|| "8000".into()));
    let listener = TcpListener::bind(&addr)?;
    eprintln!("engine up in {:.1}s; serving {} on {addr} (GLM, depth {}, {})", t0.elapsed().as_secs_f64(), ctx.model,
              srv.opt.depth, match srv.opt.cal { Some(t) => format!("calibrated gate {t}"), None => format!("flat gate {}", srv.opt.dyn_gate) });
    {
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let ctx = ctx.clone();
                std::thread::spawn(move || http::serve(stream, &ctx));
            }
        });
    }
    engine_loop(srv, rx)
}

/// The engine thread: one request at a time in arrival order; cancels and new requests are
/// taken between rounds. An engine error is fatal: a pair that failed one call cannot go on.
fn engine_loop(mut srv: Server, rx: mpsc::Receiver<Cmd>) -> tidelm_engine::Result<()> {
    let mut queue = VecDeque::new();
    let mut cancelled = std::collections::HashSet::new();
    loop {
        if queue.is_empty() {
            match rx.recv() {
                Ok(Cmd::Submit(j)) => queue.push_back(j),
                Ok(Cmd::Cancel(id)) => {
                    cancelled.insert(id);
                }
                Err(_) => break,
            }
        }
        while let Ok(c) = rx.try_recv() {
            match c {
                Cmd::Submit(j) => queue.push_back(j),
                Cmd::Cancel(id) => {
                    cancelled.insert(id);
                }
            }
        }
        let Some(mut job) = queue.pop_front() else { continue };
        if cancelled.remove(&job.id) {
            (job.sink)(Event::Done(Finish::Cancelled, Generation::default()));
            continue;
        }
        let id = job.id;
        let (s0, r0) = (srv.stats.clone(), srv.eng.res.stats.clone());
        let mut poll = || {
            while let Ok(c) = rx.try_recv() {
                match c {
                    Cmd::Submit(j) => queue.push_back(j),
                    Cmd::Cancel(c) => {
                        cancelled.insert(c);
                    }
                }
            }
            cancelled.remove(&id)
        };
        if let Err(e) = srv.run(std::mem::replace(&mut job, dummy_job()), &mut poll) {
            eprintln!("engine failed: {e}");
            let _ = srv.eng.stop_follower();
            std::process::exit(1);
        }
        let (s1, r1) = (&srv.stats, &srv.eng.res.stats);
        let rounds = (s1.rounds - s0.rounds).max(1) as f64;
        eprintln!("request {id}, GLM: {:.2} drafted / {:.2} accepted a round, copy rounds {}, \
                   hot reads {} (kept {}), swaps {}, fallbacks {}",
                  (s1.drafted - s0.drafted) as f64 / rounds, (s1.accepted - s0.accepted) as f64 / rounds,
                  s1.copy_rounds - s0.copy_rounds, r1.hot_reads - r0.hot_reads, r1.hot_kept - r0.hot_kept, r1.swaps - r0.swaps,
                  s1.fallbacks - s0.fallbacks);
    }
    srv.eng.stop_follower()?;
    Ok(())
}

fn dummy_job() -> tidelm_engine::decode::Job {
    tidelm_engine::decode::Job { id: 0, prompt: vec![], max_tokens: 0, sampling: tidelm_engine::forward::Sampling::GREEDY, seed: 0,
                                 stop: vec![], sink: Box::new(|_| {}), images: vec![], key: None, copy_draft: false }
}
