//! tidelm-server for a GLM-5.3 checkpoint (crates/tidelm-glm): the same HTTP front, the GLM
//! engine behind it, --lanes requests at a time (serve.rs's scheduler; 1: server/muse_app.py's
//! queue). main() comes here when
//! --model-dir's config.json has model_type glm_moe_dsa.
//!
//!   tidelm-server --model-dir DIR --pack PACK_r{rank}of2.bin --db HISTORY [--world 2 --rank R
//!                 --addr H:P] [--slots 12500] [--max-seq 262144] [--chunk 4096] [--graphs]
//!                 [--hot 1] [--hot-async 0] [--hot-keep 32] [--depth 7] [--cal 0.36 | --dyn 0.9] [--lookup 8] [--copy 1]
//!                 [--adapt 1] [--adapt-every 200] [--adapt-prefill 512] [--adapt-decode 256]
//!                 [--bg-every 2] [--loop-break 12] [--two-pass 512] [--two-pass-miss 0.15]
//!                 [--lanes 1] [--lane-chunk 1024]
//!                 [--state-db FILE] [--host 0.0.0.0] [--port 8000] [--served-model-name glm-5.3]
//!
//! The routing history: --db is read at boot (the Python engine's torch.save .pt or this
//! engine's .npz); adaptation writes its own .npz at --state-db (default ~/tidelm-rust-state/
//! glm-stats-rust.npz), never the Python server's file.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Arc};
use std::time::Instant;

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
    const VALUED: [&str; 50] = ["--bench-nll-ab", "--bench-prefill-ab", "--bench-ab", "--bench-ab-rounds", "--bench-nll-skip", "--bench-nll-rows", "--bench-nll-max", "--lanes", "--lane-chunk", "--bench-nll-out", "--bench-nll", "--bench-widths", "--copy", "--hot-async", "--bench-width", "--bench-ctx", "--model-dir", "--pack", "--db", "--state-db", "--world", "--rank", "--addr", "--slots",
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
        lanes: num("--lanes", 1.0)? as usize,
        guard_extra: format!("graphs {} hot {} async {} prefix {} gb {}", args.iter().any(|a| a == "--graphs"),
                             arg(args, "--hot").unwrap_or_else(|| "1".into()), arg(args, "--hot-async").unwrap_or_default(),
                             arg(args, "--prefix-disk").is_some(), arg(args, "--prefix-disk-gb").unwrap_or_default()),
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
        // the expert memory is --slots 3-bit experts' worth: a pack of narrower experts holds more
        let pack = tidelm_glm::arena::Pack::open(&opt.pack, cfg.hidden, cfg.moe_inter)?;
        let unit: usize = tidelm_glm::arena::record_sizes([3; 3], world, cfg.hidden, cfg.moe_inter).iter().sum();
        let cost = |l: usize, x: usize| pack.records.get(&(l, x)).map(|r| r.nbytes).unwrap_or(usize::MAX);
        let r = tidelm_glm::residency::placement_bytes(&m, e, &swap, opt.slots * unit, 2 * cfg.topk, &cost);
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
    // async hot reads (GLM_HOT_ASYNC) go into the background period's victims, and have none
    // without it (tp.py's rule)
    let hot_async = num("--hot-async", 0.0)? > 0.0;
    if hot_async && (num("--hot", 1.0)? == 0.0 || num("--bg-every", 2.0)? == 0.0 || num("--hot-keep", 32.0)? == 0.0
                     || num("--adapt", 1.0)? == 0.0) {
        return Err("--hot-async 1 needs --hot 1, --adapt 1 and the background period's victims (--bg-every, --hot-keep)".into());
    }
    if num("--hot", 1.0)? > 0.0 {
        eng.enable_hot(hot_async)?;
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
        copy: if num("--copy", 1.0)? > 0.0 { Some(tidelm_glm::serve::glm_copy_config()?) } else { None },
        adapt: num("--adapt", 1.0)? > 0.0,
        adapt_every: num("--adapt-every", 200.0)? as usize,
        adapt_prefill: num("--adapt-prefill", 512.0)? as usize,
        adapt_decode: num("--adapt-decode", 256.0)? as usize,
        bg_every: num("--bg-every", 2.0)? as usize,
        hot_keep: num("--hot-keep", 32.0)? as usize,
        loop_break: num("--loop-break", 12.0)? as usize,
        two_pass: num("--two-pass", 512.0)? as usize,
        two_pass_miss: num("--two-pass-miss", 0.15)?,
        lane_chunk: num("--lane-chunk", 1024.0)? as usize,
    };
    // the disk prefix store: its namespace is everything that changes what a cached row means
    let mut store = None;
    if let Some(root) = arg(args, "--prefix-disk").map(PathBuf::from) {
        use sha2::Digest;
        let cfg_bytes = std::fs::read(model_dir.join("config.json"))?;
        let key = format!("glm-rust-kv v1 {:x} {} {} {} {}", sha2::Sha256::digest(&cfg_bytes), eng.pack.source_sha256,
                          eng.opt.max_seq, eng.opt.kv_split, world);
        // GLM_ABLIT changes every row past its first edited layer: its own namespace (stock: unchanged)
        let ablit = tidelm_glm::ablit::digest()?;
        let key = if ablit.is_empty() { key } else { format!("{key} ablit {ablit}") };
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
        tidelm_cuda::mem::mark_up();
        eprintln!("memory at boot: {}", tidelm_cuda::mem::report());
        eng.follow()?;
        return Ok(());
    }
    let mut srv = Server::new(eng, sopt, Box::new(|s: &str| eprintln!("{s}")));
    srv.store = store;
    // DS's routing history for GLM's traffic, beside the state db (residency::DsHistory)
    if let Some(p) = srv.eng.res.db_path.clone() {
        let path = p.with_file_name("glm-ds-history.npz");
        let h = tidelm_glm::residency::DsHistory::open(&path, srv.eng.cfg.n_layers(), srv.eng.cfg.experts);
        eprintln!("boot: DS-rule routing history {} ({} folds)", path.display(), h.folds);
        srv.ds_history = Some(h);
    }
    if num("--warmup", 1.0)? > 0.0 {
        let tw = Instant::now();
        srv.warmup()?;
        eprintln!("boot: warm-up {:.1} s", tw.elapsed().as_secs_f64());
    }
    // --bench-width TEXT (rank 0, instead of serving): verify passes of each width timed on the
    // file's own continuation - what a longer copy costs - then rank 1 is released
    // --bench-nll JSONL (rank 0, instead of serving): teacher-forced NLL of each line's text
    if let Some(path) = arg(args, "--bench-nll") {
        let text = Text::load(&model_dir)?;
        let keep = num("--hot-keep", 32.0)? as usize;
        // --bench-nll-skip N: prefill each text's first N tokens (default 32) before scoring;
        // --bench-nll-rows A,B,..: the passes' rows, cycled (default 8; 17-33 take the wide decode
        // path past its position); --bench-nll-max: tokens a text is cut at (default 4096)
        let rows: Vec<usize> = arg(args, "--bench-nll-rows").unwrap_or_else(|| "8".into()).split(',')
            .map(|v| v.trim().parse()).collect::<std::result::Result<_, _>>().map_err(|e| format!("--bench-nll-rows: {e}"))?;
        let shape = (num("--bench-nll-skip", 32.0)? as usize, rows, num("--bench-nll-max", 4096.0)? as usize);
        // with --bench-width too, the width bench follows in the same boot
        let last = arg(args, "--bench-width").is_none();
        // --bench-nll-ab KNOB: the NLL twice in this boot, the knob off then on, the dumps to
        // --bench-nll-out's path with .off / .on appended
        if let Some(knobs) = arg(args, "--bench-nll-ab") {
            // KNOB[,KNOB..]: each in turn, with no adaptation in the bench itself; exact with
            // GLM_HOT=0 (hot reads in one arm would change what the next routes to)
            let out = arg(args, "--bench-nll-out");
            let knobs: Vec<&str> = knobs.split(',').map(str::trim).collect();
            for (i, knob) in knobs.iter().enumerate() {
                for on in [false, true] {
                    srv.eng.bench_knob(knob, on)?;
                    eprintln!("bench-nll-ab: {knob} {}", if on { "on" } else { "off" });
                    let o = out.as_ref().map(|p| format!("{p}.{knob}.{}", if on { "on" } else { "off" }));
                    // no adaptation inside the A/B: it moved residents between arms (each run of one
                    // setting differed from the last, the second of a boot 0.027 NLL better)
                    bench_nll(&mut srv.eng, &path, &text, keep, o, &shape, last && on && i + 1 == knobs.len(), false)?;
                }
                srv.eng.bench_knob_reset(knob)?;
            }
        } else {
            bench_nll(&mut srv.eng, &path, &text, keep, arg(args, "--bench-nll-out"), &shape, last, true)?;
        }
        if last {
            return Ok(());
        }
    }
    if let Some(path) = arg(args, "--bench-width") {
        let text = Text::load(&model_dir)?;
        let list = |k: &str, d: &str| -> tidelm_engine::Result<Vec<usize>> {
            Ok(arg(args, k).unwrap_or_else(|| d.into()).split(',').map(|v| v.trim().parse()).collect::<std::result::Result<_, _>>()
                .map_err(|e| format!("{k}: {e}"))?)
        };
        // --bench-ab KNOB[,KNOB..] (forward::BENCH_KNOBS): each context's widths with each knob off and
        // on in turn, --bench-ab-rounds times (ABBA order), one knob after another, in this one boot
        let ab = arg(args, "--bench-ab").map(|k| (k, num("--bench-ab-rounds", 3.0).map(|v| v as usize)));
        let ab = match ab {
            Some((k, r)) => Some((k, r?)),
            None => None,
        };
        // --bench-prefill-ab KNOB[,KNOB..]: the first context's prefill timed with each knob off and on
        // (ABBA, --bench-ab-rounds), before its widths
        let pab = arg(args, "--bench-prefill-ab").map(|k| (k, num("--bench-ab-rounds", 3.0).unwrap_or(3.0) as usize));
        return bench_width(&mut srv.eng, &path, &list("--bench-ctx", "2500")?, &list("--bench-widths", "1,2,4,6,8,9,12,16,17,24,32,33")?,
                           &text, ab, pab);
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
        api_keys: crate::api_keys(),
    });
    let addr = format!("{}:{}", arg(args, "--host").unwrap_or_else(|| "0.0.0.0".into()),
                       arg(args, "--port").unwrap_or_else(|| "8000".into()));
    let listener = TcpListener::bind(&addr)?;
    eprintln!("engine up in {:.1}s; serving {} on {addr} (GLM, depth {}, {})", t0.elapsed().as_secs_f64(), ctx.model,
              srv.opt.depth, match srv.opt.cal { Some(t) => format!("calibrated gate {t}"), None => format!("flat gate {}", srv.opt.dyn_gate) });
    eprintln!("api keys: {}", match ctx.api_keys.len() {
        0 => "none (TIDELM_API_KEY unset: every client is answered)".to_string(),
        n => format!("{n} (Authorization: Bearer or x-api-key; /health open)"),
    });
    tidelm_cuda::mem::mark_up();
    eprintln!("memory at boot: {}", tidelm_cuda::mem::report());
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

/// The engine thread: the scheduler's turns (serve.rs: requests into lanes in arrival order, a
/// round of every decoding lane a turn), cancels and new requests taken between turns. An engine
/// error is fatal: a pair that failed one call cannot go on.
fn engine_loop(mut srv: Server, rx: mpsc::Receiver<Cmd>) -> tidelm_engine::Result<()> {
    loop {
        if !srv.busy() {
            match rx.recv() {
                Ok(Cmd::Submit(j)) => srv.submit(j),
                Ok(Cmd::Cancel(id)) => srv.cancel(id),
                Err(_) => break,
            }
        }
        while let Ok(c) = rx.try_recv() {
            match c {
                Cmd::Submit(j) => srv.submit(j),
                Cmd::Cancel(id) => srv.cancel(id),
            }
        }
        match srv.step() {
            Ok(done) => {
                for f in done {
                    eprintln!("{}", f.line);
                }
            }
            Err(e) => {
                eprintln!("engine failed: {e}");
                let _ = srv.eng.stop_follower();
                std::process::exit(1);
            }
        }
    }
    srv.eng.stop_follower()?;
    Ok(())
}

/// The verify cost by width: each of --bench-ctx's lengths (ascending) of a text file as context,
/// then passes of --bench-widths rows of its continuation, each rolled back (20 timed after 5
/// warm; past 16 rows the path wide_decode_from() picks), and the MTP draft pass a copy round
/// skips.
fn bench_width(eng: &mut Engine, path: &str, ctxs: &[usize], widths: &[usize], text: &Text, ab: Option<(String, usize)>,
               pab: Option<(String, usize)>) -> tidelm_engine::Result<()> {
    let ids = text.encode(&std::fs::read_to_string(path)?).map_err(|e| format!("{path}: {e:?}"))?;
    let wmax = widths.iter().copied().max().unwrap_or(1);
    eng.set_sampling(None)?;
    eng.set_pos(0)?;
    for &ctx in ctxs {
        if ids.len() < ctx + wmax || ctx <= eng.pos {
            return Err(format!("{path}: {} tokens; contexts must ascend and leave {wmax}", ids.len()).into());
        }
        if let (Some((knobs, rounds)), true) = (&pab, ctx == ctxs[0]) {
            // the whole context prefilled from position 0 each time, lane 0's rows rewritten
            for knob in knobs.split(',').map(str::trim) {
                let mut by: [Vec<f64>; 2] = [vec![], vec![]];
                for r in 0..*rounds {
                    for on in if r % 2 == 0 { [false, true] } else { [true, false] } {
                        eng.bench_knob(knob, on)?;
                        eng.set_pos(0)?;
                        let t = Instant::now();
                        eng.forward(&ids[..ctx], false)?;
                        let dt = t.elapsed().as_secs_f64();
                        eprintln!("bench-prefill-ab: {knob} {}: {ctx} tokens in {dt:.3} s", if on { "on" } else { "off" });
                        by[on as usize].push(dt);
                    }
                }
                eng.bench_knob_reset(knob)?;
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                let f = |v: &[f64]| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" / ");
                let (a, b) = (mean(&by[0]), mean(&by[1]));
                eprintln!("bench-prefill-ab: {knob} context {ctx}: off {a:.3} s ({}), on {b:.3} s ({}), {:+.1}%", f(&by[0]), f(&by[1]),
                          100.0 * (b - a) / a);
            }
            eng.set_pos(0)?;
        }
        let t = Instant::now();
        eng.forward(&ids[eng.pos..ctx], false)?;
        eprintln!("bench-width: context to {ctx} tokens prefilled in {:.3} s", t.elapsed().as_secs_f64());
        eng.keep_hidden(eng.last_rows - 1)?;
        let Some((knobs, rounds)) = &ab else {
            bench_widths(eng, &ids, ctx, widths)?;
            continue;
        };
        for knob in knobs.split(',').map(str::trim) {
            // [arm][width (the MTP pass last)] -> each round's median
            let mut by: [Vec<Vec<f64>>; 2] = [vec![vec![]; widths.len() + 1], vec![vec![]; widths.len() + 1]];
            for r in 0..*rounds {
                for on in if r % 2 == 0 { [false, true] } else { [true, false] } {
                    eng.bench_knob(knob, on)?;
                    eprintln!("bench-ab: {knob} {}, round {}", if on { "on" } else { "off" }, r + 1);
                    for (i, m) in bench_widths(eng, &ids, ctx, widths)?.into_iter().enumerate() {
                        by[on as usize][i].push(m);
                    }
                }
            }
            // the next knob is measured against the boot's own setting of this one
            eng.bench_knob_reset(knob)?;
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            for (i, name) in widths.iter().map(|w| format!("{w:2} rows")).chain(["MTP pass".to_string()]).enumerate() {
                let (a, b) = (mean(&by[0][i]), mean(&by[1][i]));
                let f = |v: &[f64]| v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" / ");
                eprintln!("bench-ab: {knob} context {ctx}: {name}: off {a:.2} ms ({}), on {b:.2} ms ({}), {:+.1}%", f(&by[0][i]),
                          f(&by[1][i]), 100.0 * (b - a) / a);
            }
        }
    }
    eng.stop_follower()?;
    Ok(())
}

/// Each width's verify median, then the MTP pass's (ms), as printed.
fn bench_widths(eng: &mut Engine, ids: &[u32], ctx: usize, widths: &[usize]) -> tidelm_engine::Result<Vec<f64>> {
    let base = eng.pos;
    let median = |mut ts: Vec<f64>| -> (f64, f64, f64) {
        ts.sort_by(f64::total_cmp);
        (ts[ts.len() / 2], ts[0], ts[ts.len() - 1])
    };
    eprintln!("bench-width: context {base} tokens");
    let mut out = vec![];
    for &w in widths {
        let block = &ids[ctx..ctx + w];
        let mut ts = Vec::new();
        for i in 0..25 {
            let t = Instant::now();
            eng.forward(block, false)?;
            let dt = t.elapsed().as_secs_f64() * 1e3;
            eng.set_pos(base)?;
            if i >= 5 {
                ts.push(dt);
            }
        }
        let (m, lo, hi) = median(ts);
        eprintln!("bench-width: {w:2} rows: verify {m:.2} ms (min {lo:.2}, max {hi:.2})");
        out.push(m);
    }
    let mut ts = Vec::new();
    for i in 0..25 {
        let t = Instant::now();
        eng.mtp(&[ids[ctx]], tidelm_glm::forward::Hidden::Last, base - 1, true)?;
        if i >= 5 {
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let (m, lo, hi) = median(ts);
    eprintln!("bench-width: MTP draft pass {m:.2} ms (min {lo:.2}, max {hi:.2})");
    out.push(m);
    Ok(out)
}

/// Teacher-forced NLL of each JSONL line's "text" (by its "category") through the serving path:
/// a 32-token prefill and its adaptation, then passes of 8 rows as a verify runs them (graphs, hot
/// reads, the background period's victims every 2 passes, an adaptation every 200 tokens). A row's
/// next token's -log p at T = 1 comes from the pair's candidates, or the whole row when it is not
/// among them; top-1 is the row's argmax. Texts are cut at 4,096 tokens.
fn bench_nll(eng: &mut Engine, path: &str, text: &Text, keep: usize, out: Option<String>,
             (skip, rows, max): &(usize, Vec<usize>, usize), last: bool, adapt: bool) -> tidelm_engine::Result<()> {
    eng.set_sampling(Some(1.0))?;
    // --bench-nll-out: each scored token's "category nll top1" a line, for paired comparisons
    let mut dump = out.map(std::fs::File::create).transpose()?;
    let mut acc: std::collections::BTreeMap<String, (f64, f64, f64, usize)> = Default::default();
    let t0 = Instant::now();
    let mut nans = 0usize;
    for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let cat = v["category"].as_str().unwrap_or("all").to_string();
        let ids = text.encode(v["text"].as_str().unwrap_or("")).map_err(|e| format!("{path}: {e:?}"))?;
        if ids.len() < skip + 16 {
            continue;
        }
        let ids = &ids[..ids.len().min(*max)];
        eng.set_pos(0)?;
        eng.forward(&ids[..*skip], false)?;
        if adapt {
            eng.adapt(512)?;
        }
        let (mut i, mut since, mut passes) = (*skip, 0usize, 0usize);
        let a = acc.entry(cat).or_default();
        while i + 1 < ids.len() {
            let n = rows[passes % rows.len()].min(ids.len() - 1 - i);
            let am = eng.forward(&ids[i..i + n], false)?;
            for r in 0..n {
                let t = ids[i + r + 1];
                let rc = eng.candidates(r);
                if rc.vals.iter().chain(rc.lse.iter().map(|x| &x.0)).any(|v| v.is_nan()) {
                    nans += 1;
                    eprintln!("bench-nll: NaN in the candidates of row {r} of a {n}-row pass at position {i}");
                }
                let m = rc.lse.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
                let z: f64 = rc.lse.iter().map(|&(mr, s)| s as f64 * ((mr - m) as f64).exp()).sum();
                let zt = match rc.ids.iter().position(|&x| x == t) {
                    Some(j) => rc.vals[j] as f64,
                    None => eng.full_row(r)?[t as usize] as f64,
                };
                let nll = m as f64 + z.ln() - zt;
                if let Some(f) = dump.as_mut() {
                    use std::io::Write;
                    writeln!(f, "{} {nll:.6} {}", v["category"].as_str().unwrap_or("all"), (am[r] == t) as u8)?;
                }
                a.0 += nll;
                a.1 += nll * nll;
                a.2 += (am[r] == t) as u8 as f64;
                a.3 += 1;
            }
            i += n;
            since += n;
            passes += 1;
            if passes % 2 == 0 && eng.hot.is_some() {
                eng.bg_step(keep)?;
            }
            if since >= 200 {
                if adapt {
                    eng.adapt(256)?;
                }
                since = 0;
            }
        }
    }
    let (mut tn, mut tsum) = (0usize, 0f64);
    for (cat, (s, s2, top, n)) in &acc {
        let mean = s / *n as f64;
        let sd = (s2 / *n as f64 - mean * mean).max(0.0).sqrt();
        eprintln!("bench-nll: {cat}: {n} tokens, NLL {mean:.4} (sd {sd:.3}), top-1 {:.2}%", 100.0 * top / *n as f64);
        tn += n;
        tsum += s;
    }
    eprintln!("bench-nll: all: {tn} tokens, NLL {:.4}, {:.0} s, {nans} rows with NaN", tsum / tn.max(1) as f64,
              t0.elapsed().as_secs_f64());
    if last {
        eng.stop_follower()?;
    }
    Ok(())
}
