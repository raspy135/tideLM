//! Bring up the GLM engine (one box, or the pair) and run a prompt greedily: the first token's
//! latency, then plain or MTP-speculative decode, its rate and text. A check and a benchmark,
//! not a server.
//!
//!   glm_run --model-dir DIR --pack PACK_{rank} --db HISTORY [--world 2 --rank R --addr H:P]
//!           [--max-seq 262144] [--chunk 4096] [--slots 12500] [--prompt TEXT] [--max-tokens 64]
//!           [--depth 0..7]

use std::path::PathBuf;
use std::time::Instant;

use tidelm_glm::forward::{Engine, Hidden, Options, Pair, ROCE_BYTES};
use tidelm_glm::Result;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = PathBuf::from(arg(&args, "--model-dir").ok_or("--model-dir")?);
    let world: usize = arg(&args, "--world").map(|s| s.parse()).transpose()?.unwrap_or(1);
    let rank: usize = arg(&args, "--rank").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let pack = arg(&args, "--pack").ok_or("--pack")?.replace("{rank}", &rank.to_string());
    let opt = Options {
        max_seq: arg(&args, "--max-seq").map(|s| s.parse()).transpose()?.unwrap_or(32768),
        chunk: arg(&args, "--chunk").map(|s| s.parse()).transpose()?.unwrap_or(4096),
        slots: arg(&args, "--slots").map(|s| s.parse()).transpose()?.unwrap_or(12500),
        kv_split: world > 1 && arg(&args, "--kv-split").as_deref() != Some("0"),
        prefill_reduce_bf16: arg(&args, "--reduce").as_deref() != Some("fp32"),
        moe_rows: arg(&args, "--moe-rows").map(|s| s.parse()).transpose()?.unwrap_or(2048),
        pack: PathBuf::from(&pack),
        guard_extra: format!("graphs {} hot {}", args.iter().any(|a| a == "--graphs"), args.iter().any(|a| a == "--hot")),
    };
    let log = move |s: &str| {
        if rank == 0 {
            eprintln!("{s}");
        }
    };
    tidelm_cuda::set_device(0)?;
    let cfg = tidelm_glm::config::GlmConfig::load(&model_dir)?;
    let e = cfg.experts;
    let mut pair = None;
    if world == 2 {
        let p = tidelm_glm::arena::Pack::open(&opt.pack, cfg.hidden, cfg.moe_inter)?;
        let boot = opt.boot_config(&model_dir, &p)?;
        let addr = arg(&args, "--addr").unwrap_or_else(|| "10.0.0.1:29711".into());
        let (comm, mut ctl) = tidelm_comm::join(rank, &addr, &boot, std::time::Duration::from_secs(600))?;
        let hcas: Option<Vec<String>> = std::env::var("NCCL_IB_HCA").ok().map(|v| v.split(',').map(|s| s.to_string()).collect());
        let roce = tidelm_comm::roce::Roce::setup(rank, &mut ctl, ROCE_BYTES, hcas.as_deref())?;
        pair = Some(Pair { comm, ctl, roce });
    }
    // the residents: rank 0 places them from the history and sends them to rank 1
    let swap: Vec<usize> = (0..cfg.layers).filter(|&l| !cfg.dense[l]).collect();
    let residents: Vec<(usize, usize)> = if rank == 0 {
        let db = PathBuf::from(arg(&args, "--db").ok_or("--db")?);
        let mass = tidelm_glm::residency::read_mass(&db, cfg.n_layers(), e)?;
        let r = tidelm_glm::residency::placement(&mass, e, &swap, opt.slots, 2 * cfg.topk);
        if let Some(p) = pair.as_mut() {
            let mut m = tidelm_comm::Msg::default();
            m.u32s(&r.iter().map(|&(l, x)| (l * e + x) as u32).collect::<Vec<_>>());
            p.ctl.send(&m.0)?;
        }
        r
    } else {
        let msg = pair.as_mut().ok_or("rank 1 needs the pair")?.ctl.recv()?;
        tidelm_comm::Reader::new(&msg).u32s()?.into_iter().map(|v| (v as usize / e, v as usize % e)).collect()
    };
    let t0 = Instant::now();
    let mut eng = Engine::load(&model_dir, opt, rank, world, pair, &residents, &log)?;
    log(&format!("[glm] engine up in {:.0} s", t0.elapsed().as_secs_f64()));
    eng.graphs = args.iter().any(|a| a == "--graphs");
    if args.iter().any(|a| a == "--hot") {
        eng.enable_hot(false)?;
    }
    if rank == 1 {
        return eng.follow();
    }
    let text = tidelm_text::Text::load(&model_dir)?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| "Write a short poem about the sea.".into());
    let msgs = vec![serde_json::json!({"role": "user", "content": prompt})];
    let rendered = tidelm_text::glm::render(&msgs, None, Some("low"))?;
    let mut ids = text.encode(&rendered)?;
    if let Some(n) = arg(&args, "--random-prompt").map(|v| v.parse::<usize>()).transpose()? {
        // bench.py's random workload: n tokens drawn uniformly from a fixed seed
        let mut x = 0x9e3779b97f4a7c15u64;
        ids = (0..n).map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (1000 + x % 100000) as u32
        }).collect();
    }
    let max_tokens: usize = arg(&args, "--max-tokens").map(|s| s.parse()).transpose()?.unwrap_or(64);
    let depth: usize = arg(&args, "--depth").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let stop = cfg.eos.clone();
    let run = |eng: &mut Engine| -> Result<()> {
        eng.set_pos(0)?;
        let t1 = Instant::now();
        let first = eng.forward(&ids, depth > 0)?;
        let ttft = t1.elapsed().as_secs_f64();
        let mut out = vec![first[0]];
        let mut anchor = first[0];
        eng.keep_hidden(eng.last_rows - 1)?;
        let t2 = Instant::now();
        let (mut rounds, mut accepted) = (0usize, 0usize);
        let (mut t_draft, mut t_verify, mut t_fill) = (0f64, 0f64, 0f64);
        while out.len() < max_tokens && !stop.contains(&anchor) {
            rounds += 1;
            if depth == 0 {
                anchor = eng.forward(&[anchor], false)?[0];
                out.push(anchor);
                continue;
            }
            let p = eng.pos;
            let ta = Instant::now();
            let mut drafts = vec![eng.mtp(&[anchor], Hidden::Last, p - 1, true)?.unwrap()[0]];
            for k in 1..depth {
                let d = eng.mtp(&[drafts[k - 1]], Hidden::Mtp, p - 1 + k, true)?.unwrap()[0];
                drafts.push(d);
            }
            let mut block = vec![anchor];
            block.extend_from_slice(&drafts);
            let tb = Instant::now();
            let am = eng.forward(&block, false)?;
            let tc = Instant::now();
            let n_acc = (0..depth).find(|&k| am[k] != drafts[k]).unwrap_or(depth);
            accepted += n_acc;
            eng.set_pos(p + 1 + n_acc)?;
            if n_acc > 0 {
                eng.mtp(&drafts[..n_acc], Hidden::Target(0), p, false)?;
            }
            eng.keep_hidden(n_acc)?;
            t_draft += (tb - ta).as_secs_f64();
            t_verify += (tc - tb).as_secs_f64();
            t_fill += tc.elapsed().as_secs_f64();
            for &d in &drafts[..n_acc] {
                out.push(d);
                if stop.contains(&d) {
                    break;
                }
            }
            anchor = am[n_acc];
            if !drafts[..n_acc].iter().any(|d| stop.contains(d)) {
                out.push(anchor);
            } else {
                break;
            }
        }
        let dt = t2.elapsed().as_secs_f64();
        log(&format!("prompt {} tokens: first token {:.3} s; {} tokens in {:.2} s = {:.2} tok/s ({} rounds, {:.2} accepted a round)",
                     ids.len(), ttft, out.len(), dt, (out.len() - 1) as f64 / dt, rounds,
                     accepted as f64 / rounds.max(1) as f64));
        log(&format!("per round: drafts {:.1} ms, verify {:.1} ms, fill {:.1} ms", 1e3 * t_draft / rounds as f64,
                     1e3 * t_verify / rounds as f64, 1e3 * t_fill / rounds as f64));
        let r = &eng.res.stats;
        log(&format!("hot reads {} (kept {}, unkept {}, {:.3} s), swaps {}", r.hot_reads, r.hot_kept, r.hot_unkept, r.hot_s, r.swaps));
        log(&format!("--- {}", text.decode(&out)?));
        log(&format!("ids {:?}", &out[..out.len().min(48)]));
        Ok(())
    };
    if args.iter().any(|a| a == "--check") {
        // teacher-forced next-token accuracy on a known text: the decode path one row at a time,
        // in 8-row blocks, and the prefill path (a prefix at a time, the last row's argmax)
        let txt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. \
                   One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen.";
        let ids = text.encode(txt)?;
        let n = ids.len();
        let score = |pred: &[u32], from: usize| -> String {
            let hit = (from..n - 1).filter(|&i| pred[i] == ids[i + 1]).count();
            format!("{hit}/{}", n - 1 - from)
        };
        eng.set_pos(0)?;
        let mut one = vec![0u32; n];
        for i in 0..n {
            one[i] = eng.forward(&ids[i..i + 1], false)?[0];
        }
        log(&format!("decode 1 row:  {}  {:?}", score(&one, 0), text.decode(&one[..24.min(n)])?));
        eng.set_pos(0)?;
        let mut blk = vec![0u32; n];
        for c in (0..n).step_by(8) {
            let e = (c + 8).min(n);
            let r = eng.forward(&ids[c..e], false)?;
            blk[c..e].copy_from_slice(&r);
        }
        log(&format!("decode 8 rows: {}  {:?}", score(&blk, 0), text.decode(&blk[..24.min(n)])?));
        let mut pre = vec![0u32; n];
        for k in 17..=n {
            eng.set_pos(0)?;
            pre[k - 1] = eng.forward(&ids[..k], false)?[0];
        }
        log(&format!("prefill:       {}  (from row 16)", score(&pre, 16)));
        log(&format!("decode 1 row:  {}  (from row 16)", score(&one, 16)));
        eng.stop_follower()?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--bench-rows") {
        // a pass's cost by its rows after a prompt: verify widths 1..8 and a one-row MTP draft
        let first = eng.forward(&ids, true)?;
        eng.keep_hidden(eng.last_rows - 1)?;
        let p = eng.pos;
        let filler: Vec<u32> = (0..8).map(|i| ids[(i * 7) % ids.len()]).collect();
        let _ = first;
        let mut line = String::new();
        let tmax: usize = std::env::var("GLM_BENCH_T").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        for t in 1..=tmax {
            let mut ts = vec![];
            for i in 0..13 {
                let a = Instant::now();
                eng.forward(&filler[..t], false)?;
                if i >= 3 {
                    ts.push(a.elapsed().as_secs_f64() * 1e3);
                }
                eng.set_pos(p)?;
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            line += &format!(" T{t} {:.1}", ts[ts.len() / 2]);
        }
        let mut ts = vec![];
        for i in 0..13 {
            let a = Instant::now();
            eng.mtp(&filler[..1], Hidden::Last, p - 1, true)?;
            if i >= 3 {
                ts.push(a.elapsed().as_secs_f64() * 1e3);
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        log(&format!("pass ms (median of 10):{line}  mtp draft {:.2}", ts[ts.len() / 2]));
        eng.stop_follower()?;
        return Ok(());
    }
    let reps: usize = arg(&args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(1);
    for _ in 0..reps {
        run(&mut eng)?;
    }
    eng.stop_follower()?;
    Ok(())
}
