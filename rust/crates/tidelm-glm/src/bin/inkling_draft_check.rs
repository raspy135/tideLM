//! Inkling 2's drafter (inkling_draft: mtp.safetensors, "DSpark") on fabricated target features,
//! for tools/inkling_draft_ref.py to compare with a PyTorch transcription of the specification
//! (docs/inkling-dspark.md, section 8).
//!
//!   inkling_draft_check --model-dir DIR --out OUTDIR [--fp8] [--listed] [--ctx 40] [--first 33] [--start 0]
//!                       [--anchor 1000] [--seed 1] [--local-ring 2048] [--global-ring 4096]
//!                       [--max-rows 64] [--time N]
//!
//! A sequence of --ctx committed positions (--start .. --start + ctx) in lane 1 of 2, its 8 taps a
//! position pseudo-random from --seed (OUTDIR/taps.bin: bf16 [8][ctx][4608], tap-major). The
//! context is stored in two passes - --first rows, then the rest behind 3 rows of another lane -
//! and a third pass leaves 2 rows at the positions after it, as a verify pass's rejected rows
//! would. One draft round at p = start + ctx with --anchor follows (its normed embedding row
//! computed here from the checkpoint and passed as a one-row table). Written to OUTDIR:
//!   e.NAME.bin   the drafter's TIDELM_DRAFT_DUMP files: alpha, fold_w (first 4,096 rows), fold_b;
//!                cP.L the mixed and normed features of pass P for layer L (5: f_rnn's); x0; per layer
//!                qL (queries), kvL (the keys / values read: context, then the block's 8), oL (the
//!                heads' outputs), resL (the residual after the layer); hidden (the final normed
//!                rows); base (rows 1 .. 7's logits); zx ([h_i | f_rnn]); per step stateI, latentI, yI
//!                (the corrected, softcapped, cropped logits, fp32)
//!   draft.json   the arguments, the 7 tokens, their probabilities, each step's top 5
//! --time N instead: no dump, N draft rounds timed (the first is cold), nothing compared.
//! Device memory: the drafter's (about 5.5 GB as BF16, 2.9 GB as FP8) - not beside a serving engine.

use std::os::raw::c_void;
use std::path::PathBuf;
use std::time::Instant;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_glm::inkling_draft::{self, DraftOpts, InkDraft, D, DRAFTS, TAPS};
use tidelm_glm::Result;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn num(args: &[String], name: &str, default: usize) -> Result<usize> {
    Ok(arg(args, name).map(|s| s.parse()).transpose().map_err(|e| format!("{name}: {e}"))?.unwrap_or(default))
}

fn to_bf(f: f32) -> u16 {
    let b = f.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// `rows` positions' taps, bf16 [8][rows][4608]: per tap a smooth wave over (row, channel) plus
/// uniform noise (splitmix64 from `seed`), scaled 0.5 .. 2.25 by tap - the first 7 stand for
/// un-normed block residuals, the last for a normed hidden state.
fn synthetic(rows: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    let mut uniform = move || -> f32 {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let mut v = Vec::with_capacity(TAPS * rows * D);
    for tap in 0..TAPS {
        let scale = if tap + 1 == TAPS { 1.0 } else { 0.5 + 0.25 * tap as f32 };
        for r in 0..rows {
            for c in 0..D {
                let wave = (0.37 * r as f32 + 0.011 * c as f32 + tap as f32).sin();
                v.push(to_bf(scale * (0.5 * wave + uniform())));
            }
        }
    }
    v
}

fn le<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// One pass: `groups` of (lane, first position, rows of `src` from `from`), `src` bf16 [8][n][4608],
/// staged tap-major with `plane` elements a tap, the lane tables as a target pass has them.
#[allow(clippy::too_many_arguments)]
fn pass(d: &InkDraft, st: &Stream, taps: &DeviceBuffer<u16>, plane: usize, lanes: &DeviceBuffer<i32>, meta: &DeviceBuffer<i32>,
        groups: &[(usize, usize, &[u16], usize, usize)]) -> Result<()> {
    let (mut lt, mut mt, mut row) = (vec![0i32; 4 * groups.len()], vec![0i32; 4 * groups.len()], 0usize);
    for (z, &(lane, pos, src, from, n)) in groups.iter().enumerate() {
        let total = src.len() / (TAPS * D);
        for tap in 0..TAPS {
            let s = &src[(tap * total + from) * D..(tap * total + from + n) * D];
            taps.upload_async(tap * plane + row * D, s, st)?;
        }
        lt[4 * z..4 * z + 4].copy_from_slice(&[row as i32, n as i32, pos as i32, 0]);
        mt[4 * z] = lane as i32;
        row += n;
    }
    lanes.upload_async(0, &lt, st)?;
    meta.upload_async(0, &mt, st)?;
    d.store_context(st, taps.raw(), plane, row, lanes.raw(), groups.len(), meta.raw())?;
    st.synchronize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--model-dir").ok_or("--model-dir")?);
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out")?);
    let fp8 = args.iter().any(|a| a == "--fp8");
    let ctx = num(&args, "--ctx", 40)?;
    let first = num(&args, "--first", 33.min(ctx.saturating_sub(1)).max(1))?;
    let start = num(&args, "--start", 0)?;
    let anchor = num(&args, "--anchor", 1000)?;
    let seed = num(&args, "--seed", 1)? as u64;
    let local_ring = num(&args, "--local-ring", 2048)?;
    let global_ring = num(&args, "--global-ring", 4096)?;
    let max_rows = num(&args, "--max-rows", 64)?;
    let time = num(&args, "--time", 0)?;
    const DECOY: usize = 3;
    const LANE: usize = 1;
    if ctx < 2 || first == 0 || first >= ctx || first > max_rows || ctx - first + DECOY > max_rows {
        return Err(format!("--ctx {ctx} in passes of {first} and {} rows (+ {DECOY}) with --max-rows {max_rows}", ctx - first.min(ctx)).into());
    }
    std::fs::create_dir_all(&out)?;
    let prefix = out.join("e");
    if time == 0 {
        std::env::set_var("TIDELM_DRAFT_DUMP", &prefix);
    } else {
        std::env::remove_var("TIDELM_DRAFT_DUMP");
    }
    let taps_host = synthetic(ctx, seed);
    let decoy = synthetic(DECOY, seed ^ 0x5eed_dec0);
    std::fs::write(out.join("taps.bin"), le(&taps_host))?;
    let row = inkling_draft::normed_embedding_row(&dir, anchor)?;

    tidelm_cuda::set_device(0)?;
    let st = Stream::new()?;
    let t0 = Instant::now();
    // --listed: the built-in listed vocabulary too; with --time, rounds alternate over every id and
    // over the list (each timed), and the last round of each is compared
    let listed = args.iter().any(|a| a == "--listed");
    let ids = if listed { inkling_draft::listed_ids(200058)? } else { None };
    let d = InkDraft::open_with(&dir, &st, DraftOpts { fp8, lanes: 2, local_ring, global_ring, max_rows }, ids.as_deref())?;
    let b = d.bytes();
    let mb = |n: usize| n as f64 / 1e6;
    eprintln!("drafter loaded in {:.1} s ({}): {:.1} MB on the device - layers {:.1}, unembed {:.1}, head {:.1}, rings of 2 lanes \
               {:.1}, scratch {:.1}", t0.elapsed().as_secs_f64(), if fp8 { "fp8" } else { "bf16" }, mb(d.device_bytes()), mb(b.layers),
              mb(b.unembed), mb(b.markov), mb(b.rings), mb(b.scratch));

    let plane = max_rows * D;
    let taps = DeviceBuffer::<u16>::zeroed(TAPS * plane)?;
    let (lanes, meta) = (DeviceBuffer::<i32>::zeroed(4 * 2)?, DeviceBuffer::<i32>::zeroed(4 * 2)?);
    let p = start + ctx;
    // the context in two passes (the second behind another lane's rows), then two rows at p and
    // p + 1 as a verify pass's rejected rows leave them: the block must overwrite, not read, them
    pass(&d, &st, &taps, plane, &lanes, &meta, &[(LANE, start, &taps_host[..], 0, first)])?;
    pass(&d, &st, &taps, plane, &lanes, &meta, &[(0, 7, &decoy[..], 0, DECOY), (LANE, start + first, &taps_host[..], first, ctx - first)])?;
    pass(&d, &st, &taps, plane, &lanes, &meta, &[(LANE, p, &decoy[..], 0, 2)])?;

    // a one-row embedding table: the pointer the anchor's row is `anchor` rows past
    let erow = DeviceBuffer::from_slice(&row)?;
    let embed = (erow.raw() as *const u8).wrapping_sub(anchor * D * 2) as *const c_void;
    let toks = DeviceBuffer::<i32>::zeroed(DRAFTS)?;
    let probs = DeviceBuffer::<f32>::zeroed(DRAFTS)?;
    let mut sums = [(0f64, 0usize); 2];
    let mut last = [(vec![], vec![]), (vec![], vec![])];
    for rep in 0..time.max(1) * if listed { 2 } else { 1 } {
        let over_list = listed && rep % 2 == 1;
        let t1 = Instant::now();
        d.draft(&st, LANE, anchor as u32, p, start, embed, toks.raw(), probs.raw(), over_list)?;
        st.synchronize()?;
        let ms = 1e3 * t1.elapsed().as_secs_f64();
        if time > 0 {
            eprintln!("round {rep}{}: {ms:.2} ms", if over_list { " (listed)" } else { "" });
            if rep >= 2 {
                sums[over_list as usize].0 += ms;
                sums[over_list as usize].1 += 1;
            }
        }
        last[over_list as usize] = (toks.download()?, probs.download()?);
    }
    if listed {
        let avg = |i: usize| sums[i].0 / sums[i].1.max(1) as f64;
        eprintln!("every id: {:.2} ms a round, tokens {:?} probabilities {:?}", avg(0), last[0].0, last[0].1);
        eprintln!("listed:   {:.2} ms a round, tokens {:?} probabilities {:?}", avg(1), last[1].0, last[1].1);
    }
    let (tk, pr) = (toks.download()?, probs.download()?);
    eprintln!("anchor {anchor} at position {p} (context from {start}): tokens {tk:?}, probabilities {pr:?}");
    if time > 0 {
        return Ok(());
    }
    // each step's top 5 from its dumped logits
    let mut top = vec![];
    let mut bad = 0usize;
    for i in 0..DRAFTS {
        let y: Vec<f32> = std::fs::read(format!("{}.y{i}.bin", prefix.display()))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        bad += y.iter().filter(|v| v.is_nan()).count();
        let mut idx: Vec<usize> = (0..y.len()).collect();
        // the largest first, the lowest id on ties
        idx.sort_by(|&a, &b| y[b].total_cmp(&y[a]).then(a.cmp(&b)));
        top.push(idx[..5].iter().map(|&j| serde_json::json!({ "id": j, "y": y[j] })).collect::<Vec<_>>());
        if idx[0] as i32 != tk[i] {
            return Err(format!("step {i}: token {} but the dumped logits' argmax is {}", tk[i], idx[0]).into());
        }
    }
    let j = serde_json::json!({
        "anchor": anchor, "p": p, "lo": start, "start": start, "ctx": ctx, "first": first, "decoy": DECOY, "lane": LANE,
        "seed": seed, "fp8": fp8, "local_ring": local_ring, "global_ring": global_ring, "max_rows": max_rows,
        "tokens": tk, "probs": pr, "top5": top,
    });
    std::fs::write(out.join("draft.json"), serde_json::to_string_pretty(&j)?)?;
    eprintln!("wrote {} and {}.*.bin", out.join("draft.json").display(), prefix.display());
    if bad > 0 || pr.iter().any(|v| !(*v > 0.0 && *v <= 1.0 + 1e-5)) {
        return Err(format!("{bad} NaN logits, probabilities {pr:?}").into());
    }
    Ok(())
}
