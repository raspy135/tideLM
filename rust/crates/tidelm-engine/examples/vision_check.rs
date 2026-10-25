//! The vision tower against the Python engine's on the same patches: `vision_check DIR`, where
//! DIR holds a reference run (tidelm:local, the checkpoint's image_processor and
//! engine/vision.py's loader): meta.json, NAME.patches (bf16) and NAME.rows (bf16, the
//! aligner's output) for the images that have them. Prints per image the rows' agreement
//! (cosine, relative error) and the tower's time.

use std::path::PathBuf;

use tidelm_cuda::Stream;
use tidelm_engine::config::Config;
use tidelm_engine::vision::Tower;
use tidelm_model::Checkpoint;

fn bf(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn words(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn main() -> tidelm_engine::Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).ok_or("usage: vision_check DIR")?);
    let model = std::env::var("TIDELM_MODEL_DIR").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017"));
    let cfg = Config::load(&model)?;
    let ckpt = Checkpoint::open(&model)?;
    let stream = Stream::new()?;
    let t = std::time::Instant::now();
    let tower = Tower::load(&ckpt, &stream, &cfg)?;
    println!("tower: {:.2} GB in {:.1} s", tower.bytes() as f64 / 1e9, t.elapsed().as_secs_f64());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    for m in meta.as_array().unwrap() {
        let name = m["name"].as_str().unwrap();
        let Ok(want) = std::fs::read(dir.join(format!("{name}.rows"))) else { continue };
        let (vh, vw) = (m["vit_h"].as_u64().unwrap() as usize, m["vit_w"].as_u64().unwrap() as usize);
        let patches = words(&std::fs::read(dir.join(format!("{name}.patches")))?);
        tower.forward(&stream, &patches, vh, vw)?; // warm
        let t = std::time::Instant::now();
        let rows = tower.forward(&stream, &patches, vh, vw)?;
        let dt = t.elapsed().as_secs_f64();
        let got = rows.download()?;
        let want = words(&want);
        if got.len() != want.len() {
            return Err(format!("{name}: {} values, the reference has {}", got.len(), want.len()).into());
        }
        let d = cfg.dim;
        let (mut worst, mut num, mut den, mut same) = (1.0f64, 0.0f64, 0.0f64, 0usize);
        for (g, w) in got.chunks(d).zip(want.chunks(d)) {
            let (mut gw, mut gg, mut ww) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in g.iter().zip(w) {
                let (a, b) = (bf(a) as f64, bf(b) as f64);
                gw += a * b;
                gg += a * a;
                ww += b * b;
                num += (a - b) * (a - b);
                den += b * b;
            }
            worst = worst.min(gw / (gg.sqrt() * ww.sqrt()).max(1e-30));
        }
        same += got.iter().zip(&want).filter(|(a, b)| a == b).count();
        println!("{name}: {} rows, worst row cosine {worst:.6}, relative error {:.4}, {:.1}% of values equal, tower {dt:.3} s \
                  (Python {:.3} s)",
                 got.len() / d, (num / den).sqrt(), 100.0 * same as f64 / got.len() as f64, m["tower_s"].as_f64().unwrap_or(0.0));
    }
    Ok(())
}
