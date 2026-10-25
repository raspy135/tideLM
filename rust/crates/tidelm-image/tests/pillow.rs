//! tidelm_image against the checkpoint's image_processor.load_image (Pillow 13, libjpeg-turbo
//! 3.2 in tidelm:local) on the images of tests/pillow (or TIDELM_VREF: a gen_reference.py run),
//! whose meta.json holds each image's grid and the sha256 of its bf16 patches: PNG in RGB,
//! RGBA, palette, gray, 16-bit gray; JPEG 4:2:0, 4:2:2 and gray; GIF; WebP; upscaled, padded
//! and tall plans.

use sha2::{Digest, Sha256};
use tidelm_image::{prepare, Plan};

#[test]
fn patches_are_pillows_bit_for_bit() {
    let dir = std::env::var("TIDELM_VREF").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pillow"));
    let meta: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let p = Plan { patch: 14, ratio: 3, max_tokens: 1024, min_pixels: 295_936 };
    let mut bad = Vec::new();
    for m in &meta {
        let name = m["name"].as_str().unwrap();
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let got = match prepare(&bytes, &p) {
            Ok(g) => g,
            Err(e) => {
                bad.push(format!("{name}: {e}"));
                continue;
            }
        };
        let grid = (got.vit_h as u64, got.vit_w as u64);
        if grid != (m["vit_h"].as_u64().unwrap(), m["vit_w"].as_u64().unwrap()) {
            bad.push(format!("{name}: grid {grid:?}, Pillow's {} x {}", m["vit_h"], m["vit_w"]));
            continue;
        }
        let raw: Vec<u8> = got.patches.iter().flat_map(|x| x.to_le_bytes()).collect();
        let sha = format!("{:x}", Sha256::digest(&raw));
        if sha != m["sha256"].as_str().unwrap() {
            // a reference run also has the patches: say how many values differ
            let diff = std::fs::read(dir.join(format!("{name}.patches")))
                .map(|want| raw.chunks(2).zip(want.chunks(2)).filter(|(a, b)| a != b).count().to_string())
                .unwrap_or_else(|_| "some".into());
            bad.push(format!("{name}: {diff} of {} values differ", raw.len() / 2));
        } else {
            eprintln!("{name}: equal");
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
