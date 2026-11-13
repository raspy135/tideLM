//! A GLM-5.3-Flash request's images on the engine: encoded before its prompt prefills, with the
//! vision tower in lent expert slots (lend.rs), into a feature buffer allocated for the request;
//! the prompt's image rows then take those features in place of the `<|image|>` embedding
//! (flash_fwd embed_flash), and the slots go back to experts before the prefill.
//!
//! Each lane keeps its own: a lane prefilling in parts beside a decoding lane must keep its
//! features while the other lane starts a request (with images or without). A lane also
//! remembers the images its sequence holds (position, rows, the patches' key), so a follow-up
//! turn whose prompt extends the sequence reuses the prefix past an image it shares instead of
//! prefilling (and encoding) again from the first image.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tidelm_cuda::DeviceBuffer;
use tidelm_engine::vision::ImageIn;

use crate::flash_vision::{bytes_for_regions, Mem, Tower};
use crate::forward::Engine;
use crate::Result;

pub const ROW: usize = 4096;

/// MemAvailable an image request's features must leave: memguard stops the server below 4 GiB.
/// (/proc/meminfo, not cudaMemGetInfo: on the Spark's shared memory that reports about MemFree,
/// leaving out the page cache the kernel gives back.)
const FEATS_MARGIN: usize = 5 << 30;

fn mem_available() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: usize = s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb << 10)
}

/// One lane's images.
#[derive(Default)]
pub struct LaneImages {
    /// the images the lane's sequence holds: (first position, rows, key)
    pub seq: Vec<(usize, usize, u128)>,
    /// the coming prefill's: (first position, rows, first feature row in `feats`)
    pub spans: Vec<(usize, usize, usize)>,
    /// [rows][4096] bf16 for `spans`: allocated for the request, freed after its prefill
    pub feats: Option<DeviceBuffer<u16>>,
}

pub struct VisionState {
    pub dir: PathBuf,
    /// an image's merged-token cap (GLM_VISION_MAX_TOKENS, the processor's max_image_tokens)
    pub max_tokens: usize,
    pub lanes: Vec<LaneImages>,
    pub images: usize,
    pub secs: f64,
}

/// An image's rows in the prompt (one a 2 x 2 patch window).
pub fn image_rows(im: &ImageIn) -> usize {
    im.vit_h * im.vit_w / 4
}

/// An image's key: its grid and a 128-bit hash of its patches (what the tower reads), two
/// xxh64-round accumulators over the patches' 64-bit words with xxh64's final avalanche.
pub fn image_key(im: &ImageIn) -> u128 {
    const P1: u64 = 0x9E3779B185EBCA87;
    const P2: u64 = 0xC2B2AE3D27D4EB4F;
    const P3: u64 = 0x165667B19E3779F9;
    let round = |acc: u64, w: u64| acc.wrapping_add(w.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1);
    let avalanche = |mut h: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(P2);
        h ^= h >> 29;
        h = h.wrapping_mul(P3);
        h ^ (h >> 32)
    };
    let (mut a, mut b) = (P1.wrapping_add(im.vit_h as u64), P3 ^ ((im.vit_w as u64) << 20));
    for c in im.patches.chunks(4) {
        let w = c.iter().enumerate().fold(0u64, |w, (i, &x)| w | (x as u64) << (16 * i));
        a = round(a, w);
        b = round(b, w ^ P3);
    }
    let n = im.patches.len() as u64;
    ((avalanche(a ^ n) as u128) << 64) | avalanche(b.wrapping_add(n)) as u128
}

/// image_floor over a sequence's images `seq`: the first position where the two lists differ.
pub fn shared_floor(seq: &[(usize, usize, u128)], images: &[ImageIn], keys: &[u128]) -> usize {
    for i in 0..seq.len().max(images.len()) {
        let a = seq.get(i).copied();
        let b = images.get(i).map(|im| (im.start, image_rows(im), keys[i]));
        if a != b {
            return a.map_or(usize::MAX, |a| a.0).min(b.map_or(usize::MAX, |b| b.0));
        }
    }
    usize::MAX
}

impl Engine {
    /// Image input on (GLM_VISION=1): images of at most `max_tokens` rows; their features are
    /// allocated a request.
    pub fn enable_vision(&mut self, dir: &Path, max_tokens: usize) -> Result<()> {
        if !self.cfg.is_flash() || self.world != 1 {
            return Err("vision: GLM-5.3-Flash on one box only".into());
        }
        let lanes = (0..self.lanes()).map(|_| LaneImages::default()).collect();
        self.vis = Some(VisionState { dir: dir.to_path_buf(), max_tokens, lanes, images: 0, secs: 0.0 });
        Ok(())
    }

    /// How far a prompt with `images` (their keys `keys`) may reuse the lane's sequence: up to the
    /// first image the two do not share at the same position. Images share one placeholder id,
    /// so the token match alone cannot tell two of them apart.
    pub fn image_floor(&self, lane: usize, images: &[ImageIn], keys: &[u128]) -> usize {
        shared_floor(self.vis.as_ref().map_or(&[][..], |v| &v.lanes[lane].seq[..]), images, keys)
    }

    /// Encode the request's images the prefill from `keep` reads (those ending past it; the ones
    /// before are in the lane's sequence already) for the lane's coming prefill; none clears the
    /// lane's spans. The tower borrows expert slots and gives them back here. Ok: (images
    /// encoded, their rows, seconds).
    pub fn encode_images(&mut self, lane: usize, images: &[ImageIn], keys: &[u128], keep: usize) -> Result<(usize, usize, f64)> {
        let Some(v) = self.vis.as_mut() else {
            return if images.is_empty() { Ok((0, 0, 0.0)) } else { Err("image inputs on a server without GLM_VISION".into()) };
        };
        let li = &mut v.lanes[lane];
        li.spans.clear();
        li.feats = None;
        // the sequence keeps the images wholly before `keep` (image_floor matched them)
        li.seq.retain(|&(s, n, _)| s + n <= keep);
        let todo: Vec<usize> = (0..images.len()).filter(|&i| images[i].start + image_rows(&images[i]) > keep).collect();
        if todo.is_empty() {
            return Ok((0, 0, 0.0));
        }
        let t0 = Instant::now();
        if let Some(im) = todo.iter().map(|&i| &images[i]).find(|im| image_rows(im) > v.max_tokens) {
            return Err(format!("an image of {} tokens: GLM_VISION_MAX_TOKENS is {}", image_rows(im), v.max_tokens).into());
        }
        let total: usize = todo.iter().map(|&i| image_rows(&images[i])).sum();
        let need = total * ROW * 2;
        if let Some(free) = mem_available().filter(|&f| f < need + FEATS_MARGIN) {
            return Err(format!("{total} image tokens need {} MB of features: {} MB available, {} MB kept free", need >> 20,
                               free >> 20, FEATS_MARGIN >> 20).into());
        }
        let feats = DeviceBuffer::<u16>::new(total * ROW)?;
        let max_patches = todo.iter().map(|&i| images[i].vit_h * images[i].vit_w).max().unwrap();
        let dir = v.dir.clone();
        let lent = self.lend_slots(bytes_for_regions(max_patches))?;
        let enc = (|| -> Result<Vec<(usize, usize, usize)>> {
            let mut mem = Mem::new(&lent.ranges);
            let ck = tidelm_model::Checkpoint::open(&dir)?;
            let tower = Tower::load(&ck, &mut mem, &self.st, max_patches)?;
            let (mut spans, mut row) = (vec![], 0usize);
            for &i in &todo {
                let im = &images[i];
                let out = unsafe { (feats.raw() as *mut u16).add(row * ROW) } as *mut std::os::raw::c_void;
                let n = tower.forward(&self.st, &im.patches, &[(im.vit_h, im.vit_w)], out)?;
                if n != image_rows(im) {
                    return Err(format!("the tower gave {n} rows for an image of {}", image_rows(im)).into());
                }
                spans.push((im.start, n, row));
                row += n;
            }
            Ok(spans)
        })();
        // the slots go back whether or not the encode went through
        let back = self.return_slots(lent);
        let spans = enc?;
        back?;
        let secs = t0.elapsed().as_secs_f64();
        let v = self.vis.as_mut().unwrap();
        let li = &mut v.lanes[lane];
        li.seq.extend(todo.iter().map(|&i| (images[i].start, image_rows(&images[i]), keys[i])));
        li.spans = spans;
        li.feats = Some(feats);
        v.images += todo.len();
        v.secs += secs;
        Ok((todo.len(), total, secs))
    }

    /// Whether the lane's coming prefill has image rows.
    pub fn has_image_spans(&self, lane: usize) -> bool {
        self.vis.as_ref().is_some_and(|v| !v.lanes[lane].spans.is_empty())
    }

    /// The lane's prompt has prefilled (or its request ended): its spans and features are done with.
    pub fn clear_image_spans(&mut self, lane: usize) {
        if let Some(v) = self.vis.as_mut() {
            v.lanes[lane].spans.clear();
            v.lanes[lane].feats = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn im(start: usize, h: usize, w: usize, fill: u16) -> ImageIn {
        ImageIn { start, vit_h: h, vit_w: w, patches: vec![fill; h * w * 1176] }
    }

    #[test]
    fn keys_tell_images_apart() {
        let (a, b) = (im(10, 4, 6, 7), im(10, 4, 6, 7));
        assert_eq!(image_key(&a), image_key(&b));
        let mut c = im(10, 4, 6, 7);
        c.patches[12345] ^= 1;
        assert_ne!(image_key(&a), image_key(&c));
        // the same patches on another grid
        assert_ne!(image_key(&a), image_key(&ImageIn { start: 10, vit_h: 6, vit_w: 4, patches: a.patches.clone() }));
    }

    #[test]
    fn the_floor_is_the_first_image_not_shared() {
        let (x, y) = (im(100, 4, 4, 1), im(300, 4, 4, 2));
        let (kx, ky) = (image_key(&x), image_key(&y));
        let seq = vec![(100, 4, kx)];
        // no images either side: no limit
        assert_eq!(shared_floor(&[], &[], &[]), usize::MAX);
        // the follow-up turn: the lane's image again, then a new one
        assert_eq!(shared_floor(&seq, &[x.clone(), y.clone()], &[kx, ky]), 300);
        // the same image again only: no limit
        assert_eq!(shared_floor(&seq, &[x.clone()], &[kx]), usize::MAX);
        // another image where the lane's was
        let z = im(100, 4, 4, 3);
        assert_eq!(shared_floor(&seq, &[z.clone()], &[image_key(&z)]), 100);
        // the prompt without the lane's image
        assert_eq!(shared_floor(&seq, &[], &[]), 100);
        // a fresh lane, a prompt with images
        assert_eq!(shared_floor(&[], &[y], &[ky]), 300);
    }
}
