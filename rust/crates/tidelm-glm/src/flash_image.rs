//! A GLM-5.3-Flash request's images on the engine: encoded before its prompt prefills, with the
//! vision tower in lent expert slots (lend.rs), into a feature buffer the server reserved at boot;
//! the prompt's image rows then take those features in place of the `<|image|>` embedding
//! (flash_fwd embed_flash), and the slots go back to experts before the prefill.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tidelm_cuda::DeviceBuffer;
use tidelm_engine::vision::ImageIn;

use crate::flash_vision::{bytes_for_regions, Mem, Tower};
use crate::forward::Engine;
use crate::Result;

pub const ROW: usize = 4096;

pub struct VisionState {
    pub dir: PathBuf,
    /// image tokens a request may carry (GLM_VISION_MAX_TOKENS a request)
    pub max_tokens: usize,
    /// [max_tokens][4096] bf16
    pub feats: DeviceBuffer<u16>,
    /// the lane the spans are for and its spans: (first position, rows, first feature row)
    pub lane: usize,
    pub spans: Vec<(usize, usize, usize)>,
    /// per lane: its sequence's first image position (usize::MAX: none), the prefix's limit
    pub floor: Vec<usize>,
    pub images: usize,
    pub secs: f64,
}

impl Engine {
    /// Image input on (GLM_VISION=1): the feature buffer for `max_tokens` image rows a request.
    pub fn enable_vision(&mut self, dir: &Path, max_tokens: usize) -> Result<()> {
        if !self.cfg.is_flash() || self.world != 1 {
            return Err("vision: GLM-5.3-Flash on one box only".into());
        }
        let lanes = self.lanes();
        self.vis = Some(VisionState { dir: dir.to_path_buf(), max_tokens, feats: DeviceBuffer::zeroed(max_tokens * ROW)?, lane: 0,
                                      spans: vec![], floor: vec![usize::MAX; lanes], images: 0, secs: 0.0 });
        Ok(())
    }

    /// The lane's prefix limit: its sequence's first image position.
    pub fn image_floor(&self, lane: usize) -> usize {
        self.vis.as_ref().map(|v| v.floor[lane]).unwrap_or(usize::MAX)
    }

    /// Encode `images` (positions in the lane's sequence) for the lane's coming prefill; none
    /// clears the lane's spans. The tower borrows expert slots and gives them back here.
    pub fn encode_images(&mut self, lane: usize, images: &[ImageIn]) -> Result<f64> {
        let Some(v) = self.vis.as_mut() else {
            return if images.is_empty() { Ok(0.0) } else { Err("image inputs on a server without GLM_VISION".into()) };
        };
        v.spans.clear();
        if images.is_empty() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let total: usize = images.iter().map(|i| i.vit_h * i.vit_w / 4).sum();
        if total > v.max_tokens {
            return Err(format!("{total} image tokens in a request: GLM_VISION_MAX_TOKENS is {}", v.max_tokens).into());
        }
        let max_patches = images.iter().map(|i| i.vit_h * i.vit_w).max().unwrap();
        let dir = v.dir.clone();
        let lent = self.lend_slots(bytes_for_regions(max_patches))?;
        let enc = (|| -> Result<Vec<(usize, usize, usize)>> {
            let mut mem = Mem::new(&lent.ranges);
            let ck = tidelm_model::Checkpoint::open(&dir)?;
            let tower = Tower::load(&ck, &mut mem, &self.st, max_patches)?;
            let feats = self.vis.as_ref().unwrap().feats.raw();
            let (mut spans, mut row) = (vec![], 0usize);
            for im in images {
                let out = unsafe { (feats as *mut u16).add(row * ROW) } as *mut std::os::raw::c_void;
                let n = tower.forward(&self.st, &im.patches, &[(im.vit_h, im.vit_w)], out)?;
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
        v.lane = lane;
        v.floor[lane] = v.floor[lane].min(spans[0].0);
        v.spans = spans;
        v.images += images.len();
        v.secs += secs;
        Ok(secs)
    }

    /// The lane's prompt has prefilled: its spans are done with.
    pub fn clear_image_spans(&mut self, lane: usize) {
        if let Some(v) = self.vis.as_mut().filter(|v| v.lane == lane) {
            v.spans.clear();
        }
    }
}
