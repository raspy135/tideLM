//! Image inputs for the vision tower: the checkpoint's inference/image_processor.py
//! (load_image, plan_image_grid) in Rust, pixel for pixel. An image's bytes are decoded and
//! converted as Pillow's `Image.open(...).convert("RGB")` does (no EXIF rotation, alpha
//! dropped, palettes mapped), planned onto a patch grid of at most `max_tokens` LLM tokens,
//! fitted into it with `ImageOps.pad` on a gray-127 canvas (Pillow's integer bicubic,
//! reproduced here), normalized to bf16 and cut into [3 x 14 x 14] patches.
//!
//! Decoders: PNG (png), JPEG (libjpeg-turbo, the library Pillow decodes with), GIF (its first
//! frame), WebP (image-webp). Anything else is refused.

use sha2::{Digest, Sha256};

/// The checkpoint's vision_* settings the preprocessing reads.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub patch: usize,
    pub ratio: usize,
    pub max_tokens: usize,
    pub min_pixels: usize,
}

/// An image ready for the tower: its patch grid, the LLM token grid after the aligner, and
/// the patches (bf16 bits, [vit_h x vit_w][3][patch][patch]).
#[derive(Clone, Debug)]
pub struct Prepared {
    pub vit_h: usize,
    pub vit_w: usize,
    pub llm_h: usize,
    pub llm_w: usize,
    pub patches: Vec<u16>,
}

impl Prepared {
    /// image_processor.num_image_tokens: the span's length in the prompt
    pub fn tokens(&self) -> usize {
        self.llm_h * (self.llm_w + 1) + 2
    }

    /// The image's identity for the prefix cache: a digest of the grid and the patches
    /// (prefix_media.image_fingerprints hashes the same preprocessed content).
    pub fn digest(&self) -> u64 {
        let mut h = Sha256::new();
        h.update(b"prefix-image-v1");
        h.update((self.vit_h as u64).to_le_bytes());
        h.update((self.vit_w as u64).to_le_bytes());
        for p in &self.patches {
            h.update(p.to_le_bytes());
        }
        u64::from_le_bytes(h.finalize()[..8].try_into().unwrap())
    }
}

/// An RGB image, 3 bytes a pixel, row-major.
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

/// Bytes to RGB as `Image.open(BytesIO(b)).convert("RGB")`.
pub fn decode_rgb(b: &[u8]) -> Result<Rgb, String> {
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(b)
    } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
        decode_jpeg(b)
    } else if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        decode_gif(b)
    } else if b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        decode_webp(b)
    } else {
        Err("unsupported image format (PNG, JPEG, GIF and WebP are)".into())
    }
}

fn decode_png(b: &[u8]) -> Result<Rgb, String> {
    use png::{BitDepth, ColorType, Transformations};
    let mut dec = png::Decoder::new(std::io::Cursor::new(b));
    // palettes, low bit depths and tRNS expanded; 16 bits kept (Pillow reads them itself)
    dec.set_transformations(Transformations::EXPAND);
    let mut rd = dec.read_info().map_err(|e| format!("PNG: {e}"))?;
    let mut buf = vec![0u8; rd.output_buffer_size().ok_or("PNG: image too large")?];
    let info = rd.next_frame(&mut buf).map_err(|e| format!("PNG: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let (ct, bd) = rd.output_color_type();
    let ch = match ct {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => return Err("PNG: palette not expanded".into()),
    };
    let wide = bd == BitDepth::Sixteen;
    let stride = info.line_size;
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &buf[y * stride..];
        for x in 0..w {
            // a 16-bit sample: Pillow's ;16B raw modes keep its high byte, except grayscale
            // alone, which opens as I;16 and converts by saturating (1000 -> 255)
            let s = |c: usize| -> u8 {
                if wide {
                    let v = u16::from_be_bytes([row[(x * ch + c) * 2], row[(x * ch + c) * 2 + 1]]);
                    if ch == 1 { v.min(255) as u8 } else { (v >> 8) as u8 }
                } else {
                    row[x * ch + c]
                }
            };
            match ch {
                1 | 2 => px.extend_from_slice(&[s(0); 3]),
                _ => px.extend_from_slice(&[s(0), s(1), s(2)]),
            }
        }
    }
    Ok(Rgb { w, h, px })
}

fn decode_jpeg(b: &[u8]) -> Result<Rgb, String> {
    let img = turbojpeg::decompress(b, turbojpeg::PixelFormat::RGB).map_err(|e| format!("JPEG: {e}"))?;
    let (w, h) = (img.width, img.height);
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        px.extend_from_slice(&img.pixels[y * img.pitch..y * img.pitch + w * 3]);
    }
    Ok(Rgb { w, h, px })
}

fn decode_gif(b: &[u8]) -> Result<Rgb, String> {
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::Indexed);
    let mut dec = opts.read_info(std::io::Cursor::new(b)).map_err(|e| format!("GIF: {e}"))?;
    let (w, h) = (dec.width() as usize, dec.height() as usize);
    let global = dec.global_palette().map(|p| p.to_vec());
    let bg = dec.bg_color().unwrap_or(0);
    let frame = dec.read_next_frame().map_err(|e| format!("GIF: {e}"))?.ok_or("GIF: no frames")?;
    let pal = frame.palette.clone().or(global).ok_or("GIF: no palette")?;
    let color = |i: usize| -> [u8; 3] {
        if 3 * i + 2 < pal.len() { [pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]] } else { [0, 0, 0] }
    };
    // the first frame on the logical screen, the rest of it the background color
    let mut px: Vec<u8> = std::iter::repeat_n(color(bg), w * h).flatten().collect();
    let (fl, ft, fw, fh) = (frame.left as usize, frame.top as usize, frame.width as usize, frame.height as usize);
    for y in 0..fh.min(h.saturating_sub(ft)) {
        for x in 0..fw.min(w.saturating_sub(fl)) {
            let c = color(frame.buffer[y * fw + x] as usize);
            let o = ((ft + y) * w + fl + x) * 3;
            px[o..o + 3].copy_from_slice(&c);
        }
    }
    Ok(Rgb { w, h, px })
}

fn decode_webp(b: &[u8]) -> Result<Rgb, String> {
    let mut dec = image_webp::WebPDecoder::new(std::io::Cursor::new(b)).map_err(|e| format!("WebP: {e}"))?;
    let (w, h) = dec.dimensions();
    let (w, h) = (w as usize, h as usize);
    let ch = if dec.has_alpha() { 4 } else { 3 };
    let mut buf = vec![0u8; dec.output_buffer_size().ok_or("WebP: image too large")?];
    dec.read_image(&mut buf).map_err(|e| format!("WebP: {e}"))?;
    let px = if ch == 3 { buf } else { buf.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect() };
    Ok(Rgb { w, h, px })
}

// ------------------------------------------------------------------ the plan

fn tokens(nh: usize, nw: usize) -> usize {
    nh * (nw + 1) + 2
}

/// llm_grid: the aligner's token grid of a pixel size
fn llm_grid(bh: usize, bw: usize, p: &Plan) -> (usize, usize) {
    ((bh / p.patch).div_ceil(p.ratio), (bw / p.patch).div_ceil(p.ratio))
}

/// solve_resize_ratio: the largest aspect-keeping pixel size within max_tokens
fn solve_resize_ratio(h: usize, w: usize, p: &Plan) -> (usize, usize) {
    let r = h as f64 / w as f64;
    let mw = ((p.max_tokens - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let mh = mw * r;
    let cell = p.patch * p.ratio;
    if mw < 1.0 {
        return ((p.max_tokens - 2) / 2 * cell, cell);
    }
    if mh < 1.0 {
        return (cell, (p.max_tokens - 3) * cell);
    }
    let beta = (mw.floor() * cell as f64 / w as f64).min(mh.floor() * cell as f64 / h as f64);
    let fl = |x: f64| (x / p.patch as f64).floor() as usize * p.patch;
    (fl(h as f64 * beta), fl(w as f64 * beta))
}

/// plan_image_grid (vision_max_wh_ratio null): (llm_h, llm_w, best_h, best_w) for an
/// image of `w0` x `h0` pixels
pub fn plan(w0: usize, h0: usize, p: &Plan) -> (usize, usize, usize, usize) {
    let (mut w, mut h) = (w0, h0);
    if w * h > 0 && w * h < p.min_pixels {
        let s = (p.min_pixels as f64 / (w * h) as f64).powf(0.5);
        w = (w as f64 * s) as usize;
        h = (h as f64 * s) as usize;
    }
    let (mut bw, mut bh) = (w.div_ceil(p.patch) * p.patch, h.div_ceil(p.patch) * p.patch);
    let (mut lh, mut lw) = llm_grid(bh, bw, p);
    if tokens(lh, lw) > p.max_tokens {
        (bh, bw) = solve_resize_ratio(h, w, p);
        (lh, lw) = llm_grid(bh, bw, p);
    }
    (lh, lw, bh, bw)
}

// ------------------------------------------------------------------ Pillow's bicubic

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// precompute_coeffs + normalize_coeffs_8bpc (Resample.c) for in -> out: per output its
/// first input and the fixed-point weights of its inputs
fn coeffs(inp: usize, out: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = inp as f64 / out as f64;
    let fscale = scale.max(1.0);
    let support = 2.0 * fscale;
    let ss = 1.0 / fscale;
    (0..out)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(inp as i64) as usize - xmin;
            let mut k: Vec<f64> = (0..xmax).map(|x| bicubic(((x + xmin) as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            let one = (1u32 << PRECISION_BITS) as f64;
            let ki = k.iter().map(|&v| if v < 0.0 { (-0.5 + v * one) as i32 } else { (0.5 + v * one) as i32 }).collect();
            (xmin, ki)
        })
        .collect()
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Image.resize((ow, oh), BICUBIC) of an RGB image: the horizontal pass (over the rows the
/// vertical one reads) into u8, then the vertical; a pass whose side does not change is
/// skipped, and an unchanged size is a copy.
pub fn resize_bicubic(img: &Rgb, ow: usize, oh: usize) -> Rgb {
    if (ow, oh) == (img.w, img.h) {
        return Rgb { w: ow, h: oh, px: img.px.clone() };
    }
    let kv = coeffs(img.h, oh);
    let (y0, y1) = (kv[0].0, kv[oh - 1].0 + kv[oh - 1].1.len());
    let tmp;
    let (src, first_row) = if ow != img.w {
        let kh = coeffs(img.w, ow);
        let rows = y1 - y0;
        let mut px = vec![0u8; ow * rows * 3];
        for y in 0..rows {
            let row = &img.px[(y0 + y) * img.w * 3..(y0 + y + 1) * img.w * 3];
            for (xx, (xmin, k)) in kh.iter().enumerate() {
                for c in 0..3 {
                    let mut s: i32 = 1 << (PRECISION_BITS - 1);
                    for (x, &kx) in k.iter().enumerate() {
                        s = s.wrapping_add(row[(x + xmin) * 3 + c] as i32 * kx);
                    }
                    px[(y * ow + xx) * 3 + c] = clip8(s);
                }
            }
        }
        tmp = Rgb { w: ow, h: rows, px };
        (&tmp, y0)
    } else {
        (img, 0)
    };
    if oh == img.h {
        return Rgb { w: src.w, h: src.h, px: src.px.clone() };
    }
    let mut px = vec![0u8; src.w * oh * 3];
    for (yy, (ymin, k)) in kv.iter().enumerate() {
        let ymin = ymin - first_row;
        for xx in 0..src.w {
            for c in 0..3 {
                let mut s: i32 = 1 << (PRECISION_BITS - 1);
                for (y, &ky) in k.iter().enumerate() {
                    s = s.wrapping_add(src.px[((ymin + y) * src.w + xx) * 3 + c] as i32 * ky);
                }
                px[(yy * src.w + xx) * 3 + c] = clip8(s);
            }
        }
    }
    Rgb { w: src.w, h: oh, px }
}

/// round(): ties to even, as Python's
fn round_even(x: f64) -> usize {
    x.round_ties_even() as usize
}

/// ImageOps.pad(img, (w, h), color=(127, 127, 127)): `contain` (BICUBIC, the original's
/// aspect), centered on the canvas.
pub fn pad(img: &Rgb, w: usize, h: usize) -> Rgb {
    let (mut tw, mut th) = (w, h);
    let (ir, dr) = (img.w as f64 / img.h as f64, w as f64 / h as f64);
    if ir != dr {
        if ir > dr {
            let nh = round_even(img.h as f64 / img.w as f64 * w as f64);
            if nh != h {
                th = nh;
            }
        } else {
            let nw = round_even(img.w as f64 / img.h as f64 * h as f64);
            if nw != w {
                tw = nw;
            }
        }
    }
    let r = resize_bicubic(img, tw, th);
    if (r.w, r.h) == (w, h) {
        return r;
    }
    let mut out = Rgb { w, h, px: vec![127u8; w * h * 3] };
    let (ox, oy) = if r.w != w { (round_even((w - r.w) as f64 * 0.5), 0) } else { (0, round_even((h - r.h) as f64 * 0.5)) };
    for y in 0..r.h.min(h.saturating_sub(oy)) {
        let n = r.w.min(w - ox) * 3;
        out.px[((oy + y) * w + ox) * 3..((oy + y) * w + ox) * 3 + n].copy_from_slice(&r.px[y * r.w * 3..y * r.w * 3 + n]);
    }
    out
}

/// ((u8 / 255 - 0.5) / 0.5) in f32, as bf16 (round to nearest even): image_processor's
/// normalization, one value per byte
fn norm_lut() -> [u16; 256] {
    let mut t = [0u16; 256];
    for (v, o) in t.iter_mut().enumerate() {
        let f = ((v as f32 / 255.0f32) - 0.5f32) / 0.5f32;
        let b = f.to_bits();
        *o = ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16;
    }
    t
}

/// load_image: bytes to the tower's input.
pub fn prepare(bytes: &[u8], p: &Plan) -> Result<Prepared, String> {
    let img = decode_rgb(bytes)?;
    if img.w == 0 || img.h == 0 {
        return Err("an empty image".into());
    }
    let (llm_h, llm_w, bh, bw) = plan(img.w, img.h, p);
    let img = pad(&img, bw, bh);
    let (vit_h, vit_w) = (bh / p.patch, bw / p.patch);
    let lut = norm_lut();
    let ps = p.patch;
    let mut patches = Vec::with_capacity(vit_h * vit_w * 3 * ps * ps);
    for ph in 0..vit_h {
        for pw in 0..vit_w {
            for c in 0..3 {
                for py in 0..ps {
                    for px in 0..ps {
                        let (y, x) = (ph * ps + py, pw * ps + px);
                        patches.push(lut[img.px[(y * bw + x) * 3 + c] as usize]);
                    }
                }
            }
        }
    }
    Ok(Prepared { vit_h, vit_w, llm_h, llm_w, patches })
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Plan = Plan { patch: 14, ratio: 3, max_tokens: 1024, min_pixels: 295_936 };

    #[test]
    fn plans_match_image_processor() {
        // (W, H) -> (best W, best H, llm h, llm w): the grids the spec checked in tidelm:local
        for (w, h, bw, bh, lh, lw) in [(64, 64, 546, 546, 13, 13), (1, 1, 546, 546, 13, 13), (640, 480, 644, 490, 12, 16),
                                       (1280, 960, 1288, 966, 23, 31), (1920, 1080, 1708, 966, 23, 41), (1554, 1078, 1554, 1078, 26, 37),
                                       (2000, 2000, 1302, 1302, 31, 31), (4032, 3024, 1512, 1134, 27, 36), (50, 20000, 42, 16800, 400, 1)] {
            assert_eq!(plan(w, h, &P), (lh, lw, bh, bw), "{w} x {h}");
        }
    }

    #[test]
    fn the_normalization_is_256_distinct_bf16_values() {
        let t = norm_lut();
        assert_eq!(t[0], 0xbf80); // -1.0
        assert_eq!(t[255], 0x3f80); // 1.0
        assert_eq!(f32::from_bits((t[127] as u32) << 16), -0.003936767578125);
        let mut s = t.to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 256);
    }
}
