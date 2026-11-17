//! Inkling 2's image input (HF model_type `inkling2`) as transformers' Inkling2ImageProcessor
//! and the tower's embedding layer make it, on the CPU: an image is used at its own resolution
//! (the reference never resizes), one language-model token a 40 x 40 pixel cell.
//!
//! - Decode as Pillow's `convert("RGBA" if has_transparency_data else "RGB")`: the crate's
//!   decoders for files without transparency; with it (PNG alpha channels and tRNS, WebP alpha,
//!   a GIF frame with a transparent index) the alpha is kept.
//! - Alpha is composited onto the processor's checkerboard (204 / 153 in 8-pixel squares)
//!   before anything else.
//! - `rows = ceil(H / 40)`, `cols = ceil(W / 40)`; the image is padded at the bottom and right
//!   with -1 on the 0..255 scale and normalized (x - 127.5) / 127.5 in fp32 (the backend's fused
//!   rescale 1/255 and mean 0.5, std 0.5), then rounded to bf16 (the model's cast).
//! - The processor's `pixel_values` are [rows * cols][40][40][3]; Inkling2VisionEmbeddings
//!   reshapes them `(-1, 2, 20, 2, 20, 3).transpose(2, 3)` into [tokens * 4][1200]: patches
//!   token-major (tokens row-major), a token's four 20 x 20 patches in (row, column) order, a
//!   patch's values (y, x, channel). `Prepared::patches` is that final layout, what the
//!   patch-embedding GEMM reads.
//!
//! Ours, not the reference's: `max_tokens`. An image whose grid would hold more tokens is
//! first scaled down (aspect kept, `glm5next::resize`: torchvision's antialiased uint8 bicubic,
//! the resize transformers' torchvision backend would do) to the largest size whose grid fits.
//! The scaling comes after the checkerboard, so a capped image's squares shrink with it.
//!
//! Where Pillow's own PNG reading departs from the PNG specification the port follows the
//! specification (Pillow 12.3.0 compares a tRNS colour key in the wrong scale for 2- and 4-bit
//! and for 16-bit samples, so a keyed pixel stays opaque there or another one turns
//! transparent); `tiny_files_decode_as_pillow` has the cases.

use super::{glm5next, Rgb};
use sha2::{Digest, Sha256};

/// A language-model image token's cell, in pixels.
pub const CELL: usize = 40;
/// The tower's patch, in pixels.
pub const PATCH: usize = 20;
/// Patches a cell side (spatial_merge_size).
pub const MERGE: usize = CELL / PATCH;
/// A patch's values: 20 x 20 pixels of 3 channels.
pub const PATCH_K: usize = 3 * PATCH * PATCH;

/// The pad on the 0..255 scale, before the normalization.
const PAD: f32 = -1.0;

/// An image ready for the tower.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// the token grid: ceil(H / 40) rows of ceil(W / 40) tokens
    pub rows: usize,
    pub cols: usize,
    /// bf16 [rows * cols * 4][1200]
    pub patches: Vec<u16>,
    /// sha256 of the grid and the patches (cache keys)
    pub digest: [u8; 32],
}

impl Prepared {
    /// the image's placeholder tokens in the prompt: one a cell
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }

    /// the tower's patch rows (glm5next::Prepared's `gh`)
    pub fn gh(&self) -> usize {
        self.rows * MERGE
    }

    /// the tower's patch columns (glm5next::Prepared's `gw`)
    pub fn gw(&self) -> usize {
        self.cols * MERGE
    }
}

/// Decoded pixels, row-major: 3 bytes a pixel (RGB), or 4 (RGBA) for a file with transparency.
#[derive(Clone, Debug)]
pub struct Pixels {
    pub w: usize,
    pub h: usize,
    pub channels: usize,
    pub px: Vec<u8>,
}

fn opaque(r: Rgb) -> Pixels {
    Pixels { w: r.w, h: r.h, channels: 3, px: r.px }
}

/// Bytes to pixels as `Image.open(BytesIO(b))` and `convert("RGBA" if has_transparency_data
/// else "RGB")`: PNG, JPEG, GIF (its first frame) and WebP.
pub fn decode(b: &[u8]) -> Result<Pixels, String> {
    if b.is_empty() {
        Err("an empty image (no bytes)".into())
    } else if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(b)
    } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
        super::decode_jpeg(b).map(opaque)
    } else if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        decode_gif(b)
    } else if b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        decode_webp(b)
    } else {
        Err("unsupported image format (PNG, JPEG, GIF and WebP are)".into())
    }
}

/// A PNG with an alpha channel or a tRNS chunk as RGBA; any other through the crate's decoder.
fn decode_png(b: &[u8]) -> Result<Pixels, String> {
    use png::{BitDepth, ColorType, Transformations};
    let mut dec = png::Decoder::new(std::io::Cursor::new(b));
    // palettes, low bit depths and tRNS (to an alpha channel) expanded; 16 bits kept
    dec.set_transformations(Transformations::EXPAND);
    let mut rd = dec.read_info().map_err(|e| format!("PNG: {e}"))?;
    let (ct, bd) = rd.output_color_type();
    let ch = match ct {
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgba => 4,
        _ => return super::decode_png(b).map(opaque),
    };
    let wide = bd == BitDepth::Sixteen;
    // 16-bit gray with a colour key is Pillow's I;16, whose gray saturates (1000 -> 255) as the
    // crate's decoder has it; gray with an alpha channel keeps the high byte like the rest
    let saturate = wide && rd.info().color_type == ColorType::Grayscale;
    let mut buf = vec![0u8; rd.output_buffer_size().ok_or("PNG: image too large")?];
    let info = rd.next_frame(&mut buf).map_err(|e| format!("PNG: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = info.line_size;
    let mut px = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        let row = &buf[y * stride..];
        for x in 0..w {
            // a 16-bit sample's high byte (Pillow's ;16B raw modes)
            let s = |c: usize| -> u8 { if wide { row[(x * ch + c) * 2] } else { row[x * ch + c] } };
            if ch == 2 {
                let g = if saturate { u16::from_be_bytes([row[x * 4], row[x * 4 + 1]]).min(255) as u8 } else { s(0) };
                px.extend_from_slice(&[g, g, g, s(1)]);
            } else {
                px.extend_from_slice(&[s(0), s(1), s(2), s(3)]);
            }
        }
    }
    Ok(Pixels { w, h, channels: 4, px })
}

/// A GIF whose first frame has a transparent index as RGBA: that index and the screen outside
/// the frame transparent (Pillow fills the screen with the index); any other through the
/// crate's decoder.
fn decode_gif(b: &[u8]) -> Result<Pixels, String> {
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::Indexed);
    let mut dec = opts.read_info(std::io::Cursor::new(b)).map_err(|e| format!("GIF: {e}"))?;
    let (w, h) = (dec.width() as usize, dec.height() as usize);
    let frame = dec.next_frame_info().map_err(|e| format!("GIF: {e}"))?.ok_or("GIF: no frames")?;
    let Some(clear) = frame.transparent else {
        return super::decode_gif(b).map(opaque);
    };
    let (fl, ft, fw, fh) = (frame.left as usize, frame.top as usize, frame.width as usize, frame.height as usize);
    let local = frame.palette.clone();
    let pal = local.or_else(|| dec.global_palette().map(|p| p.to_vec())).ok_or("GIF: no palette")?;
    let mut idx = vec![0u8; dec.buffer_size()];
    dec.read_into_buffer(&mut idx).map_err(|e| format!("GIF: {e}"))?;
    let mut px = vec![0u8; w * h * 4];
    for y in 0..fh.min(h.saturating_sub(ft)) {
        for x in 0..fw.min(w.saturating_sub(fl)) {
            let i = idx[y * fw + x];
            if i != clear {
                let i = i as usize;
                let c = if 3 * i + 2 < pal.len() { [pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]] } else { [0, 0, 0] };
                let o = ((ft + y) * w + fl + x) * 4;
                px[o..o + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
    }
    Ok(Pixels { w, h, channels: 4, px })
}

/// A WebP with alpha as RGBA; any other through the crate's decoder.
fn decode_webp(b: &[u8]) -> Result<Pixels, String> {
    let mut dec = image_webp::WebPDecoder::new(std::io::Cursor::new(b)).map_err(|e| format!("WebP: {e}"))?;
    if !dec.has_alpha() {
        return super::decode_webp(b).map(opaque);
    }
    let (w, h) = dec.dimensions();
    let mut px = vec![0u8; dec.output_buffer_size().ok_or("WebP: image too large")?];
    dec.read_image(&mut px).map_err(|e| format!("WebP: {e}"))?;
    Ok(Pixels { w: w as usize, h: h as usize, channels: 4, px })
}

/// Inkling2ImageProcessor.convert_to_rgb of an RGBA image: over a checkerboard of 8-pixel
/// squares, 204 where the square's row and column sum is even and 153 elsewhere, in the
/// processor's integer arithmetic `(rgb * a + background * (255 - a) + 127) // 255`.
pub fn composite(w: usize, h: usize, rgba: &[u8]) -> Rgb {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let p = &rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
            let a = p[3] as u32;
            let bg: u32 = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
            for &v in &p[..3] {
                px.push(((v as u32 * a + bg * (255 - a) + 127) / 255) as u8);
            }
        }
    }
    Rgb { w, h, px }
}

/// The token grid (rows, cols) of a `w` x `h` image.
pub fn grid(w: usize, h: usize) -> (usize, usize) {
    (h.div_ceil(CELL), w.div_ceil(CELL))
}

/// The size (w, h) an image is prepared at under a cap of `max_tokens` (0: none): its own when
/// its grid fits, else the largest scaled-down one whose grid does - the longer side the
/// largest that fits, the shorter the nearest to the image's aspect (at least 1). Never larger
/// than the image.
pub fn fit(w: usize, h: usize, max_tokens: usize) -> (usize, usize) {
    let fits = |(w, h): (usize, usize)| {
        let (r, c) = grid(w, h);
        r.saturating_mul(c) <= max_tokens
    };
    if max_tokens == 0 || w == 0 || h == 0 || fits((w, h)) {
        return (w, h);
    }
    let (long, short) = (w.max(h) as u128, w.min(h) as u128);
    let size = |l: usize| {
        let s = (((short * l as u128 + long / 2) / long) as usize).max(1);
        if w >= h { (l, s) } else { (s, l) }
    };
    // tokens grow with the longer side; 1 pixel of it is one token, which any cap holds
    let (mut lo, mut hi) = (1, w.max(h) - 1);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(size(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    size(lo)
}

/// fp32 to bf16, round to nearest even
fn bf16(f: f32) -> u16 {
    let b = f.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// rescale_and_normalize of a value on the 0..255 scale, as bf16: the torchvision backend's
/// fused form, mean and std 0.5 * (1 / (1/255)) = 127.5 in fp32, `(x - mean) / std`
fn norm(v: f32) -> u16 {
    bf16((v - 127.5f32) / 127.5f32)
}

fn lut() -> [u16; 256] {
    let mut t = [0u16; 256];
    for (v, o) in t.iter_mut().enumerate() {
        *o = norm(v as f32);
    }
    t
}

/// sha256 of the grid and the patches: what the tower reads, so equal digests are equal inputs
fn digest(rows: usize, cols: usize, patches: &[u16]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((rows as u32).to_le_bytes());
    h.update((cols as u32).to_le_bytes());
    let mut buf = [0u8; 2 * PATCH_K];
    for p in patches.chunks(PATCH_K) {
        for (b, v) in buf.chunks_exact_mut(2).zip(p) {
            b.copy_from_slice(&v.to_le_bytes());
        }
        h.update(&buf[..2 * p.len()]);
    }
    h.finalize().into()
}

/// An RGB image of its final size into the padded grid's patches.
fn cut(w: usize, h: usize, px: &[u8]) -> Prepared {
    let (rows, cols) = grid(w, h);
    let lut = lut();
    let mut patches = vec![norm(PAD); rows * cols * MERGE * MERGE * PATCH_K];
    for y in 0..h {
        let (r, ir, py) = (y / CELL, y % CELL / PATCH, y % PATCH);
        // a row's runs of 20 pixels: each one row of one patch
        for (run, src) in px[y * w * 3..(y + 1) * w * 3].chunks(PATCH * 3).enumerate() {
            let (c, ic) = (run / MERGE, run % MERGE);
            let o = (((r * cols + c) * MERGE + ir) * MERGE + ic) * PATCH_K + py * PATCH * 3;
            for (d, &v) in patches[o..o + src.len()].iter_mut().zip(src) {
                *d = lut[v as usize];
            }
        }
    }
    let digest = digest(rows, cols, &patches);
    Prepared { rows, cols, patches, digest }
}

/// Image bytes to the tower's input, of at most `max_tokens` tokens (0: no cap).
pub fn prepare(bytes: &[u8], max_tokens: usize) -> Result<Prepared, String> {
    let p = decode(bytes)?;
    if p.w == 0 || p.h == 0 {
        return Err("an empty image".into());
    }
    if p.channels == 4 {
        prepare_rgb(&composite(p.w, p.h, &p.px), max_tokens)
    } else {
        prepare_rgb(&Rgb { w: p.w, h: p.h, px: p.px }, max_tokens)
    }
}

/// Raw pixels (row-major, `channels` 3: RGB, 4: RGBA with straight alpha) to the tower's input.
pub fn prepare_pixels(w: usize, h: usize, channels: usize, data: &[u8], max_tokens: usize) -> Result<Prepared, String> {
    if w == 0 || h == 0 {
        return Err("an empty image".into());
    }
    if channels != 3 && channels != 4 {
        return Err(format!("{channels} channels (3: RGB and 4: RGBA are)"));
    }
    if w.checked_mul(h).and_then(|n| n.checked_mul(channels)) != Some(data.len()) {
        return Err(format!("{} bytes for {w} x {h} pixels of {channels} channels", data.len()));
    }
    if channels == 4 {
        prepare_rgb(&composite(w, h, data), max_tokens)
    } else {
        prepare_rgb(&Rgb { w, h, px: data.to_vec() }, max_tokens)
    }
}

/// An RGB image to the tower's input: scaled down when its grid exceeds `max_tokens`, padded
/// to the grid, normalized and cut.
pub fn prepare_rgb(img: &Rgb, max_tokens: usize) -> Result<Prepared, String> {
    if img.w == 0 || img.h == 0 {
        return Err("an empty image".into());
    }
    if img.w.checked_mul(img.h).and_then(|n| n.checked_mul(3)) != Some(img.px.len()) {
        return Err(format!("{} bytes for {} x {} RGB pixels", img.px.len(), img.w, img.h));
    }
    let (w, h) = fit(img.w, img.h, max_tokens);
    if (w, h) != (img.w, img.h) {
        let r = glm5next::resize(img, w, h);
        return Ok(cut(r.w, r.h, &r.px));
    }
    Ok(cut(img.w, img.h, &img.px))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD_BITS: u16 = 0xbf81;

    fn bf(bits: u16) -> f64 {
        f32::from_bits((bits as u32) << 16) as f64
    }

    fn sha(patches: &[u16]) -> String {
        let raw: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
        format!("{:x}", Sha256::digest(&raw))
    }

    fn hex(d: &[u8]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// The generator the numpy goldens were made with: the high byte of x = 1664525 x + 1013904223
    fn lcg(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        }).collect()
    }

    /// torch's permute of a row-major array
    fn permute<T: Copy>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
        let n = shape.len();
        assert_eq!(shape.iter().product::<usize>(), data.len());
        let mut stride = vec![1; n];
        for i in (0..n - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        let mut idx = vec![0usize; n];
        let mut out = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            out.push(data[(0..n).map(|k| idx[k] * stride[perm[k]]).sum::<usize>()]);
            for k in (0..n).rev() {
                idx[k] += 1;
                if idx[k] < shape[perm[k]] {
                    break;
                }
                idx[k] = 0;
            }
        }
        out
    }

    /// fp32 to bf16 by its definition: the nearer of the two neighbours, a tie to the even one
    fn nearest_bf16(f: f32) -> u16 {
        let lo = (f.to_bits() >> 16) as u16;
        let (dl, dh) = ((f as f64 - bf(lo)).abs(), (bf(lo + 1) - f as f64).abs());
        if dh < dl || (dh == dl && lo & 1 == 1) { lo + 1 } else { lo }
    }

    /// The reference, step by step: Inkling2ImageProcessor._preprocess on an RGB image, then
    /// Inkling2VisionEmbeddings.forward's reshape, with the model's cast to bf16.
    fn reference(w: usize, h: usize, rgb: &[u8]) -> (usize, usize, Vec<u16>) {
        let (rows, cols) = (h.div_ceil(40), w.div_ceil(40));
        let (ph, pw) = (rows * 40, cols * 40);
        // image.float() [C][H][W], pad(fill_value=-1.0) at the bottom and right
        let mut chw = vec![-1.0f32; 3 * ph * pw];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    chw[(c * ph + y) * pw + x] = rgb[(y * w + x) * 3 + c] as f32;
                }
            }
        }
        // rescale_and_normalize: (x - 127.5) / 127.5
        let x: Vec<u16> = chw.iter().map(|&v| nearest_bf16((v - 127.5) / 127.5)).collect();
        // image.reshape(channels, rows, 40, columns, 40).permute(1, 3, 2, 4, 0).flatten(0, 1)
        let pixel_values = permute(&x, &[3, rows, 40, cols, 40], &[1, 3, 2, 4, 0]);
        // pixel_values.reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3), then reshape(-1, 1200)
        (rows, cols, permute(&pixel_values, &[rows * cols, 2, 20, 2, 20, 3], &[0, 1, 3, 2, 4, 5]))
    }

    #[test]
    fn normalization_and_pad_by_hand() {
        let t = lut();
        // (v - 127.5) / 127.5 to 8 significant bits:
        //   0 -> -1; 255 -> 1; 127 -> -1.0039 x 2^-8, mantissa 0.502 x 128 -> 1; 128 its negative
        //   1 -> -1.98431 x 2^-1: 0.98431 x 128 = 125.99 -> 126; 64 -> -1.99216 x 2^-2: 126.996 -> 127
        //   191 = -(64); 200 -> 1.13725 x 2^-1: 17.57 -> 18; 204 -> 0.6 = 1.2 x 2^-1: 25.6 -> 26
        //   153 -> 0.2 = 1.6 x 2^-3: 76.8 -> 77
        for (v, want) in [(0, 0xbf80), (255, 0x3f80), (127, 0xbb81), (128, 0x3b81), (1, 0xbf7e), (64, 0xbeff),
                          (191, 0x3eff), (200, 0x3f12), (204, 0x3f1a), (153, 0x3e4d)] {
            assert_eq!(t[v], want, "{v}");
        }
        // the pad: (-1 - 127.5) / 127.5 = -1.00784 -> -(1 + 1/128), not black's -1
        assert_eq!(norm(PAD), PAD_BITS);
        assert_eq!(bf(PAD_BITS), -1.0078125);
        // each value the bf16 nearest to the exact quotient, and all 257 distinct
        for v in -1i32..256 {
            let (got, exact) = (norm(v as f32), (v as f64 - 127.5) / 127.5);
            let d = (bf(got) - exact).abs();
            assert!(d <= (bf(got + 1) - exact).abs() && d <= (bf(got - 1) - exact).abs(), "{v}");
            assert_eq!(got, nearest_bf16((v as f32 - 127.5) / 127.5), "{v}");
        }
        let mut s = t.to_vec();
        s.push(PAD_BITS);
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 257);
        // the crate's other form of the same normalization, ((v / 255) - 0.5) / 0.5, rounds alike
        assert_eq!(t, crate::norm_lut());
    }

    #[test]
    fn a_tiny_image_value_by_value() {
        // 3 x 2 pixels: one token, real values in the first patch's first two rows only
        let data = [0, 1, 64, 127, 128, 153, 191, 200, 204,
                    255, 0, 255, 1, 1, 1, 128, 127, 0];
        let p = prepare_pixels(3, 2, 3, &data, 0).unwrap();
        assert_eq!((p.rows, p.cols, p.tokens(), p.gh(), p.gw()), (1, 1, 1, 2, 2));
        let mut want = vec![PAD_BITS; 4 * 1200];
        want[..9].copy_from_slice(&[0xbf80, 0xbf7e, 0xbeff, 0xbb81, 0x3b81, 0x3e4d, 0x3eff, 0x3f12, 0x3f1a]);
        want[60..69].copy_from_slice(&[0x3f80, 0xbf80, 0x3f80, 0xbf7e, 0xbf7e, 0xbf7e, 0x3b81, 0xbb81, 0xbf80]);
        assert_eq!(p.patches, want);
    }

    #[test]
    fn pixels_at_the_pad_edge() {
        // 41 x 21: the last column alone in the second token, the last row alone in the lower patches
        let (w, h) = (41, 21);
        let data = lcg(w * h * 3, 7);
        let p = prepare_pixels(w, h, 3, &data, 0).unwrap();
        assert_eq!((p.rows, p.cols), (1, 2));
        let t = lut();
        let at = |y: usize, x: usize, c: usize| t[data[(y * w + x) * 3 + c] as usize];
        let v = |patch: usize, py: usize, px: usize, c: usize| p.patches[patch * 1200 + (py * 20 + px) * 3 + c];
        for c in 0..3 {
            assert_eq!(v(0, 0, 0, c), at(0, 0, c));
            assert_eq!(v(0, 19, 19, c), at(19, 19, c));
            assert_eq!(v(1, 0, 0, c), at(0, 20, c)); // the token's right patch
            assert_eq!(v(2, 0, 19, c), at(20, 19, c)); // the image's last row: the lower patches' first
            assert_eq!(v(3, 0, 0, c), at(20, 20, c));
            assert_eq!(v(2, 1, 0, c), PAD_BITS); // below it
            assert_eq!(v(4, 0, 0, c), at(0, 40, c)); // the last column: the second token's first
            assert_eq!(v(4, 0, 1, c), PAD_BITS); // right of it
            assert_eq!(v(6, 0, 0, c), at(20, 40, c)); // the corner pixel
            assert_eq!(v(6, 0, 1, c), PAD_BITS);
            assert_eq!(v(6, 1, 0, c), PAD_BITS);
            assert_eq!(v(5, 0, 0, c), PAD_BITS); // patches wholly in the pad
            assert_eq!(v(7, 19, 19, c), PAD_BITS);
        }
        assert_eq!(p.patches.iter().filter(|&&b| b != PAD_BITS).count(), w * h * 3);
        // the numpy transcription's values (golden.py, see numpy_goldens)
        for (i, want) in [(0, 0xbf06), (57, 0xbe05), (60, 0x3f7a), (2457, 0xbf58), (3600, 0xbf5a), (4800, 0xbedf),
                          (7200, 0x3ef3), (7203, 0xbf81)] {
            assert_eq!(p.patches[i], want, "{i}");
        }
        assert_eq!(sha(&p.patches), "381709b2728d84878ae12c1d38e7a40a62686ba6d067b501059ed7b0ffe71310");
    }

    fn coded(y: usize, x: usize, c: usize) -> u8 {
        ((y * 31 + x * 17 + c * 101) % 256) as u8
    }

    #[test]
    fn patch_order_index_by_index() {
        // 97 x 55: 2 rows of 3 tokens, 17 real columns in the last and 15 real rows
        let (w, h) = (97, 55);
        let data: Vec<u8> = (0..w * h * 3).map(|i| coded(i / 3 / w, i / 3 % w, i % 3)).collect();
        let p = prepare_pixels(w, h, 3, &data, 0).unwrap();
        assert_eq!((p.rows, p.cols, p.tokens(), p.patches.len()), (2, 3, 6, 6 * 4 * 1200));
        let t = lut();
        for k in 0..24 {
            // [tokens][2][2] patches: the token row-major in the 2 x 3 grid, then (in_row, in_col)
            let (token, in_row, in_col) = (k / 4, k % 4 / 2, k % 2);
            let (row, col) = (token / 3, token % 3);
            for j in 0..1200 {
                // [20][20][3] values: (y, x, channel)
                let (py, px, c) = (j / 60, j / 3 % 20, j % 3);
                let (y, x) = (row * 40 + in_row * 20 + py, col * 40 + in_col * 20 + px);
                let want = if y < h && x < w { t[coded(y, x, c) as usize] } else { PAD_BITS };
                assert_eq!(p.patches[k * 1200 + j], want, "patch {k} value {j}: pixel ({y}, {x}, {c})");
            }
        }
    }

    #[test]
    fn layout_is_the_references_reshapes() {
        for (w, h) in [(97, 55), (41, 21), (1, 1), (40, 40), (80, 41), (121, 39), (39, 121), (7, 200)] {
            let data = lcg(w * h * 3, (w * 1000 + h) as u32);
            let p = prepare_pixels(w, h, 3, &data, 0).unwrap();
            let (rows, cols, want) = reference(w, h, &data);
            assert_eq!((p.rows, p.cols), (rows, cols), "{w} x {h}");
            assert!(p.patches == want, "{w} x {h}");
        }
    }

    #[test]
    fn alpha_goes_onto_the_checkerboard() {
        let (w, h) = (20, 20);
        let mut px = vec![0u8; w * h * 4];
        let mut set = |y: usize, x: usize, v: [u8; 4]| px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&v);
        set(0, 0, [255, 0, 10, 128]); // on 204: (32640 + 25908 + 127) // 255, (25908 + 127) // 255, (1280 + 25908 + 127) // 255
        set(8, 7, [255, 255, 255, 1]); // on 153: (255 + 38862 + 127) // 255 = 153
        set(0, 9, [10, 20, 30, 255]); // opaque: itself
        set(9, 9, [0, 0, 0, 254]); // on 204: (204 + 127) // 255 = 1
        set(3, 17, [200, 100, 0, 51]); // on 204: (10200 + 41616 + 127) // 255 = 203, (5100 + 41743) // 255 = 183, 41743 // 255 = 163
        let c = composite(w, h, &px);
        assert_eq!((c.w, c.h, c.px.len()), (w, h, w * h * 3));
        let at = |y: usize, x: usize| [c.px[(y * w + x) * 3], c.px[(y * w + x) * 3 + 1], c.px[(y * w + x) * 3 + 2]];
        assert_eq!(at(0, 0), [230, 102, 107]);
        assert_eq!(at(8, 7), [153, 153, 153]);
        assert_eq!(at(0, 9), [10, 20, 30]);
        assert_eq!(at(9, 9), [1, 1, 1]);
        assert_eq!(at(3, 17), [203, 183, 163]);
        // fully transparent pixels are the board: 204 where ((y // 8) + (x // 8)) % 2 == 0, else 153
        for (y, x, bg) in [(1, 1, 204), (7, 7, 204), (0, 8, 153), (7, 15, 153), (8, 0, 153), (15, 7, 153), (8, 8, 204),
                           (15, 15, 204), (16, 16, 204), (16, 15, 153), (19, 0, 204), (0, 16, 204), (8, 16, 153)] {
            assert_eq!(at(y, x), [bg; 3], "({y}, {x})");
        }
        // prepare_pixels composites, then pads: the board is not the pad
        let p = prepare_pixels(w, h, 4, &px, 0).unwrap();
        let q = prepare_pixels(w, h, 3, &c.px, 0).unwrap();
        assert_eq!((p.patches == q.patches, p.digest == q.digest), (true, true));
        assert_eq!(&p.patches[..3], &[norm(230.0), norm(102.0), norm(107.0)]);
        assert_eq!(p.patches[60 * 19 + 57], lut()[204]);
        assert_eq!(p.patches[1200], PAD_BITS);
        // an opaque RGBA image is its RGB
        let rgb = lcg(33 * 9 * 3, 3);
        let rgba: Vec<u8> = rgb.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
        assert_eq!(composite(33, 9, &rgba).px, rgb);
    }

    #[test]
    fn the_cap_scales_down_to_the_largest_fit() {
        // 1000 x 600 is 15 x 25 = 375 tokens. Under 100: 480 x 288 is 8 x 12 = 96, and one more
        // pixel of width (481 x 289) is 8 x 13 = 104
        assert_eq!(fit(1000, 600, 100), (480, 288));
        assert_eq!(fit(600, 1000, 100), (288, 480));
        assert_eq!(fit(1000, 600, 375), (1000, 600));
        assert_eq!(fit(1000, 600, 0), (1000, 600));
        assert_eq!(fit(1000, 600, 1), (40, 24));
        // 50 x 20000: 640 high is 16 rows, 1.6 pixels wide rounds to 2
        assert_eq!(fit(50, 20000, 16), (2, 640));
        assert_eq!(fit(100_000, 1, 5), (200, 1));
        let tokens = |w: usize, h: usize| grid(w, h).0 * grid(w, h).1;
        for (w, h) in [(1000, 600), (600, 1000), (1920, 1080), (4032, 3024), (333, 517), (41, 40), (40, 41), (50, 20000),
                       (20000, 50), (4000, 1), (81, 81), (640, 480)] {
            for cap in [1, 2, 3, 5, 16, 64, 100, 256, 1000, 1024, 4096, 1 << 20] {
                let (tw, th) = fit(w, h, cap);
                assert!(tokens(tw, th) <= cap, "{w} x {h} under {cap}: {tw} x {th}");
                assert!(tw >= 1 && th >= 1 && tw <= w && th <= h, "{w} x {h} under {cap}: {tw} x {th}");
                if tokens(w, h) <= cap {
                    assert_eq!((tw, th), (w, h), "{w} x {h} fits {cap}");
                    continue;
                }
                assert!(tokens(tw, th) <= tokens(w, h));
                // the aspect: the shorter side within half a pixel (or the 1-pixel floor)
                let (l, s, long, short) = if w >= h { (tw, th, w, h) } else { (th, tw, h, w) };
                let exact = short as f64 * l as f64 / long as f64;
                assert!((s as f64 - exact).abs() <= 0.5 || (s == 1 && exact < 1.0), "{w} x {h} under {cap}: {tw} x {th}");
                // the largest: one more pixel of the longer side does not fit
                let s1 = ((short * (l + 1) + long / 2) / long).max(1);
                let (r, c) = if w >= h { grid(l + 1, s1) } else { grid(s1, l + 1) };
                assert!(r * c > cap, "{w} x {h} under {cap}: {tw} x {th} is not the largest");
            }
        }
    }

    #[test]
    fn a_capped_image_is_composited_then_resized() {
        // 200 x 120 is 3 x 5 = 15 tokens; under 6: 120 x 72 (2 x 3), 121 x 73 would be 2 x 4
        let (w, h) = (200, 120);
        let rgb = lcg(w * h * 3, 11);
        let free = prepare_pixels(w, h, 3, &rgb, 0).unwrap();
        assert_eq!((free.rows, free.cols, free.tokens()), (3, 5, 15));
        for cap in [15, 16, 1000] {
            let p = prepare_pixels(w, h, 3, &rgb, cap).unwrap();
            assert_eq!((p.rows, p.cols, p.digest == free.digest), (3, 5, true), "under {cap}");
        }
        let p = prepare_pixels(w, h, 3, &rgb, 6).unwrap();
        assert_eq!((p.rows, p.cols, p.tokens()), (2, 3, 6));
        let small = glm5next::resize(&Rgb { w, h, px: rgb.clone() }, 120, 72);
        assert_eq!((small.w, small.h), (120, 72));
        let q = prepare_rgb(&small, 0).unwrap();
        assert_eq!((p.patches == q.patches, p.digest == q.digest), (true, true));
        // 72 rows: the second token row's last 8 are the pad
        assert_eq!(p.patches.iter().filter(|&&b| b == PAD_BITS).count(), 120 * 8 * 3);
        for cap in 1..15 {
            let p = prepare_pixels(w, h, 3, &rgb, cap).unwrap();
            assert!(p.tokens() <= cap && p.tokens() >= 1, "{} tokens under {cap}", p.tokens());
            assert_eq!(p.patches.len(), p.tokens() * 4 * 1200);
        }
        // RGBA: the checkerboard at the image's own size, then the scaling
        let rgba = lcg(w * h * 4, 12);
        let p = prepare_pixels(w, h, 4, &rgba, 6).unwrap();
        let q = prepare_rgb(&glm5next::resize(&composite(w, h, &rgba), 120, 72), 0).unwrap();
        assert_eq!((p.rows, p.cols, p.patches == q.patches), (2, 3, true));
    }

    #[test]
    fn an_image_smaller_than_a_cell_is_one_token() {
        for cap in [0, 1, 7] {
            let p = prepare_pixels(1, 1, 3, &[255, 0, 128], cap).unwrap();
            assert_eq!((p.rows, p.cols, p.tokens(), p.patches.len()), (1, 1, 1, 4800));
            assert_eq!(&p.patches[..4], &[0x3f80, 0xbf80, 0x3b81, PAD_BITS]);
            assert!(p.patches[3..].iter().all(|&b| b == PAD_BITS));
        }
        let p = prepare_pixels(40, 40, 3, &[9; 4800], 1).unwrap();
        assert_eq!((p.tokens(), p.patches.iter().any(|&b| b == PAD_BITS)), (1, false));
        assert_eq!(prepare_pixels(39, 3, 4, &[0; 39 * 3 * 4], 0).unwrap().tokens(), 1);
        // a large image under a cap of one token
        let p = prepare_pixels(300, 100, 3, &lcg(300 * 100 * 3, 5), 1).unwrap();
        assert_eq!((p.rows, p.cols), (1, 1));
    }

    #[test]
    fn the_digest_covers_the_grid_and_the_patches() {
        // the same patches on two grids: 80 x 40 and 40 x 80 of one colour
        let a = prepare_pixels(80, 40, 3, &[77; 80 * 40 * 3], 0).unwrap();
        let b = prepare_pixels(40, 80, 3, &[77; 80 * 40 * 3], 0).unwrap();
        assert_eq!(((a.rows, a.cols), (b.rows, b.cols), a.patches == b.patches), ((1, 2), (2, 1), true));
        assert_ne!(a.digest, b.digest);
        // one pixel's last bit
        let mut px = vec![77u8; 80 * 40 * 3];
        px[80 * 40 * 3 - 1] = 76;
        let c = prepare_pixels(80, 40, 3, &px, 0).unwrap();
        assert_ne!(a.digest, c.digest);
        assert_eq!(a.digest, prepare_pixels(80, 40, 3, &[77; 80 * 40 * 3], 64).unwrap().digest);
        // sha256(rows u32 le, cols u32 le, the patches' u16 le)
        let mut raw = vec![1, 0, 0, 0, 2, 0, 0, 0];
        raw.extend(a.patches.iter().flat_map(|v| v.to_le_bytes()));
        assert_eq!(a.digest[..], Sha256::digest(&raw)[..]);
    }

    #[test]
    fn bad_input_is_refused() {
        assert_eq!(prepare(&[], 0).unwrap_err(), "an empty image (no bytes)");
        assert!(prepare(b"not an image at all", 0).unwrap_err().starts_with("unsupported image format"));
        assert!(prepare(b"BM\x00\x00\x00\x00", 0).unwrap_err().starts_with("unsupported image format"));
        let png = png_bytes(8, 8, png::ColorType::Rgba, png::BitDepth::Eight, &lcg(8 * 8 * 4, 1), None, None);
        assert!(prepare(&png, 0).is_ok());
        assert!(prepare(&png[..png.len() / 2], 0).unwrap_err().starts_with("PNG: "));
        assert!(prepare(&png[..8], 0).unwrap_err().starts_with("PNG: "));
        assert!(prepare(&[0xff, 0xd8, 0xff, 0xe0, 0, 0], 0).unwrap_err().starts_with("JPEG: "));
        assert!(prepare(b"GIF89a\x01\x00", 0).unwrap_err().starts_with("GIF: "));
        assert!(prepare(b"RIFF\x04\x00\x00\x00WEBP", 0).unwrap_err().starts_with("WebP: "));
        assert_eq!(prepare_pixels(0, 5, 3, &[], 0).unwrap_err(), "an empty image");
        assert_eq!(prepare_pixels(5, 0, 4, &[], 0).unwrap_err(), "an empty image");
        assert!(prepare_pixels(2, 2, 2, &[0; 8], 0).unwrap_err().starts_with("2 channels"));
        assert!(prepare_pixels(2, 2, 3, &[0; 11], 0).unwrap_err().starts_with("11 bytes for 2 x 2"));
        assert!(prepare_pixels(2, 2, 4, &[0; 12], 0).unwrap_err().starts_with("12 bytes for 2 x 2"));
        assert!(prepare_rgb(&Rgb { w: 2, h: 2, px: vec![0; 16] }, 0).is_err());
        assert!(prepare_rgb(&Rgb { w: 0, h: 2, px: vec![] }, 0).is_err());
    }

    fn png_bytes(w: usize, h: usize, color: png::ColorType, depth: png::BitDepth, data: &[u8], palette: Option<&[u8]>,
                 trns: Option<&[u8]>) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
            enc.set_color(color);
            enc.set_depth(depth);
            if let Some(p) = palette {
                enc.set_palette(p.to_vec());
            }
            if let Some(t) = trns {
                enc.set_trns(t.to_vec());
            }
            let mut wr = enc.write_header().unwrap();
            wr.write_image_data(data).unwrap();
            wr.finish().unwrap();
        }
        out
    }

    #[test]
    fn files_encoded_here_decode() {
        let (w, h) = (53, 47);
        let rgba = lcg(w * h * 4, 21);
        let rgb = lcg(w * h * 3, 22);
        let want_rgba = prepare_pixels(w, h, 4, &rgba, 0).unwrap();
        let want_rgb = prepare_pixels(w, h, 3, &rgb, 0).unwrap();
        assert_eq!((want_rgba.rows, want_rgba.cols), (2, 2));

        // PNG: RGBA keeps its alpha for the checkerboard, RGB is itself
        let b = png_bytes(w, h, png::ColorType::Rgba, png::BitDepth::Eight, &rgba, None, None);
        let d = decode(&b).unwrap();
        assert_eq!((d.w, d.h, d.channels, d.px == rgba), (w, h, 4, true));
        assert_eq!(prepare(&b, 0).unwrap().digest, want_rgba.digest);
        assert_ne!(want_rgba.digest, prepare_rgb(&crate::decode_rgb(&b).unwrap(), 0).unwrap().digest); // alpha dropped is another image
        let b = png_bytes(w, h, png::ColorType::Rgb, png::BitDepth::Eight, &rgb, None, None);
        assert_eq!(decode(&b).unwrap().channels, 3);
        assert_eq!(prepare(&b, 0).unwrap().digest, want_rgb.digest);
        // the cap through the bytes
        assert_eq!(prepare(&b, 1).unwrap().digest, prepare_pixels(w, h, 3, &rgb, 1).unwrap().digest);
        assert_eq!(prepare(&b, 1).unwrap().tokens(), 1);

        // PNG: a palette with tRNS is each entry's alpha; without, RGB
        let pal: Vec<u8> = lcg(16 * 3, 23);
        let alphas: Vec<u8> = lcg(16, 24);
        let idx: Vec<u8> = lcg(w * h, 25).iter().map(|v| v % 16).collect();
        let b = png_bytes(w, h, png::ColorType::Indexed, png::BitDepth::Eight, &idx, Some(&pal), Some(&alphas));
        let want: Vec<u8> = idx.iter().flat_map(|&i| { let i = i as usize; [pal[3 * i], pal[3 * i + 1], pal[3 * i + 2], alphas[i]] }).collect();
        let d = decode(&b).unwrap();
        assert_eq!((d.channels, d.px == want), (4, true));
        assert_eq!(prepare(&b, 0).unwrap().digest, prepare_pixels(w, h, 4, &want, 0).unwrap().digest);
        let b = png_bytes(w, h, png::ColorType::Indexed, png::BitDepth::Eight, &idx, Some(&pal), None);
        let want: Vec<u8> = idx.iter().flat_map(|&i| { let i = i as usize; [pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]] }).collect();
        let d = decode(&b).unwrap();
        assert_eq!((d.channels, d.px == want), (3, true));

        // JPEG: no alpha, the crate's decoder's pixels
        let jpg = turbojpeg::compress(turbojpeg::Image { pixels: &rgb[..], width: w, pitch: w * 3, height: h, format: turbojpeg::PixelFormat::RGB },
                                      90, turbojpeg::Subsamp::Sub2x2).unwrap();
        let d = decode(&jpg).unwrap();
        let r = crate::decode_rgb(&jpg).unwrap();
        assert_eq!((d.w, d.h, d.channels, d.px == r.px), (w, h, 3, true));
        let p = prepare(&jpg, 0).unwrap();
        assert_eq!((p.rows, p.cols, p.digest == prepare_rgb(&r, 0).unwrap().digest), (2, 2, true));

        // GIF: a transparent index is alpha 0, as is the screen the frame leaves out
        let gpal = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let (fw, fh) = (30usize, 20usize);
        let gidx: Vec<u8> = lcg(fw * fh, 26).iter().map(|v| v % 4).collect();
        let mut gif_t = Vec::new();
        let mut gif_o = Vec::new();
        for (out, clear) in [(&mut gif_t, Some(2u8)), (&mut gif_o, None)] {
            let mut enc = gif::Encoder::new(out, w as u16, h as u16, &gpal).unwrap();
            let mut f = gif::Frame::from_indexed_pixels(fw as u16, fh as u16, gidx.clone(), clear);
            (f.left, f.top) = (5, 9);
            enc.write_frame(&f).unwrap();
        }
        let d = decode(&gif_t).unwrap();
        assert_eq!((d.w, d.h, d.channels), (w, h, 4));
        for y in 0..h {
            for x in 0..w {
                let got = &d.px[(y * w + x) * 4..(y * w + x) * 4 + 4];
                let inside = (5..5 + fw).contains(&x) && (9..9 + fh).contains(&y);
                let i = if inside { gidx[(y - 9) * fw + x - 5] as usize } else { 2 };
                if i == 2 {
                    assert_eq!(got[3], 0, "({y}, {x})");
                } else {
                    assert_eq!(got, &[gpal[3 * i], gpal[3 * i + 1], gpal[3 * i + 2], 255], "({y}, {x})");
                }
            }
        }
        let p = prepare(&gif_t, 0).unwrap();
        assert_eq!(p.digest, prepare_pixels(w, h, 4, &d.px, 0).unwrap().digest);
        assert_eq!(p.patches[0], lut()[204]); // the screen's corner: the board
        let d = decode(&gif_o).unwrap();
        assert_eq!((d.channels, d.px == crate::decode_rgb(&gif_o).unwrap().px), (3, true));

        // WebP (lossless): RGBA keeps its alpha, RGB is itself
        let mut b = Vec::new();
        image_webp::WebPEncoder::new(&mut b).encode(&rgba, w as u32, h as u32, image_webp::ColorType::Rgba8).unwrap();
        let d = decode(&b).unwrap();
        assert_eq!((d.w, d.h, d.channels), (w, h, 4));
        // a lossless file need not keep the colour under alpha 0, which the board replaces anyway
        assert!(d.px.chunks(4).zip(rgba.chunks(4)).all(|(a, b)| a[3] == b[3] && (a[3] == 0 || a == b)));
        assert_eq!(prepare(&b, 0).unwrap().digest, want_rgba.digest);
        let mut b = Vec::new();
        image_webp::WebPEncoder::new(&mut b).encode(&rgb, w as u32, h as u32, image_webp::ColorType::Rgb8).unwrap();
        assert_eq!(decode(&b).unwrap().channels, 3);
        assert_eq!(prepare(&b, 0).unwrap().digest, want_rgb.digest);
    }

    /// Hand-built files and what Pillow 12.3.0 makes of them (`convert("RGBA")`, each with
    /// has_transparency_data): golden.py's `tiny` table.
    #[test]
    fn tiny_files_decode_as_pillow() {
        let cases: [(&str, &str, &[[u8; 4]]); 9] = [
            ("palette, tRNS per entry",
             "89504e470d0a1a0a0000000d49484452000000030000000108030000002c3ee48600000009504c54450a141e28323c46505a16ac84740000000274524e538007d374c0d00000000c4944415478da63606064020000080004081d630a0000000049454e44ae426082",
             &[[10, 20, 30, 128], [40, 50, 60, 7], [70, 80, 90, 255]]),
            ("palette, one transparent entry",
             "89504e470d0a1a0a0000000d49484452000000030000000108030000002c3ee48600000009504c54450a141e28323c46505a16ac84740000000274524e53ff00e5b7304a0000000c4944415478da63606064020000080004081d630a0000000049454e44ae426082",
             &[[10, 20, 30, 255], [40, 50, 60, 0], [70, 80, 90, 255]]),
            ("8-bit gray, key 7",
             "89504e470d0a1a0a0000000d49484452000000030000000108000000003e8b4b680000000274524e530007e8f7589b0000000c4944415478da6360e7f80f000128010fcb2d29130000000049454e44ae426082",
             &[[7, 7, 7, 0], [8, 8, 8, 255], [255, 255, 255, 255]]),
            ("1-bit gray, key 1",
             "89504e470d0a1a0a0000000d4948445200000004000000010100000000d14732600000000274524e5300010194fdae0000000a4944415478da6358000000a200a17105cb410000000049454e44ae426082",
             &[[255, 255, 255, 0], [0, 0, 0, 255], [255, 255, 255, 0], [0, 0, 0, 255]]),
            ("8-bit RGB, key (1, 2, 3)",
             "89504e470d0a1a0a0000000d4948445200000003000000010802000000948283e30000000674524e53000100020003c94babf5000000124944415478da63606462666462399162040003aa016cf77b57b90000000049454e44ae426082",
             &[[1, 2, 3, 0], [1, 2, 4, 255], [200, 100, 50, 255]]),
            ("8-bit gray and alpha",
             "89504e470d0a1a0a0000000d49484452000000020000000108040000005e2bb7010000000d4944415478da63e01239f11f0002f801e683cefebb0000000049454e44ae426082",
             &[[10, 10, 10, 20], [200, 200, 200, 255]]),
            ("16-bit gray and alpha: the high bytes",
             "89504e470d0a1a0a0000000d4948445200000003000000011004000000e179007c000000154944415478da631032f9ffff7f43c37fe6170cff012d44072d4c03a1330000000049454e44ae426082",
             &[[18, 18, 18, 255], [255, 255, 255, 128], [3, 3, 3, 0]]),
            ("16-bit RGBA: the high bytes",
             "89504e470d0a1a0a0000000d4948445200000002000000011006000000a4b2a3c90000001a4944415478da63103209ab98b5e7de87ffff19fe3332d4ff0700455e08b5b3b201c40000000049454e44ae426082",
             &[[18, 86, 154, 222], [255, 0, 1, 127]]),
            ("GIF, 4 x 2 screen, a 2 x 1 frame at (1, 1) of indices 2 3, index 2 transparent",
             "474946383961040002008101000a141e28323c46505a646e7821f90401000002002c0100010002000100000202d40a003b",
             &[[70, 80, 90, 0], [70, 80, 90, 0], [70, 80, 90, 0], [70, 80, 90, 0], [70, 80, 90, 0], [70, 80, 90, 0], [100, 110, 120, 255], [70, 80, 90, 0]]),
        ];
        for (name, file, pillow) in cases {
            let d = decode(&unhex(file)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((d.channels, d.w * d.h), (4, pillow.len()), "{name}");
            for (i, (got, want)) in d.px.chunks(4).zip(pillow).enumerate() {
                // the colour under alpha 0 is the board's whatever the file has
                assert!(got[3] == want[3] && (want[3] == 0 || got == want), "{name}: pixel {i} {got:?}, Pillow's {want:?}");
            }
            let want: Vec<u8> = pillow.iter().flatten().copied().collect();
            assert_eq!(composite(d.w, d.h, &d.px).px, composite(d.w, d.h, &want).px, "{name}");
        }
        // Colour keys Pillow 12.3.0 compares in the wrong scale; the PNG specification's
        // reading here (the keyed sample transparent, no other).
        //   4-bit gray 0 5 15 1, key 5: Pillow compares the key to the expanded 0 85 255 17 and
        //   leaves all four opaque
        let d = decode(&unhex("89504e470d0a1a0a0000000d494844520000000400000001040000000019a7bd100000000274524e53000506f939b70000000b4944415478da6360fd080000fe00f7a0876fbf0000000049454e44ae426082")).unwrap();
        assert_eq!(d.px, [0, 0, 0, 255, 85, 85, 85, 0, 255, 255, 255, 255, 17, 17, 17, 255]);
        //   16-bit gray 1000 232 100, key 1000: Pillow's gray saturates (1000 -> 255), which is
        //   kept, but it compares the key's low byte 232 to that: (255, opaque) (232, clear) (100)
        let d = decode(&unhex("89504e470d0a1a0a0000000d49484452000000030000000110000000006e1b972b0000000274524e5303e8f36ff4b10000000f4944415478da63607ec1f08221050007bd0238d5bc76d50000000049454e44ae426082")).unwrap();
        assert_eq!(d.px, [255, 255, 255, 0, 232, 232, 232, 255, 100, 100, 100, 255]);
        //   16-bit RGB (0x0102, 0x0304, 0x0506) (0x0200, 0x0400, 0x0600), key the first: Pillow
        //   compares the key's low bytes to the pixels' high bytes: (1, 3, 5, opaque) (2, 4, 6, clear)
        let d = decode(&unhex("89504e470d0a1a0a0000000d49484452000000020000000110020000002bd0349e0000000674524e530102030405065e92d116000000154944415478da63606462666165636260616063000000eb0022f5c271c60000000049454e44ae426082")).unwrap();
        assert_eq!(d.px, [1, 3, 5, 0, 2, 4, 6, 255]);
    }

    /// Values from a numpy transcription of the reference (Inkling2ImageProcessor._preprocess
    /// and convert_to_rgb, Inkling2VisionEmbeddings' reshape; fp32 arithmetic, bf16 by nearest
    /// neighbour), run 2026-11-16 with numpy 2.5.1 and, for the files, Pillow 12.3.0's decode:
    ///   a = pil_to_tensor(im.convert("RGBA" if im.has_transparency_data else "RGB"))  # or the LCG's bytes, [H][W][C] -> [C][H][W]
    ///   if C == 4: bg = where((arange(H)[:, None] // 8 + arange(W)[None, :] // 8) % 2 == 0, 204, 153)
    ///              a = ((a[:3] * a[3:4] + bg * (255 - a[3:4]) + 127) // 255) in int32, to uint8
    ///   x = pad(a.astype(float32), bottom rows * 40 - H, right cols * 40 - W, value -1.0)
    ///   x = (x - float32(127.5)) / float32(127.5)
    ///   pixel_values = x.reshape(3, rows, 40, cols, 40).transpose(1, 3, 2, 4, 0).reshape(rows * cols, 40, 40, 3)
    ///   patches = pixel_values.reshape(-1, 2, 20, 2, 20, 3).swapaxes(2, 3).reshape(-1, 1200) as bf16
    ///   sha256 of patches as little-endian u16; the digest with the grid's two u32 in front
    #[test]
    fn numpy_goldens() {
        // a random RGBA image, 93 x 47
        let (w, h) = (93, 47);
        let data = lcg(w * h * 4, 20261116);
        assert_eq!(&data[..8], &[123, 201, 141, 141, 138, 59, 167, 18]);
        let p = prepare_pixels(w, h, 4, &data, 0).unwrap();
        assert_eq!((p.rows, p.cols), (2, 3));
        for (i, want) in [(0, 0x3e7d), (1, 0x3f16), (2, 0x3ea7), (3, 0x3f10), (1199, 0x3f02), (1200, 0xbe05), (9659, 0xbf81),
                          (17196, 0xbf81), (27998, 0xbf81), (27999, 0xbf81), (28799, 0xbf81)] {
            assert_eq!(p.patches[i], want, "{i}");
        }
        assert_eq!(sha(&p.patches), "f6d57506a9ea0659749bced2faf7c9f70a78f113e0228de1c5649f6d1153aadb");
        assert_eq!(hex(&p.digest), "9eb7b806ef9d09c5403ddd4e9bc97f8f5e3bf0ee957a8af5aa7bf8b3c93a0bd5");

        // the crate's Pillow test images: (rows, cols, sha256 of the patches)
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pillow");
        for (name, rows, cols, want) in [
            ("small_rgba.png", 2, 2, "84dc1a6fdbcc0fa448cfa3d558f40a94dcfc1d68daa99054449296337b29fe5b"), // RGBA: on the board
            ("pal.png", 5, 8, "9cdac8e831366f73f056e05f9910ead8db6bb1493e5fa9b1ab957126df4b70c1"),
            ("gray16.png", 2, 3, "71184944358886a2dbed6506407c8ff5455e9b40be8bedd6b1b1ec27d6836307"),
            ("tall_gray.png", 23, 1, "b03de0f640f50c997ee91535c3e6a0841eacc6adb536625ee5957ebf478dc360"),
            ("synth.png", 12, 16, "756aaad887438378dc4408df2771d26450f34eb094187b1f406dba96cbbda692"),
            ("anim.gif", 2, 3, "e6718e24b1561fb468699b4ce1177c172f857d95805cf2bb99eb23a7b6e57b57"),
            ("synth.webp", 12, 16, "c69d4daffd653f7af84346fb91d8c73eb6ce871d9be65db9e3d0e0c729e2125b"),
            ("synth.jpg", 12, 16, "023cd7873b8a48816f54fde7a75557f3196765f5dd3029c05342097388c96947"),
            ("gray.jpg", 12, 16, "d762891c3cd07aaeb6ad5b883e0329d5ea10f3a43000540ba321c1cd6e72d883"),
            ("noise422.jpg", 9, 13, "907bad2620ab46f1bb308cbd827391ebbea38ec2f8d818ccc59c21321a57e660"),
        ] {
            let p = prepare(&std::fs::read(dir.join(name)).unwrap(), 0).unwrap();
            assert_eq!((p.rows, p.cols), (rows, cols), "{name}");
            assert_eq!(sha(&p.patches), want, "{name}");
        }
    }
}
