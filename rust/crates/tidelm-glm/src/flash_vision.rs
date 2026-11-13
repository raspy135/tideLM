//! GLM-5.3-Flash's vision tower (transformers' Glm5NextVisionModel): an image's patches
//! (bf16 [N][3 x 2 x 14 x 14], merge-window-major: tidelm_image::glm5next) in, one 4096-wide row a
//! 2 x 2 patch window out, the rows its `<|image|>` tokens take in place of their embeddings.
//!
//! The tower's weights (1.13 GB, BF16 as the checkpoint has them) and its scratch are carved from
//! whatever device memory it is given (Mem): in serving, expert slots lent for the request
//! (lend.rs), so a text-only server keeps that memory for experts; in a check, a buffer of its own.
//!
//! Per block (Glm5NextVisionBlock): RMSNorm (two roundings, eps 1e-5), qkv with bias, the q / k
//! heads' own RMSNorm, the 2D RoPE (rows' angles then columns', theta 1e4, over the whole head),
//! full attention within each image, proj with bias + the residual, RMSNorm, gate | up with biases,
//! SwiGLU clamped at 10, down with bias + the residual. Then the post norm, the 2 x 2 downsample conv
//! (a GEMM over 4 consecutive rows: the patches are window-major, its weight permuted at load), and
//! the merger (proj, LayerNorm + GELU, a clamped SwiGLU MLP to 4096). The GEMMs are ds41.cu's
//! BF16 kernels with an fp32 epilogue for the bias (vision.cu), so the rows agree with PyTorch's to
//! rounding, not bits.

use std::os::raw::c_void;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_linear_bf16, ds_mma_bf16};
use tidelm_kernels::flash::{flash_swiglu, flash_vit_ln_gelu, flash_vit_qknorm};
use tidelm_kernels::glm::glm_rmsnorm;
use tidelm_kernels::vision::{ds_vit_attn, ds_vit_epilogue, ds_vit_rope};
use tidelm_model::Checkpoint;

use crate::{ck, Result};

const D: usize = 1024;
const HEADS: usize = 16;
const INTER: usize = 4096;
const OUT: usize = 4096;
const PINTER: usize = 10240;
const DEPTH: usize = 24;
/// a patch's values: 3 channels x 2 frames x 14 x 14
pub const PATCH_K: usize = 1176;
/// padded to the tensor-core GEMM's multiple of 64
const PATCH_KP: usize = 1216;
const EPS: f32 = 1e-5;
const LIMIT: f32 = 10.0;
/// rows of a chunked GEMM's fp32 output: 16M floats
const F32_SCRATCH: usize = 16 << 20;
const P: &str = "model.visual.";

/// Device memory to carve the tower from: regions, each filled from its start.
pub struct Mem {
    regions: Vec<(*mut c_void, usize, usize)>,
}

impl Mem {
    pub fn new(regions: &[(*mut c_void, usize)]) -> Self {
        Mem { regions: regions.iter().map(|&(p, n)| (p, n, 0)).collect() }
    }

    /// `bytes` (256-aligned) from the first region with room.
    pub fn take(&mut self, bytes: usize) -> Result<*mut c_void> {
        let need = bytes.div_ceil(256) * 256;
        for r in self.regions.iter_mut() {
            let at = r.2.div_ceil(256) * 256;
            if at + need <= r.1 {
                r.2 = at + need;
                return Ok(unsafe { (r.0 as *mut u8).add(at) } as *mut c_void);
            }
        }
        Err(format!("vision tower: no region holds {bytes} more bytes").into())
    }
}

struct Block {
    n1: *mut c_void,
    n2: *mut c_void,
    qkv_w: *mut c_void,
    qkv_b: *mut c_void,
    qn: *mut c_void,
    kn: *mut c_void,
    proj_w: *mut c_void,
    proj_b: *mut c_void,
    gu_w: *mut c_void,
    gu_b: *mut c_void,
    down_w: *mut c_void,
    down_b: *mut c_void,
}

pub struct Tower {
    patch_w: *mut c_void,
    patch_b: *mut c_void,
    blocks: Vec<Block>,
    post: *mut c_void,
    ds_w: *mut c_void,
    ds_b: *mut c_void,
    mproj: *mut c_void,
    mln_w: *mut c_void,
    mln_b: *mut c_void,
    mgu: *mut c_void,
    mdown: *mut c_void,
    /// scratch for up to `max` patches
    max: usize,
    patches: *mut c_void,
    x: *mut c_void,
    h: *mut c_void,
    qkv: *mut c_void,
    att: *mut c_void,
    cs: *mut c_void,
    sn: *mut c_void,
    f32b: *mut c_void,
    gu: *mut c_void,
    mo: *mut c_void,
    y: *mut c_void,
    z: *mut c_void,
}

/// The tower's weight bytes and its scratch for `max` patches (what load takes from its Mem, with
/// each piece's alignment).
pub fn bytes_for(max: usize) -> usize {
    let block = 2 * D + 3 * D * D + 3 * D + 128 + D * D + D + 2 * INTER * D + 2 * INTER + D * INTER + D;
    let w = 2 * (D * PATCH_KP + D + DEPTH * block + D + OUT * 4 * D + OUT + OUT * OUT + 2 * OUT + 2 * PINTER * OUT + OUT * PINTER);
    let t = max / 4;
    let scratch = 2 * (max * PATCH_KP + 3 * max * D + 3 * max * D + 2048 * 2 * INTER + 2048 * INTER + 2 * t * OUT)
        + 4 * (2 * max * 32 + F32_SCRATCH);
    w + scratch + (DEPTH * 12 + 32) * 256
}

/// Bytes to give Tower::load when its Mem is several regions: bytes_for plus, for the room a
/// region's end may leave unused, twice the largest piece (the merger's gate | up, or qkv's scratch).
pub fn bytes_for_regions(max: usize) -> usize {
    bytes_for(max) + 2 * (2 * 2 * PINTER * OUT).max(2 * max * 3 * D)
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn put(st: &Stream, dst: *mut c_void, v: &[u16]) -> Result<()> {
    unsafe {
        tidelm_cuda::memcpy_async(dst, v.as_ptr() as *const c_void, v.len() * 2, tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
    }
    st.synchronize()?;
    Ok(())
}

/// TIDELM_VISION_DUMP=PREFIX (checks only): an intermediate [rows][cols] bf16 to PREFIX.NAME.bin.
fn dump(st: &Stream, name: &str, p: *const c_void, rows: usize, cols: usize) -> Result<()> {
    let Ok(prefix) = std::env::var("TIDELM_VISION_DUMP") else { return Ok(()) };
    st.synchronize()?;
    let mut v = vec![0u8; rows * cols * 2];
    ck(unsafe { tidelm_cuda::ffi::cudaMemcpy(v.as_mut_ptr() as *mut c_void, p, v.len(), tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST) },
       "vision dump")?;
    std::fs::write(format!("{prefix}.{name}.bin"), v)?;
    Ok(())
}

impl Tower {
    /// The tower's tensors from the checkpoint into `mem`, its scratch for `max` patches after them.
    pub fn load(ck: &Checkpoint, mem: &mut Mem, st: &Stream, max: usize) -> Result<Self> {
        let rd = |n: &str| -> Result<Vec<u16>> {
            let (b, m) = ck.read_tensor_vec(&format!("{P}{n}")).map_err(|e| format!("{P}{n}: {e}"))?;
            if m.dtype != "BF16" {
                return Err(format!("{P}{n}: {} (BF16 expected)", m.dtype).into());
            }
            Ok(u16s(&b))
        };
        let mut ld = |v: Vec<u16>| -> Result<*mut c_void> {
            let p = mem.take(v.len() * 2)?;
            put(st, p, &v)?;
            Ok(p)
        };
        // patch_embed [1024][3][2][14][14] as a [1024][1176] GEMM, K padded with zeros
        let pw = rd("patch_embed.proj.weight")?;
        let mut pwp = vec![0u16; D * PATCH_KP];
        for o in 0..D {
            pwp[o * PATCH_KP..o * PATCH_KP + PATCH_K].copy_from_slice(&pw[o * PATCH_K..(o + 1) * PATCH_K]);
        }
        let patch_w = ld(pwp)?;
        let patch_b = ld(rd("patch_embed.proj.bias")?)?;
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let b = format!("blocks.{i}.");
            let cat = |a: &str, c: &str| -> Result<Vec<u16>> {
                let mut v = rd(&format!("{b}{a}"))?;
                v.extend(rd(&format!("{b}{c}"))?);
                Ok(v)
            };
            blocks.push(Block {
                n1: ld(rd(&format!("{b}norm1.weight"))?)?,
                n2: ld(rd(&format!("{b}norm2.weight"))?)?,
                qkv_w: ld(rd(&format!("{b}attn.qkv.weight"))?)?,
                qkv_b: ld(rd(&format!("{b}attn.qkv.bias"))?)?,
                qn: ld(rd(&format!("{b}attn.q_norm.weight"))?)?,
                kn: ld(rd(&format!("{b}attn.k_norm.weight"))?)?,
                proj_w: ld(rd(&format!("{b}attn.proj.weight"))?)?,
                proj_b: ld(rd(&format!("{b}attn.proj.bias"))?)?,
                gu_w: ld(cat("mlp.gate_proj.weight", "mlp.up_proj.weight")?)?,
                gu_b: ld(cat("mlp.gate_proj.bias", "mlp.up_proj.bias")?)?,
                down_w: ld(rd(&format!("{b}mlp.down_proj.weight"))?)?,
                down_b: ld(rd(&format!("{b}mlp.down_proj.bias"))?)?,
            });
        }
        let post = ld(rd("post_layernorm.weight")?)?;
        // downsample [4096][1024][2][2] as a GEMM over a window's 4 rows: W'[o][k * 1024 + c] = W[o][c][k]
        let dw = rd("downsample.weight")?;
        let mut dwp = vec![0u16; OUT * 4 * D];
        for o in 0..OUT {
            for c in 0..D {
                for k in 0..4 {
                    dwp[o * 4 * D + k * D + c] = dw[(o * D + c) * 4 + k];
                }
            }
        }
        let ds_w = ld(dwp)?;
        let ds_b = ld(rd("downsample.bias")?)?;
        let mproj = ld(rd("merger.proj.weight")?)?;
        let mln_w = ld(rd("merger.post_projection_norm.weight")?)?;
        let mln_b = ld(rd("merger.post_projection_norm.bias")?)?;
        let mut g = rd("merger.gate_proj.weight")?;
        g.extend(rd("merger.up_proj.weight")?);
        let mgu = ld(g)?;
        let mdown = ld(rd("merger.down_proj.weight")?)?;
        let t = max / 4;
        let mut z = |b: usize| mem.take(b);
        Ok(Tower {
            patch_w, patch_b, blocks, post, ds_w, ds_b, mproj, mln_w, mln_b, mgu, mdown, max,
            patches: z(2 * max * PATCH_KP)?,
            x: z(2 * max * D)?,
            h: z(2 * max * D)?,
            qkv: z(2 * max * 3 * D)?,
            att: z(2 * max * D)?,
            cs: z(4 * max * 32)?,
            sn: z(4 * max * 32)?,
            f32b: z(4 * F32_SCRATCH)?,
            gu: z(2 * 2048 * 2 * INTER)?,
            mo: z(2 * 2048 * INTER)?,
            y: z(2 * t * OUT)?,
            z: z(2 * t * OUT)?,
        })
    }

    pub fn max_patches(&self) -> usize {
        self.max
    }

    /// y = x [m][k] . W^T [n][k] bf16 -> f32 scratch (bias / residual epilogue) or bf16 at `out`
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, st: &Stream, x: *const c_void, w: *const c_void, m: usize, n: usize, k: usize, out_bf16: Option<*mut c_void>)
            -> Result<()> {
        let null = std::ptr::null_mut();
        let (yb, yf) = match out_bf16 { Some(p) => (p, null), None => (null, self.f32b) };
        let rc = unsafe {
            if m > 16 {
                ds_mma_bf16(st.raw(), x, k as i64, w, m as i32, n as i32, k as i32, 1, yb, yf, n as i64, null)
            } else {
                ds_linear_bf16(st.raw(), x, k as i64, w, m as i32, n as i32, k as i32, 1, yb, yf, n as i64, null)
            }
        };
        ck(rc, "vision gemm")
    }

    /// out [rows][n] = bf16(residual + bf16(x W^T + bias)) over row chunks that fit the fp32 scratch
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, st: &Stream, x: *mut c_void, w: *const c_void, b: *const c_void, rows: usize, n: usize, k: usize,
              residual: *mut c_void, out: *mut c_void) -> Result<()> {
        let chunk = (F32_SCRATCH / n).max(1);
        for r0 in (0..rows).step_by(chunk) {
            let m = chunk.min(rows - r0);
            let xo = unsafe { (x as *mut u16).add(r0 * k) } as *const c_void;
            self.gemm(st, xo, w, m, n, k, None)?;
            let res: *const c_void = if residual.is_null() {
                std::ptr::null()
            } else {
                (unsafe { (residual as *mut u16).add(r0 * n) }) as *const c_void
            };
            let o = unsafe { (out as *mut u16).add(r0 * n) } as *mut c_void;
            ck(unsafe { ds_vit_epilogue(st.raw(), self.f32b, b, res, o, m as i32, n as i32, 0) }, "vision epilogue")?;
        }
        Ok(())
    }

    /// The images' patches (bf16 [N][1176] in order, `grids` each image's (patch rows, patch columns),
    /// both even) -> out bf16 [N / 4][4096]. Returns the rows written.
    pub fn forward(&self, st: &Stream, patches: &[u16], grids: &[(usize, usize)], out: *mut c_void) -> Result<usize> {
        let n: usize = grids.iter().map(|&(h, w)| h * w).sum();
        if n == 0 || n > self.max || patches.len() != n * PATCH_K || grids.iter().any(|&(h, w)| h % 2 != 0 || w % 2 != 0) {
            return Err(format!("vision: {n} patches (max {}), {} values, grids {grids:?}", self.max, patches.len()).into());
        }
        let s = st.raw();
        // patches padded to 1216 columns, the RoPE table (window-major positions)
        let mut pp = vec![0u16; n * PATCH_KP];
        for r in 0..n {
            pp[r * PATCH_KP..r * PATCH_KP + PATCH_K].copy_from_slice(&patches[r * PATCH_K..(r + 1) * PATCH_K]);
        }
        put(st, self.patches, &pp)?;
        let inv: Vec<f32> = (0..16).map(|j| 1.0f32 / 10000f32.powf((2 * j) as f32 / 32.0)).collect();
        let (mut cs, mut sn) = (Vec::with_capacity(n * 32), Vec::with_capacity(n * 32));
        for &(gh, gw) in grids {
            for bh in 0..gh / 2 {
                for bw in 0..gw / 2 {
                    for mh in 0..2 {
                        for mw in 0..2 {
                            let (ph, pw) = ((2 * bh + mh) as f32, (2 * bw + mw) as f32);
                            for j in 0..32 {
                                let a = if j < 16 { ph * inv[j] } else { pw * inv[j - 16] };
                                cs.push(a.cos());
                                sn.push(a.sin());
                            }
                        }
                    }
                }
            }
        }
        unsafe {
            for (dst, v) in [(self.cs, &cs), (self.sn, &sn)] {
                tidelm_cuda::memcpy_async(dst, v.as_ptr() as *const c_void, v.len() * 4, tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
            }
        }
        st.synchronize()?;
        let null = std::ptr::null_mut();
        self.linear(st, self.patches, self.patch_w, self.patch_b, n, D, PATCH_KP, null, self.x)?;
        dump(st, "embed", self.x, n, D)?;
        for (bi, b) in self.blocks.iter().enumerate() {
            ck(unsafe { glm_rmsnorm(s, self.x, D as i64, std::ptr::null(), 0, b.n1, self.h, D as i64, n as i32, D as i32, EPS) }, "norm1")?;
            self.linear(st, self.h, b.qkv_w, b.qkv_b, n, 3 * D, D, null, self.qkv)?;
            ck(unsafe { flash_vit_qknorm(s, self.qkv, n as i32, HEADS as i32, b.qn, b.kn, EPS) }, "qk norm")?;
            ck(unsafe { ds_vit_rope(s, self.qkv, n as i32, HEADS as i32, self.cs, self.sn) }, "vision rope")?;
            let mut r0 = 0;
            for &(gh, gw) in grids {
                let m = gh * gw;
                let q = unsafe { (self.qkv as *mut u16).add(r0 * 3 * D) } as *const c_void;
                let o = unsafe { (self.att as *mut u16).add(r0 * D) } as *mut c_void;
                ck(unsafe { ds_vit_attn(s, q, m as i32, HEADS as i32, o) }, "vision attention")?;
                r0 += m;
            }
            self.linear(st, self.att, b.proj_w, b.proj_b, n, D, D, self.x, self.x)?;
            ck(unsafe { glm_rmsnorm(s, self.x, D as i64, std::ptr::null(), 0, b.n2, self.h, D as i64, n as i32, D as i32, EPS) }, "norm2")?;
            for r0 in (0..n).step_by(2048) {
                let m = 2048.min(n - r0);
                let hx = unsafe { (self.h as *mut u16).add(r0 * D) } as *mut c_void;
                let xr = unsafe { (self.x as *mut u16).add(r0 * D) } as *mut c_void;
                self.linear(st, hx, b.gu_w, b.gu_b, m, 2 * INTER, D, null, self.gu)?;
                ck(unsafe { flash_swiglu(s, self.gu, (self.gu as *mut u16).add(INTER) as *const c_void, (2 * INTER) as i64, self.mo,
                                         INTER as i64, m as i32, INTER as i32, LIMIT) }, "vision swiglu")?;
                self.linear(st, self.mo, b.down_w, b.down_b, m, D, INTER, xr, xr)?;
            }
            if bi < 2 {
                dump(st, &format!("b{bi}.qkv"), self.qkv, n, 3 * D)?;
                dump(st, &format!("b{bi}.att"), self.att, n, D)?;
            }
            dump(st, &format!("b{bi}"), self.x, n, D)?;
        }
        ck(unsafe { glm_rmsnorm(s, self.x, D as i64, std::ptr::null(), 0, self.post, self.h, D as i64, n as i32, D as i32, EPS) }, "post norm")?;
        let t = n / 4;
        // the 2 x 2 downsample: a window's 4 rows are one row of 4096
        dump(st, "post", self.h, n, D)?;
        self.linear(st, self.h, self.ds_w, self.ds_b, t, OUT, 4 * D, null, self.y)?;
        dump(st, "down", self.y, t, OUT)?;
        self.gemm(st, self.y, self.mproj, t, OUT, OUT, Some(self.z))?;
        ck(unsafe { flash_vit_ln_gelu(s, self.z, self.mln_w, self.mln_b, self.y, t as i32, OUT as i32, EPS) }, "merger norm")?;
        for r0 in (0..t).step_by(512) {
            let m = 512.min(t - r0);
            let yi = unsafe { (self.y as *mut u16).add(r0 * OUT) } as *const c_void;
            self.gemm(st, yi, self.mgu, m, 2 * PINTER, OUT, Some(self.gu))?;
            ck(unsafe { flash_swiglu(s, self.gu, (self.gu as *mut u16).add(PINTER) as *const c_void, (2 * PINTER) as i64, self.mo,
                                     PINTER as i64, m as i32, PINTER as i32, LIMIT) }, "merger swiglu")?;
            let o = unsafe { (out as *mut u16).add(r0 * OUT) } as *mut c_void;
            self.gemm(st, self.mo, self.mdown, m, OUT, PINTER, Some(o))?;
        }
        st.synchronize()?;
        Ok(t)
    }
}

/// A tower in a buffer of its own (checks; serving lends expert slots instead).
pub struct OwnedTower {
    pub tower: Tower,
    _mem: DeviceBuffer<u8>,
}

impl OwnedTower {
    pub fn load(ck: &Checkpoint, st: &Stream, max: usize) -> Result<Self> {
        let buf = DeviceBuffer::<u8>::new(bytes_for(max))?;
        let mut mem = Mem::new(&[(buf.raw(), buf.len())]);
        let tower = Tower::load(ck, &mut mem, st, max)?;
        Ok(OwnedTower { tower, _mem: buf })
    }
}
