//! The checkpoint's config.json as engine/glm/config.py reads it: GLM-5.3 (`glm_moe_dsa`),
//! GLM-5.3-Flash (`glm5_next`, its `text_config`) or Inkling 2 (`inkling2`, its `text_config`).

use std::path::Path;

use crate::Result;

/// The model family a checkpoint is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// GLM-5.3: MLA + DSA on every layer (RoPE 64), plain residual, EXL3 everywhere
    Glm53,
    /// GLM-5.3-Flash: KDA linear attention on most layers, MLA + DSA without RoPE on every 4th,
    /// mHC hyper-connections, EXL3 routed experts and BF16 everything else
    Flash,
    /// Inkling 2: grouped-query attention with a relative bias and a sink (a 1,024-token window on
    /// most layers, the whole context on every 6th), short convolutions, the block attention
    /// residual, EXL3 routed experts with no shared expert, BF16 everything else
    Inkling2,
}

/// A layer's attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    /// GLM-5.3's MLA with RoPE and its DSA indexer
    Mla,
    /// Kimi Delta Attention (GLM-5.3-Flash's linear_attention layers)
    Kda,
    /// GLM-5.3-Flash's deepseek_sparse_attention: MLA without RoPE, the indexer over 4:1 key pools
    Dsa,
    /// Inkling 2's sliding-window layer: keys p - window + 1 ..= p
    Swa,
    /// Inkling 2's full-attention layer: every earlier key, q / k / v each from its own mix of the
    /// block states, k and v through the dynamic short convolution
    Full,
}

/// mHC (hyper-connections, DeepSeek-V4's): the residual is `mult` streams.
#[derive(Clone, Copy, Debug)]
pub struct Hc {
    pub mult: usize,
    pub iters: usize,
    pub eps: f32,
}

/// KDA's shapes.
#[derive(Clone, Copy, Debug)]
pub struct KdaCfg {
    pub heads: usize,
    pub dim: usize,
    pub lower: f32,
}

/// Inkling 2's shapes and constants (modeling_inkling2.py's names in the comments).
#[derive(Clone, Copy, Debug)]
pub struct InkCfg {
    /// key / value heads (num_key_value_heads); a query head's is head * kv_heads / heads
    pub kv_heads: usize,
    pub head_dim: usize,
    /// a head's relative state (d_rel): mixed with rel_logits_proj into a bias per backward distance
    pub d_rel: usize,
    /// sliding_window_size: a Swa row's keys, itself included; also its relative bias's extent
    pub window: usize,
    /// rel_extent: distances a Full row's relative bias covers (zero past it)
    pub rel_extent: usize,
    /// the short convolutions' taps (sconv_kernel_size)
    pub conv: usize,
    /// n_layers_per_attn_res_block: a layer l with l % block == 0 saves its input as a block state
    /// and restarts the residual from its attention output
    pub block: usize,
    /// qk_l2_clip_threshold, rel_l2_clip_threshold
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// mlp_gate_softcap, mlp_up_softcap: c * tanh(x / c) on the SwiGLU's linear gate factor and up
    pub gate_cap: f32,
    pub up_cap: f32,
    /// final_logit_softcapping
    pub logit_cap: f32,
    /// Full layers scale q and the relative state by 1 + alpha * ln(max((p + 1) / floor, 1))
    pub log_floor: f32,
    pub log_alpha: f32,
    /// unpadded_vocab_size: the head's rows that are tokens (the rest are never emitted)
    pub real_vocab: usize,
}

#[derive(Clone, Debug)]
pub struct GlmConfig {
    pub arch: Arch,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    /// tokens a pooled indexer key covers (GLM-5.3-Flash 4; 1: the indexer scores tokens)
    pub index_pool: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub routed_scaling: f32,
    /// clamp of the SwiGLU's gate (above) and up (both sides); infinity: none
    pub swiglu_limit: f32,
    /// per layer (decoder layers, then the MTP layer): its attention
    pub attn: Vec<AttnKind>,
    /// per layer: runs its own indexer ("full"); else it reuses the last full layer's picks. The
    /// MTP layer is always full (tp.py:563). Only meaningful on MLA / DSA layers.
    pub index_full: Vec<bool>,
    /// per layer: a dense MLP (layers 0-2), else MoE (the MTP layer is MoE)
    pub dense: Vec<bool>,
    pub hc: Option<Hc>,
    pub kda: Option<KdaCfg>,
    pub ink: Option<InkCfg>,
    pub mtp: bool,
    pub eos: Vec<u32>,
    pub head_bits: usize,
    pub mtp_bits: usize,
}

impl GlmConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let top: serde_json::Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        match top.get("model_type").and_then(|v| v.as_str()) {
            Some("glm_moe_dsa") => Self::load_glm53(&top),
            Some("glm5_next") => Self::load_flash(&top),
            Some("inkling2") => Self::load_inkling2(&top),
            t => Err(format!("config.json: model_type {t:?}, not glm_moe_dsa, glm5_next or inkling2").into()),
        }
    }

    fn load_glm53(raw: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| -> Result<usize> {
            raw.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("config.json: no {k}").into())
        };
        let layers = u("num_hidden_layers")?;
        let mtp = u("num_nextn_predict_layers").unwrap_or(0) > 0;
        let mut index_full = types(raw, layers, "indexer_types", "full")?;
        let mut dense = types(raw, layers, "mlp_layer_types", "dense")?;
        if !index_full[0] {
            return Err("config.json: layer 0's indexer must be full".into());
        }
        if mtp {
            index_full.push(true);
            dense.push(false);
        }
        let interleave = |k: &str| raw.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
        if !interleave("rope_interleave") || !interleave("indexer_rope_interleave") {
            return Err("config.json: only interleaved RoPE is implemented".into());
        }
        check_routing(raw)?;
        let q = raw.get("quantization_config").ok_or("config.json: no quantization_config (an EXL3 checkpoint)")?;
        if q.get("quant_method").and_then(|v| v.as_str()) != Some("exl3")
            || q.get("codebook").and_then(|v| v.as_str()).unwrap_or("mul1") != "mul1"
        {
            return Err("config.json: the engine reads EXL3 checkpoints with the mul1 codebook".into());
        }
        let theta = raw.pointer("/rope_parameters/rope_theta").and_then(|v| v.as_f64())
            .or_else(|| raw.get("rope_theta").and_then(|v| v.as_f64())).ok_or("config.json: no rope_theta")?;
        let n = layers + mtp as usize;
        let c = GlmConfig {
            arch: Arch::Glm53,
            hidden: u("hidden_size")?,
            layers,
            heads: u("num_attention_heads")?,
            q_lora: u("q_lora_rank")?,
            kv_lora: u("kv_lora_rank")?,
            nope: u("qk_nope_head_dim")?,
            rope: u("qk_rope_head_dim")?,
            v_dim: u("v_head_dim")?,
            index_heads: u("index_n_heads")?,
            index_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
            index_pool: 1,
            experts: u("n_routed_experts")?,
            topk: u("num_experts_per_tok")?,
            moe_inter: u("moe_intermediate_size")?,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            rms_eps: raw.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
            rope_theta: theta,
            routed_scaling: raw.get("routed_scaling_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            swiglu_limit: f32::INFINITY,
            attn: vec![AttnKind::Mla; n],
            index_full,
            dense,
            hc: None,
            kda: None,
            ink: None,
            mtp,
            eos: eos(raw),
            head_bits: q.get("head_bits").and_then(|v| v.as_u64()).unwrap_or(6) as usize,
            mtp_bits: q.get("mtp_bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
        };
        // what the kernels are built for
        let want = [(c.hidden, 6144, "hidden"), (c.kv_lora, 512, "kv_lora"), (c.rope, 64, "rope"), (c.nope, 192, "nope"),
                    (c.v_dim, 256, "v_head_dim"), (c.index_heads, 32, "index heads"), (c.index_dim, 128, "index dim"),
                    (c.experts, 256, "experts")];
        for (got, w, what) in want {
            if got != w {
                return Err(format!("config.json: {what} {got}, the kernels take {w}").into());
            }
        }
        if c.topk > 8 {
            return Err("config.json: top-k past 8".into());
        }
        Ok(c)
    }

    /// GLM-5.3-Flash: config.json's text_config (the vision tower is not served), the EXL3
    /// quantization at the top level (it covers the routed experts only).
    fn load_flash(top: &serde_json::Value) -> Result<Self> {
        let raw = top.get("text_config").ok_or("config.json: glm5_next without text_config")?;
        let u = |k: &str| -> Result<usize> {
            raw.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("config.json: no {k}").into())
        };
        let f = |k: &str| raw.get(k).and_then(|v| v.as_f64());
        let b = |k: &str| raw.get(k).and_then(|v| v.as_bool());
        let layers = u("num_hidden_layers")?;
        let mtp = u("num_nextn_predict_layers").unwrap_or(0) > 0;
        let lt = raw.get("layer_types").and_then(|v| v.as_array()).ok_or("config.json: no layer_types")?;
        if lt.len() != layers {
            return Err(format!("config.json: layer_types has {} entries for {layers} layers", lt.len()).into());
        }
        let mut attn = Vec::with_capacity(layers + 1);
        for (l, t) in lt.iter().enumerate() {
            attn.push(match t.as_str() {
                Some("linear_attention") => AttnKind::Kda,
                Some("deepseek_sparse_attention") => AttnKind::Dsa,
                t => return Err(format!("config.json: layer {l}'s type {t:?}").into()),
            });
        }
        let mut index_full = types(raw, layers, "indexer_types", "full")?;
        let mut dense = types(raw, layers, "mlp_layer_types", "dense")?;
        if index_full.iter().zip(&attn).any(|(&full, &a)| a == AttnKind::Dsa && !full) {
            return Err("config.json: a sparse-attention layer that shares its indexer (not implemented)".into());
        }
        if mtp {
            attn.push(AttnKind::Dsa);
            index_full.push(true);
            dense.push(false);
        }
        check_routing(raw)?;
        let q = top.get("quantization_config").ok_or("config.json: no quantization_config (an EXL3 checkpoint)")?;
        if q.get("quant_method").and_then(|v| v.as_str()) != Some("exl3")
            || !matches!(q.get("codebook").and_then(|v| v.as_str()), Some("mcg") | Some("mul1"))
        {
            return Err("config.json: the routed experts must be EXL3 (mcg or mul1)".into());
        }
        if !b("mhc").unwrap_or(false) || u("hc_mult")? != 4 {
            return Err("config.json: GLM-5.3-Flash without mHC of 4 streams".into());
        }
        let la = raw.get("linear_attn_config").ok_or("config.json: no linear_attn_config")?;
        let lu = |k: &str| la.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("linear_attn_config: no {k}"));
        let kda = KdaCfg {
            heads: lu("num_heads")?,
            dim: lu("head_dim")?,
            lower: la.get("gate_lower_bound").and_then(|v| v.as_f64()).ok_or("linear_attn_config: no gate_lower_bound")? as f32,
        };
        if kda.dim != 128 || lu("short_conv_kernel_size")? != 4 {
            return Err("config.json: KDA other than head_dim 128 and a 4-tap conv".into());
        }
        if !b("index_kpool_compress").unwrap_or(false) || !b("index_kpool_always_select_tail").unwrap_or(false) {
            return Err("config.json: an indexer without compressed key pools and the tail (not implemented)".into());
        }
        let c = GlmConfig {
            arch: Arch::Flash,
            hidden: u("hidden_size")?,
            layers,
            heads: u("num_attention_heads")?,
            q_lora: u("q_lora_rank")?,
            kv_lora: u("kv_lora_rank")?,
            nope: u("qk_nope_head_dim")?,
            rope: u("qk_rope_head_dim")?,
            v_dim: u("v_head_dim")?,
            index_heads: u("index_n_heads")?,
            index_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
            index_pool: u("index_kpool")?,
            experts: u("n_routed_experts")?,
            topk: u("num_experts_per_tok")?,
            moe_inter: u("moe_intermediate_size")?,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            rms_eps: f("rms_norm_eps").unwrap_or(1e-5) as f32,
            rope_theta: 0.0,
            routed_scaling: f("routed_scaling_factor").unwrap_or(1.0) as f32,
            swiglu_limit: f("swiglu_limit").map(|v| v as f32).unwrap_or(f32::INFINITY),
            attn,
            index_full,
            dense,
            hc: Some(Hc { mult: 4, iters: u("hc_sinkhorn_iters")?, eps: f("hc_eps").unwrap_or(1e-6) as f32 }),
            kda: Some(kda),
            ink: None,
            mtp,
            eos: eos(raw),
            head_bits: 16,
            mtp_bits: 4,
        };
        let want = [(c.hidden, 4096, "hidden"), (c.kv_lora, 512, "kv_lora"), (c.rope, 0, "rope"), (c.nope, 256, "nope"),
                    (c.v_dim, 256, "v_head_dim"), (c.index_heads, 32, "index heads"), (c.index_dim, 128, "index dim"),
                    (c.index_pool, 4, "index pool"), (c.experts, 288, "experts")];
        for (got, w, what) in want {
            if got != w {
                return Err(format!("config.json: {what} {got}, the kernels take {w}").into());
            }
        }
        if c.topk > 8 || c.index_topk % c.index_pool != 0 {
            return Err("config.json: top-k past 8, or an index top-k that is not whole pools".into());
        }
        Ok(c)
    }

    /// Inkling 2: config.json's text_config (the vision tower is not served; the DSpark drafter of
    /// mtp_config is not ported, so `mtp` is false and decoding is plain), the EXL3 quantization at
    /// the top level (the routed experts only, as GLM-5.3-Flash's).
    fn load_inkling2(top: &serde_json::Value) -> Result<Self> {
        let raw = top.get("text_config").ok_or("config.json: inkling2 without text_config")?;
        let u = |k: &str| -> Result<usize> {
            raw.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("config.json: no {k}").into())
        };
        let f = |k: &str| -> Result<f32> {
            raw.get(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("config.json: no {k}").into())
        };
        let layers = u("num_hidden_layers")?;
        let local = raw.get("local_layer_ids").and_then(|v| v.as_array()).ok_or("config.json: no local_layer_ids")?;
        let mut attn = vec![AttnKind::Full; layers];
        for v in local {
            let l = v.as_u64().map(|v| v as usize).filter(|&l| l < layers).ok_or("config.json: a local_layer_ids entry past the layers")?;
            attn[l] = AttnKind::Swa;
        }
        let q = top.get("quantization_config").ok_or("config.json: no quantization_config (an EXL3 checkpoint)")?;
        if q.get("quant_method").and_then(|v| v.as_str()) != Some("exl3")
            || !matches!(q.get("codebook").and_then(|v| v.as_str()), Some("mcg") | Some("mul1"))
        {
            return Err("config.json: the routed experts must be EXL3 (mcg or mul1); the official NVFP4 / MXFP8 experts are not read".into());
        }
        // the router: sigmoid scores, the bias in the selection only (the reference's defaults; the
        // checkpoint's config does not name a scoring function)
        if raw.get("scoring_func").and_then(|v| v.as_str()).unwrap_or("sigmoid") != "sigmoid"
            || raw.get("n_shared_experts").and_then(|v| v.as_u64()).unwrap_or(0) != 0
            || !raw.get("use_gate_bias").and_then(|v| v.as_bool()).unwrap_or(false)
        {
            return Err("config.json: Inkling 2 routing other than sigmoid with a selection bias and no shared expert".into());
        }
        let ink = InkCfg {
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            d_rel: u("d_rel")?,
            window: u("sliding_window_size")?,
            rel_extent: u("rel_extent")?,
            conv: u("sconv_kernel_size")?,
            block: u("n_layers_per_attn_res_block")?,
            qk_clip: f("qk_l2_clip_threshold")?,
            rel_clip: f("rel_l2_clip_threshold")?,
            gate_cap: f("mlp_gate_softcap")?,
            up_cap: f("mlp_up_softcap")?,
            logit_cap: f("final_logit_softcapping")?,
            log_floor: f("log_scaling_n_floor")?,
            log_alpha: f("log_scaling_alpha")?,
            real_vocab: u("unpadded_vocab_size")?,
        };
        let c = GlmConfig {
            arch: Arch::Inkling2,
            hidden: u("hidden_size")?,
            layers,
            heads: u("num_attention_heads")?,
            q_lora: 0,
            kv_lora: 0,
            nope: 0,
            rope: 0,
            v_dim: 0,
            index_heads: 0,
            index_dim: 0,
            index_topk: 0,
            index_pool: 1,
            experts: u("n_routed_experts")?,
            topk: u("num_experts_per_tok")?,
            moe_inter: u("intermediate_size")?,
            inter: 0,
            vocab: u("vocab_size")?,
            rms_eps: f("rms_norm_eps")?,
            rope_theta: 0.0,
            routed_scaling: f("route_scale")?,
            swiglu_limit: f32::INFINITY,
            attn,
            index_full: vec![false; layers],
            dense: vec![false; layers],
            hc: None,
            kda: None,
            ink: Some(ink),
            mtp: false,
            eos: eos(top),
            head_bits: 16,
            mtp_bits: 4,
        };
        // what the kernels are built for
        let want = [(c.hidden, 4608, "hidden"), (c.heads, 48, "heads"), (ink.kv_heads, 8, "key/value heads"),
                    (ink.head_dim, 128, "head dim"), (ink.d_rel, 16, "d_rel"), (ink.window, 1024, "window"),
                    (ink.rel_extent, 1024, "rel_extent"), (ink.conv, 4, "conv taps"), (c.experts, 384, "experts"),
                    (c.moe_inter, 2304, "expert intermediate"),
                    (ink.gate_cap.to_bits() as usize, 4f32.to_bits() as usize, "gate softcap (bits)"),
                    (ink.up_cap.to_bits() as usize, 25f32.to_bits() as usize, "up softcap (bits)"),
                    (u("swa_num_attention_heads")?, c.heads, "window layers' heads"),
                    (u("swa_num_key_value_heads")?, ink.kv_heads, "window layers' key/value heads"),
                    (u("swa_head_dim")?, ink.head_dim, "window layers' head dim")];
        for (got, w, what) in want {
            if got != w {
                return Err(format!("config.json: {what} {got}, the kernels take {w}").into());
            }
        }
        if c.topk > 8 || ink.block == 0 || ink.real_vocab > c.vocab || c.attn[0] != AttnKind::Swa {
            return Err("config.json: top-k past 8, no residual block size, a vocabulary past the head, or a full-attention layer 0".into());
        }
        Ok(c)
    }

    /// Decoder layers plus the MTP layer.
    pub fn n_layers(&self) -> usize {
        self.layers + self.mtp as usize
    }

    pub fn mtp_layer(&self) -> usize {
        self.layers
    }

    pub fn qk_dim(&self) -> usize {
        self.nope + self.rope
    }

    /// kv_b's rows a head (key nope dims, then value dims)
    pub fn kvb_rows(&self) -> usize {
        self.nope + self.v_dim
    }

    pub fn is_flash(&self) -> bool {
        self.arch == Arch::Flash
    }

    pub fn is_inkling(&self) -> bool {
        self.arch == Arch::Inkling2
    }

    /// Inkling 2: the block states a layer's attention input mixes with its own input (the inputs
    /// of the earlier layers l' with l' % block == 0); its MLP input sees one more when l itself
    /// is such a layer.
    pub fn ink_blocks_before(&self, l: usize) -> usize {
        self.ink.map_or(0, |k| l.div_ceil(k.block))
    }

    /// Inkling 2: the block states in all (the final mix's count).
    pub fn ink_blocks(&self) -> usize {
        self.ink_blocks_before(self.layers)
    }

    /// The decoder's KDA layers, in order (their state's layer index is their place here).
    pub fn kda_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&l| self.attn[l] == AttnKind::Kda).collect()
    }

    /// Tokens a sparse-attention row attends at most: the indexer's top-k and, with key pools,
    /// the incomplete tail pool's tokens.
    pub fn attend_max(&self) -> usize {
        self.index_topk + self.index_pool - 1
    }
}

fn types(raw: &serde_json::Value, layers: usize, k: &str, a: &str) -> Result<Vec<bool>> {
    let v = raw.get(k).and_then(|v| v.as_array()).ok_or_else(|| format!("config.json: no {k}"))?;
    if v.len() != layers {
        return Err(format!("config.json: {k} has {} entries for {layers} layers", v.len()).into());
    }
    Ok(v.iter().map(|x| x.as_str() == Some(a)).collect())
}

fn check_routing(raw: &serde_json::Value) -> Result<()> {
    if raw.get("n_group").and_then(|v| v.as_u64()).unwrap_or(1) != 1
        || raw.get("scoring_func").and_then(|v| v.as_str()) != Some("sigmoid")
    {
        return Err("config.json: routing must be sigmoid with one group".into());
    }
    Ok(())
}

fn eos(raw: &serde_json::Value) -> Vec<u32> {
    match raw.get("eos_token_id") {
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
        Some(v) => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        None => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_families_parse() {
        let flash = Path::new("/home/ryan/models/GLM-5.3-Flash-4bpw-dense");
        if flash.join("config.json").exists() {
            let c = GlmConfig::load(flash).unwrap();
            assert_eq!(c.arch, Arch::Flash);
            assert_eq!((c.layers, c.n_layers(), c.hidden, c.experts), (45, 46, 4096, 288));
            assert_eq!(c.kda_layers().len(), 34);
            assert_eq!(c.attn[3], AttnKind::Dsa);
            assert_eq!(c.attn[45], AttnKind::Dsa);
            assert_eq!(c.dense.iter().filter(|&&d| d).count(), 3);
            assert_eq!(c.attend_max(), 2051);
            assert_eq!(c.swiglu_limit, 10.0);
            assert_eq!(c.eos, vec![154820, 154827, 154829]);
        }
        let ink = Path::new("/home/ryan/models/Inkling-2-Small-EXL3-4bpw");
        if ink.join("config.json").exists() {
            let c = GlmConfig::load(ink).unwrap();
            let k = c.ink.unwrap();
            assert_eq!(c.arch, Arch::Inkling2);
            assert_eq!((c.layers, c.n_layers(), c.hidden, c.experts, c.moe_inter, c.heads), (48, 48, 4608, 384, 2304, 48));
            assert_eq!(c.attn.iter().filter(|&&a| a == AttnKind::Full).count(), 8);
            assert!((0..48).all(|l| (c.attn[l] == AttnKind::Full) == (l % 6 == 5)));
            assert!(!c.mtp && c.dense.iter().all(|&d| !d) && c.hc.is_none());
            assert_eq!((k.kv_heads, k.window, k.block, k.real_vocab), (8, 1024, 6, 200058));
            assert_eq!((c.routed_scaling, k.gate_cap, k.up_cap, k.logit_cap), (8.0, 4.0, 25.0, 30.0));
            assert_eq!(c.eos, vec![200006]);
            // layer 0 mixes nothing, layers 1..=6 one saved state, the final mix all eight
            assert_eq!((c.ink_blocks_before(0), c.ink_blocks_before(1), c.ink_blocks_before(6), c.ink_blocks_before(7)), (0, 1, 1, 2));
            assert_eq!(c.ink_blocks(), 8);
        }
        let glm = Path::new("/home/ryan/models/GLM-5.3-dense-exlr8");
        if glm.join("config.json").exists() {
            let c = GlmConfig::load(glm).unwrap();
            assert_eq!(c.arch, Arch::Glm53);
            assert!(c.attn.iter().all(|&a| a == AttnKind::Mla) && c.hc.is_none() && c.swiglu_limit.is_infinite());
        }
    }
}
