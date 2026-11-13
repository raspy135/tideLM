//! The checkpoint's config.json as engine/glm/config.py reads it: GLM-5.3 (`glm_moe_dsa`) or
//! GLM-5.3-Flash (`glm5_next`, its `text_config`).

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
            t => Err(format!("config.json: model_type {t:?}, not glm_moe_dsa or glm5_next").into()),
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
        let flash = Path::new("/home/ryan/models/GLM-5.3-Flash-3.5bpw-dense");
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
        let glm = Path::new("/home/ryan/models/GLM-5.3-dense-exlr8");
        if glm.join("config.json").exists() {
            let c = GlmConfig::load(glm).unwrap();
            assert_eq!(c.arch, Arch::Glm53);
            assert!(c.attn.iter().all(|&a| a == AttnKind::Mla) && c.hc.is_none() && c.swiglu_limit.is_infinite());
        }
    }
}
