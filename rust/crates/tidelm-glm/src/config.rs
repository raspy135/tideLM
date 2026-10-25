//! The checkpoint's config.json as engine/glm/config.py reads it.

use std::path::Path;

use crate::Result;

#[derive(Clone, Debug)]
pub struct GlmConfig {
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
    pub experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub routed_scaling: f32,
    /// per layer (decoder layers, then the MTP layer): runs its own indexer ("full"); else it
    /// reuses the last full layer's picks. The MTP layer is always full (tp.py:563).
    pub index_full: Vec<bool>,
    /// per layer: a dense MLP (layers 0-2), else MoE (the MTP layer is MoE)
    pub dense: Vec<bool>,
    pub mtp: bool,
    pub eos: Vec<u32>,
    pub head_bits: usize,
    pub mtp_bits: usize,
}

impl GlmConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        let u = |k: &str| -> Result<usize> {
            raw.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("config.json: no {k}").into())
        };
        if raw.get("model_type").and_then(|v| v.as_str()) != Some("glm_moe_dsa") {
            return Err("config.json: not a glm_moe_dsa checkpoint".into());
        }
        let layers = u("num_hidden_layers")?;
        let mtp = u("num_nextn_predict_layers").unwrap_or(0) > 0;
        let types = |k: &str, a: &str| -> Result<Vec<bool>> {
            let v = raw.get(k).and_then(|v| v.as_array()).ok_or_else(|| format!("config.json: no {k}"))?;
            if v.len() != layers {
                return Err(format!("config.json: {k} has {} entries for {layers} layers", v.len()).into());
            }
            Ok(v.iter().map(|x| x.as_str() == Some(a)).collect())
        };
        let mut index_full = types("indexer_types", "full")?;
        let mut dense = types("mlp_layer_types", "dense")?;
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
        if u("n_group").unwrap_or(1) != 1 || raw.get("scoring_func").and_then(|v| v.as_str()) != Some("sigmoid") {
            return Err("config.json: routing must be sigmoid with one group".into());
        }
        let q = raw.get("quantization_config").ok_or("config.json: no quantization_config (an EXL3 checkpoint)")?;
        if q.get("quant_method").and_then(|v| v.as_str()) != Some("exl3")
            || q.get("codebook").and_then(|v| v.as_str()).unwrap_or("mul1") != "mul1"
        {
            return Err("config.json: the engine reads EXL3 checkpoints with the mul1 codebook".into());
        }
        let theta = raw.pointer("/rope_parameters/rope_theta").and_then(|v| v.as_f64())
            .or_else(|| raw.get("rope_theta").and_then(|v| v.as_f64())).ok_or("config.json: no rope_theta")?;
        let eos = match raw.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
            Some(v) => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            None => vec![],
        };
        let c = GlmConfig {
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
            experts: u("n_routed_experts")?,
            topk: u("num_experts_per_tok")?,
            moe_inter: u("moe_intermediate_size")?,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            rms_eps: raw.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
            rope_theta: theta,
            routed_scaling: raw.get("routed_scaling_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            index_full,
            dense,
            mtp,
            eos,
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
}
