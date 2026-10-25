//! `inference/config.json`, the fields the engine reads (tools/v41_ref.py::Args).

use std::path::Path;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub norm_eps: f32,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer: i64,
    pub compress_rope_theta: f32,
    pub compress_ratios: Vec<usize>,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram_layer_ids: Vec<usize>,
    pub engram_vocab_size: i64,
    pub engram_num_embeddings: Vec<i64>,
    pub engram_max_ngram_size: usize,
    pub engram_pad_id: usize,
    pub engram_compressed_vocab_size: i64,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub n_mtp_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_n_activated_experts: usize,
    /// the vision tower (inference/vision.py); a text-only checkpoint has no vision_* keys
    #[serde(default)]
    pub vision_n_layers: usize,
    #[serde(default)]
    pub vision_dim: usize,
    #[serde(default)]
    pub vision_n_heads: usize,
    #[serde(default)]
    pub vision_inter_dim: usize,
    #[serde(default)]
    pub vision_patch_size: usize,
    #[serde(default)]
    pub vision_downsample_ratio: usize,
    #[serde(default)]
    pub vision_max_n_token: usize,
    #[serde(default)]
    pub vision_min_pixels: usize,
    #[serde(default)]
    pub vision_max_wh_ratio: Option<f64>,
    #[serde(default)]
    pub vision_rope_theta: f32,
    #[serde(default)]
    pub image_token_id: u32,
}

impl Config {
    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let raw = std::fs::read(model_dir.join("inference/config.json"))?;
        let mut cfg: Config = serde_json::from_slice(&raw)?;
        // The engine serves a wider index window than the checkpoint's reference 512
        // (v41_engine.py: DSV41_INDEX_TOPK default 1024); it only matters past 1024
        // compressed positions, where the indexer starts pruning.
        cfg.index_topk = 1024;
        Ok(cfg)
    }

    /// The layer whose compressed KV layer `l` attends over (the latest kv source at or
    /// below it), or None for window-only layers.
    pub fn kv_source_of(&self, l: usize) -> Option<usize> {
        if self.compress_ratios[l] == 0 {
            return None;
        }
        self.kv_source_layers.iter().copied().filter(|&s| s <= l).max()
    }

    pub fn hc_mix_width(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }
}
