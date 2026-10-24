//! Tokenizer access and the engram compressed token map.
//!
//! The map is `inference/engram.py::build_compressed_token_map`: every vocabulary id is
//! decoded on its own, normalized (NFKC, NFD, strip accents, lowercase, whitespace runs
//! to one space, strip — with a sentinel so a lone-space token survives), and ids whose
//! normalized text agrees share one compressed id. HF `tokenizers` is a Rust library;
//! the Python calls the same normalizer code this crate calls, so the map is identical
//! by construction and the test pins it against the checkpoint's sha256 anyway.

use std::collections::HashMap;
use std::path::Path;

use tokenizers::normalizers::{
    Lowercase, NormalizerWrapper, Replace, Sequence, Strip, StripAccents, NFD, NFKC,
};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

pub use tokenizers;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub struct Text {
    pub tokenizer: Tokenizer,
}

impl Text {
    pub fn load(model_dir: &Path) -> Result<Self, Error> {
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
        Ok(Text { tokenizer })
    }

    /// Plain encoding, no special tokens (the engine adds its own template).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.tokenizer.encode(text, false)?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        Ok(self.tokenizer.decode(ids, false)?)
    }

    /// Vocabulary size including added tokens (Python's `len(tokenizer)`).
    pub fn vocab_len(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// `build_compressed_token_map`: (map from vocabulary id to compressed id, size of
    /// the compressed vocabulary).
    pub fn compressed_token_map(&self) -> Result<(Vec<i64>, usize), Error> {
        let sentinel = "\u{e000}";
        let normalizer = Sequence::new(vec![
            NormalizerWrapper::NFKC(NFKC),
            NormalizerWrapper::NFD(NFD),
            NormalizerWrapper::StripAccents(StripAccents),
            NormalizerWrapper::Lowercase(Lowercase),
            NormalizerWrapper::Replace(Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::Regex(r"[ \t\r\n]+".into()),
                " ",
            )?),
            NormalizerWrapper::Replace(Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::Regex(r"^ $".into()),
                sentinel,
            )?),
            NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
            NormalizerWrapper::Replace(Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::String(sentinel.into()),
                " ",
            )?),
        ]);
        let n = self.vocab_len();
        let mut key_to_new: HashMap<String, i64> = HashMap::new();
        let mut lookup = vec![0i64; n];
        for (id, slot) in lookup.iter_mut().enumerate() {
            let text = self.tokenizer.decode(&[id as u32], false)?;
            let key = if text.contains('\u{fffd}') {
                // a partial UTF-8 byte token: nothing to normalize, key it by its raw form
                self.tokenizer.id_to_token(id as u32).unwrap_or_default()
            } else {
                let mut s = NormalizedString::from(text.as_str());
                normalizer.normalize(&mut s)?;
                let normalized = s.get().to_string();
                if normalized.is_empty() { text } else { normalized }
            };
            let next = key_to_new.len() as i64;
            *slot = *key_to_new.entry(key).or_insert(next);
        }
        Ok((lookup, key_to_new.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;
    use tidelm_core::engram_hash::{EngramLayout, NgramHashState};

    fn model_dir() -> Option<PathBuf> {
        let dir = std::env::var("TIDELM_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
        });
        dir.join("tokenizer.json").exists().then_some(dir)
    }

    fn truth() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../fixtures/engram_truth.json")).unwrap()
    }

    fn i64s(v: &serde_json::Value) -> Vec<i64> {
        v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
    }

    #[test]
    fn token_map_and_prompt_hashes_match_the_checkpoint() {
        let Some(dir) = model_dir() else {
            eprintln!("tokenizer not present; skipping");
            return;
        };
        let t = truth();
        let text = Text::load(&dir).unwrap();
        assert_eq!(text.vocab_len() as i64, t["vocab_len"].as_i64().unwrap());

        let ids: Vec<u32> = i64s(&t["prompt_ids"]).iter().map(|&v| v as u32).collect();
        let encoded = text
            .encode("The capital of France is Paris. The capital of Japan is")
            .unwrap();
        assert_eq!(encoded, ids, "tokenizer encoding diverged from the Python tokenizer");
        for (id, want) in t["decode_samples"].as_object().unwrap() {
            let id: u32 = id.parse().unwrap();
            assert_eq!(text.decode(&[id]).unwrap(), want.as_str().unwrap(), "decode({id})");
        }

        let (map, size) = text.compressed_token_map().unwrap();
        assert_eq!(size as i64, t["compressed_vocab"].as_i64().unwrap());
        let as_i32: Vec<u8> = map.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
        let sha = format!("{:x}", Sha256::digest(&as_i32));
        if sha != t["token_map_sha256"].as_str().unwrap() {
            // localize the first divergence before failing
            let head = i64s(&t["token_map_head"]);
            let first = head.iter().zip(&map).position(|(a, b)| a != b);
            panic!("token map sha256 {sha} differs; first head divergence at {first:?}");
        }

        let layout = EngramLayout::new(&[1, 14], 4, 8, 16_000_000, size as i64);
        let mut state = NgramHashState::new(layout, map, 2);
        assert_eq!(state.pad_id(), t["pad_id"].as_i64().unwrap());
        let got = state.hash(&ids, 0, None);
        let want: Vec<i64> = t["hashes"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|pos| pos.as_array().unwrap().iter().flat_map(i64s))
            .collect();
        assert_eq!(got, want, "prompt hashes");

        // the incremental call continues from the cache, as the decode loop does
        let cont: Vec<u32> = i64s(&t["cont_ids"]).iter().map(|&v| v as u32).collect();
        let got = state.hash(&cont, ids.len(), None);
        let want: Vec<i64> = t["cont_hashes"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|pos| pos.as_array().unwrap().iter().flat_map(i64s))
            .collect();
        assert_eq!(got, want, "continuation hashes");
    }
}
