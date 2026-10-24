//! Safetensors shard headers and expert run grouping, from `engine/experts.py`
//! (`ShardFile`). Only the header layout is parsed here — a u64 LE length, a JSON
//! object of `{name: {dtype, shape, data_offsets: [lo, hi]}}` — tensors are read
//! through `direct_io` by byte span.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::direct_io::DirectFile;

pub const EXPERT_NAMES: [&str; 6] =
    ["w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"];

/// One maximal contiguous file range covering some of an expert's tensors:
/// `members` are `(index into EXPERT_NAMES, offset inside the run, nbytes)`.
#[derive(Debug, Clone)]
pub struct Run {
    pub lo: u64,
    pub hi: u64,
    pub members: Vec<(usize, usize, usize)>,
}

/// One safetensors shard: header spans + an O_DIRECT fd.
pub struct Shard {
    pub path: PathBuf,
    pub base: u64,
    spans: HashMap<String, (u64, u64)>,
    pub file: DirectFile,
    runs: Mutex<HashMap<String, Vec<Run>>>,
}

impl Shard {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = DirectFile::open(path)?;
        let mut len8 = [0u8; 8];
        file.pread_buffered(&mut len8, 0)?;
        let n = u64::from_le_bytes(len8);
        let mut raw = vec![0u8; n as usize];
        file.pread_buffered(&mut raw, 8)?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let base = 8 + n;
        let mut spans = HashMap::with_capacity(header.len());
        for (name, meta) in header {
            if name == "__metadata__" {
                continue;
            }
            let offs = meta
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing data_offsets"))?;
            let lo = offs[0].as_u64().unwrap_or(0) + base;
            let hi = offs[1].as_u64().unwrap_or(0) + base;
            spans.insert(name, (lo, hi));
        }
        Ok(Shard { path: path.to_path_buf(), base, spans, file, runs: Mutex::new(HashMap::new()) })
    }

    /// Absolute byte span of one tensor.
    pub fn span(&self, name: &str) -> io::Result<(u64, u64)> {
        self.spans.get(name).copied().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("{name} not in {}", self.path.display()))
        })
    }

    /// The 6 tensors of one expert grouped into maximal contiguous file ranges.
    ///
    /// These shards store ALL the scale tensors near the front and all the weight
    /// tensors far behind them, but within each group the three tensors of an expert
    /// are adjacent. So an expert is exactly two runs — a ~1.1 MB scale run and a
    /// ~17.7 MB weight run — not six, and not one. Six separate O_DIRECT round trips
    /// collapsed the read rate from ~4.7 GB/s to ~0.6 GB/s when little else was in
    /// flight; the grouping is a measured requirement, not tidiness.
    pub fn expert_runs(&self, prefix: &str) -> io::Result<Vec<Run>> {
        if let Some(cached) = self.runs.lock().unwrap().get(prefix) {
            return Ok(cached.clone());
        }
        let mut spans: Vec<(usize, u64, u64)> = Vec::with_capacity(6);
        for (i, name) in EXPERT_NAMES.iter().enumerate() {
            let (lo, hi) = self.span(&format!("{prefix}{name}"))?;
            spans.push((i, lo, hi));
        }
        spans.sort_by_key(|&(_, lo, _)| lo);
        let mut runs: Vec<Run> = Vec::new();
        for (i, lo, hi) in spans {
            match runs.last_mut() {
                Some(run) if run.hi == lo => {
                    run.members.push((i, (lo - run.lo) as usize, (hi - lo) as usize));
                    run.hi = hi;
                }
                _ => runs.push(Run { lo, hi, members: vec![(i, 0, (hi - lo) as usize)] }),
            }
        }
        self.runs.lock().unwrap().insert(prefix.to_string(), runs.clone());
        Ok(runs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tiny_shard(dir: &Path) -> PathBuf {
        // two experts' tensors laid out scales-first, weights-behind, like the real shards
        let names = EXPERT_NAMES;
        let mut header = serde_json::Map::new();
        let mut off = 0u64;
        let mut order: Vec<(String, u64)> = Vec::new();
        for group in [["w1.scale", "w2.scale", "w3.scale"], ["w1.weight", "w2.weight", "w3.weight"]] {
            for e in 0..2 {
                for name in group {
                    let len = if name.ends_with("scale") { 64 } else { 256 };
                    order.push((format!("layers.0.ffn.experts.{e}.{name}"), len));
                    off += len;
                }
            }
        }
        let mut pos = 0u64;
        for (name, len) in &order {
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "U8", "shape": [len],
                    "data_offsets": [pos, pos + len]
                }),
            );
            pos += len;
        }
        let hdr = serde_json::to_vec(&header).unwrap();
        let path = dir.join("tiny.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hdr).unwrap();
        let data: Vec<u8> = (0..off).map(|i| (i % 251) as u8).collect();
        f.write_all(&data).unwrap();
        let _ = names;
        path
    }

    #[test]
    fn runs_group_adjacent_tensors() {
        let dir = std::env::temp_dir().join(format!("tidelm-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = tiny_shard(&dir);
        let shard = Shard::open(&path).unwrap();
        let runs = shard.expert_runs("layers.0.ffn.experts.0.").unwrap();
        // scales of expert 0 are adjacent (one run), weights of expert 0 adjacent (one run)
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].members.len(), 3);
        assert_eq!(runs[1].members.len(), 3);
        // members carry EXPERT_NAMES indices: scales are 1, 3, 5; weights 0, 2, 4
        let scale_ids: Vec<usize> = runs[0].members.iter().map(|m| m.0).collect();
        assert_eq!(scale_ids, vec![1, 3, 5]);
        let weight_ids: Vec<usize> = runs[1].members.iter().map(|m| m.0).collect();
        assert_eq!(weight_ids, vec![0, 2, 4]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
