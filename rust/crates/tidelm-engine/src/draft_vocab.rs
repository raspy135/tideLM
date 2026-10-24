//! The drafter's listed vocabulary (engine/draft_vocab.py, DSV41_DRAFT_VOCAB): the draft head
//! and the markov bias head score only these ids instead of all 129,280. Both heads are reads
//! whose time is their row count, and the draft only proposes - a token it cannot propose costs
//! acceptance at that position, never a token of output (greedy verification accepts exactly the
//! target's argmax; sampled verification gives an unlisted token q = 0 and stays exact).
//!
//! The list is a ranking file (ids, most frequent first, `#` comments). The listed ids are its
//! first `2 x rows` distinct ids (then the lowest unlisted ones if it is shorter), sorted; a pair
//! splits them into two contiguous runs, rank 0 the lower. Contiguous runs keep every rank-0 id
//! below every rank-1 id, so "the lower column wins a tie" is "the lower id wins", as in a
//! full-vocabulary argmax. One box lists the same `2 x rows` ids as the pair, so the two draft
//! the same proposals.
//!
//! Both ranks must list the same ids or they emit different drafts with nothing in the log: the
//! list's digest is in the boot config.
//!
//! On by default with the repo's v1 ranking (engine/draft_vocab_ranking_v1.txt, built into the
//! binary): measured on the pair (tidelm-batch, a truncated backbone under the whole drafter),
//! a draft step takes 4.7 ms listed against 5.7 ms over all ids; Python's A/B over 24 prompts
//! found acceptance unchanged (-0.0013 +- 0.0013, docs/draft-vocab.md).

use crate::Result;

/// DSV41_DRAFT_VOCAB_ROWS' default: ids a rank of the pair (0.28% of the Python A/B set's
/// tokens unlisted)
pub const DEFAULT_ROWS: usize = 40960;
/// fp8 head scales are 32 x 32 blocks; a rank's rows stay whole blocks
const ROW_ALIGN: usize = 128;

/// The ranking file's ids, most frequent first, duplicates dropped. An id outside the
/// vocabulary is an error rather than skipped: a list built for another tokenizer must not load.
pub fn ranking(text: &str, vocab: usize) -> Result<Vec<u32>> {
    let mut seen = vec![false; vocab];
    let mut out = Vec::new();
    for line in text.lines() {
        for word in line.split('#').next().unwrap_or("").split_whitespace() {
            let i: usize = word.parse().map_err(|_| format!("draft vocabulary: '{word}' is not an id"))?;
            if i >= vocab {
                return Err(format!("draft vocabulary: id {i} outside the vocabulary (0..{})", vocab - 1).into());
            }
            if !std::mem::replace(&mut seen[i], true) {
                out.push(i as u32);
            }
        }
    }
    Ok(out)
}

/// The listed ids, ascending: the ranking's first `want` ids, then the lowest unlisted ones.
pub fn select(order: &[u32], want: usize, vocab: usize) -> Result<Vec<u32>> {
    if want > vocab {
        return Err(format!("{want} listed ids exceed the vocabulary ({vocab})").into());
    }
    let mut picked: Vec<u32> = order.iter().copied().take(want).collect();
    if picked.len() < want {
        let mut taken = vec![false; vocab];
        for &i in &picked {
            taken[i as usize] = true;
        }
        picked.extend((0..vocab as u32).filter(|&i| !taken[i as usize]).take(want - picked.len()));
    }
    picked.sort_unstable();
    Ok(picked)
}

/// The ranking the Python profile ships (DSV41_DRAFT_VOCAB=engine/draft_vocab_ranking_v1.txt).
const RANKING_V1: &str = include_str!("../../../../engine/draft_vocab_ranking_v1.txt");

/// TIDELM_DRAFT_VOCAB (unset: the built-in v1 ranking; `all`: no list; otherwise a ranking
/// file) and TIDELM_DRAFT_VOCAB_ROWS (ids a rank of a pair): the listed ids, or None for the
/// whole vocabulary.
pub fn from_env(vocab: usize) -> Result<Option<Vec<u32>>> {
    let path = std::env::var("TIDELM_DRAFT_VOCAB").unwrap_or_default();
    if path.trim() == "all" {
        return Ok(None);
    }
    let rows = match std::env::var("TIDELM_DRAFT_VOCAB_ROWS") {
        Ok(v) => v.trim().parse::<usize>().map_err(|_| format!("TIDELM_DRAFT_VOCAB_ROWS={v}: not a number"))?,
        Err(_) => DEFAULT_ROWS,
    };
    if rows == 0 || rows % ROW_ALIGN != 0 {
        return Err(format!("TIDELM_DRAFT_VOCAB_ROWS must be a positive multiple of {ROW_ALIGN}, got {rows}").into());
    }
    let text = if path.trim().is_empty() {
        RANKING_V1.to_string()
    } else {
        std::fs::read_to_string(path.trim()).map_err(|e| format!("TIDELM_DRAFT_VOCAB {path}: {e}"))?
    };
    Ok(Some(select(&ranking(&text, vocab)?, 2 * rows, vocab)?))
}

/// FNV-1a over the ids, for the boot config.
pub fn digest(ids: &[u32]) -> u64 {
    ids.iter().flat_map(|i| i.to_le_bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_dedupes_and_rejects_foreign_ids() {
        assert_eq!(ranking("# header\n5 3 5\n7 # tail 9\n", 10).unwrap(), vec![5, 3, 7]);
        assert!(ranking("12\n", 10).is_err());
        assert!(ranking("x\n", 10).is_err());
    }

    #[test]
    fn select_takes_the_top_then_the_lowest_unlisted_sorted() {
        assert_eq!(select(&[9, 2, 7], 2, 10).unwrap(), vec![2, 9]);
        assert_eq!(select(&[9, 2], 4, 10).unwrap(), vec![0, 1, 2, 9]);
        assert!(select(&[1], 11, 10).is_err());
    }
}
