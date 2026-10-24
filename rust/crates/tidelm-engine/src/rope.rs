//! RoPE tables, from tools/v41_ref.py::precompute_freqs_cis (YaRN when
//! original_seq_len > 0), in the same fp32 operation order as the torch code; the
//! scalar bounds are Python doubles there and f64 here.

/// `(cos, sin)` per (position, pair): `[max_pos][rd / 2]`, flattened.
pub fn table(
    rd: usize,
    max_pos: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<[f32; 2]> {
    let half = rd / 2;
    let mut freqs: Vec<f32> =
        (0..half).map(|i| 1.0f32 / base.powf((2 * i) as f32 / rd as f32)).collect();
    if original_seq_len > 0 {
        let corrected = |rotations: f64| {
            rd as f64 * (original_seq_len as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(rd as f64 - 1.0);
        let denom = (high - low).max(1e-3) as f32;
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / denom).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut out = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for &f in &freqs {
            let a = p as f32 * f;
            out.push([a.cos(), a.sin()]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_rope_matches_the_closed_form() {
        let t = table(64, 10, 0, 10000.0, 16.0, 32.0, 1.0);
        assert_eq!(t.len(), 10 * 32);
        assert_eq!(t[0], [1.0, 0.0]);
        let f1 = 1.0f32 / 10000f32.powf(2.0 / 64.0);
        assert!((t[3 * 32 + 1][0] - (3.0 * f1).cos()).abs() < 1e-7);
    }

    #[test]
    fn yarn_keeps_high_frequencies_and_scales_low_ones() {
        let plain = table(64, 2, 0, 160000.0, 16.0, 32.0, 1.0);
        let yarn = table(64, 2, 65536, 160000.0, 16.0, 32.0, 1.0);
        // pair 0 is the fastest rotation: below `low`, untouched by the ramp
        assert_eq!(plain[32], yarn[32]);
        // the slowest pair is fully interpolated: its angle is divided by the factor
        let a_plain = plain[32 + 31][1].asin();
        let a_yarn = yarn[32 + 31][1].asin();
        assert!((a_plain / a_yarn - 16.0).abs() < 1e-3, "{a_plain} / {a_yarn}");
    }
}
