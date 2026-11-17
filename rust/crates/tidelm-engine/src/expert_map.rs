//! A moment of an engine's residency, for the server's /expert-map page (GET /v1/expert-map).
//!
//! Host data only, copied on the engine thread between its steps on rank 0: no device read, no
//! collective, nothing sent to rank 1 (whose residency is rank 0's, mirrored), so a watched engine
//! computes what an unwatched one does. The server asks for one only while a page polls it.

/// An expert's state in `ExpertMap::state`.
pub const ABSENT: u8 = 0;
pub const RESIDENT: u8 = 1;
/// DeepSeek: in the hot-read ring, routable until the ring wraps over it
pub const HOT: u8 = 2;
/// a read in flight (a hot read, or an adaptation's newcomer): not routable yet
pub const LOADING: u8 = 3;

pub struct ExpertMap {
    /// the model's layer index of each row: the layers with routed experts, in order
    pub layers: Vec<usize>,
    pub experts: usize,
    /// [row * experts + expert]
    pub state: Vec<u8>,
    /// the routing history's mass, [row * experts + expert], when the engine keeps one
    pub demand: Option<Vec<f64>>,
    /// the arena's slots, and those among them kept for reads on demand (DeepSeek's ring, GLM's
    /// transient slot)
    pub slots: usize,
    pub spare: usize,
    /// boxes: each holds its part of every resident expert
    pub world: usize,
    /// requests queued or running
    pub busy: bool,
    /// counters since boot, in display order
    pub counters: Vec<(&'static str, f64)>,
}

impl ExpertMap {
    /// Experts in each state, by the constants above.
    pub fn counts(&self) -> [usize; 4] {
        let mut n = [0; 4];
        for &s in &self.state {
            n[(s as usize).min(3)] += 1;
        }
        n
    }

    /// Each row's states as a string of digits.
    pub fn state_rows(&self) -> Vec<String> {
        self.state.chunks(self.experts.max(1)).map(|r| r.iter().map(|&s| (b'0' + s.min(3)) as char).collect()).collect()
    }

    /// Each row's demand as base-36 digits, 0 (none) to z (the row's most used expert): the square
    /// root of an expert's mass over its row's largest, so the long tail keeps some of the scale.
    pub fn demand_rows(&self) -> Option<Vec<String>> {
        let d = self.demand.as_ref()?;
        Some(d.chunks(self.experts.max(1)).map(|r| {
            let top = r.iter().copied().fold(0.0f64, f64::max);
            r.iter().map(|&m| {
                let v = if top > 0.0 && m > 0.0 { ((m / top).sqrt() * 35.0).round().clamp(1.0, 35.0) as u32 } else { 0 };
                char::from_digit(v, 36).unwrap()
            }).collect()
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_counts() {
        let m = ExpertMap { layers: vec![3, 5], experts: 4, state: vec![0, 1, 1, 3, 2, 0, 0, 1],
                            demand: Some(vec![0.0, 1.0, 4.0, 1e-9, 0.0, 0.0, 0.0, 0.0]), slots: 8, spare: 2, world: 2, busy: false,
                            counters: vec![] };
        assert_eq!(m.counts(), [3, 3, 1, 1]);
        assert_eq!(m.state_rows(), ["0113", "2001"]);
        // 1/4 of the top is half the scale; a trace of mass is never shown as none; an unused row is all 0
        assert_eq!(m.demand_rows().unwrap(), ["0iz1", "0000"]);
    }
}
