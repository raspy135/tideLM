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

/// Why an expert came into memory (`MapEvent::kind`): what the page shows a read's cause with.
/// A read after a token picked the expert while it was not loaded (its share of that token's routing)
pub const EV_HOT: u8 = 1;
/// The same, read inline in the pass that missed it (GLM_HOT_RESCUE: the share passed the gate)
pub const EV_RESCUE: u8 = 2;
/// Brought in by the routing history's demand (an adaptation), over a less wanted resident
pub const EV_ADAPT: u8 = 3;
/// Loaded for a continuing conversation before its prompt (session preload)
pub const EV_PRELOAD: u8 = 4;
/// A read the predictor asked for (GLM_HOT_PREDICT)
pub const EV_PREDICT: u8 = 5;
/// An inline read used for its pass and not kept: no victim of its width was queued
pub const EV_DROPPED: u8 = 6;
/// The names the server sends beside the kinds, by kind
pub const EV_NAMES: [&str; 7] = ["", "hot read", "inline rescue", "adaptation", "preload", "predicted read", "rescue not kept"];

/// One residency event: (layer, expert) came in (or, EV_DROPPED, was used and let go), `share` the
/// missed pick's share of its token's routing weight (0 when no token caused it), `victim` the expert
/// whose slot it took. `at`: seconds since the Unix epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapEvent {
    pub seq: u64,
    pub at: f64,
    pub layer: u16,
    pub expert: u16,
    pub kind: u8,
    pub share: f32,
    pub victim: Option<(u16, u16)>,
}

/// The engine's recent residency events, for the page: a bounded ring, host bookkeeping only (it
/// decides nothing; both ranks may keep one, rank 0's is read).
#[derive(Default, Debug)]
pub struct EventLog {
    seq: u64,
    ring: std::collections::VecDeque<MapEvent>,
}

impl EventLog {
    pub const CAP: usize = 8192;

    pub fn push(&mut self, layer: usize, expert: usize, kind: u8, share: f32, victim: Option<(usize, usize)>) {
        self.seq += 1;
        if self.ring.len() == Self::CAP {
            self.ring.pop_front();
        }
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        self.ring.push_back(MapEvent { seq: self.seq, at, layer: layer as u16, expert: expert as u16, kind, share,
                                       victim: victim.map(|(l, x)| (l as u16, x as u16)) });
    }

    /// The events of the last `secs` seconds, oldest first, at most `max` (the newest kept).
    pub fn recent(&self, secs: f64, max: usize) -> Vec<MapEvent> {
        let Some(last) = self.ring.back() else { return vec![] };
        let from = last.at.max(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)) - secs;
        let mut v: Vec<MapEvent> = self.ring.iter().rev().take_while(|e| e.at >= from).take(max).copied().collect();
        v.reverse();
        v
    }
}

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
    /// the recent residency events, oldest first (empty for an engine that keeps none)
    pub events: Vec<MapEvent>,
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
                            counters: vec![], events: vec![] };
        assert_eq!(m.counts(), [3, 3, 1, 1]);
        assert_eq!(m.state_rows(), ["0113", "2001"]);
        // 1/4 of the top is half the scale; a trace of mass is never shown as none; an unused row is all 0
        assert_eq!(m.demand_rows().unwrap(), ["0iz1", "0000"]);
    }

    #[test]
    fn the_event_ring_keeps_the_newest() {
        let mut log = EventLog::default();
        assert!(log.recent(5.0, 10).is_empty());
        for i in 0..EventLog::CAP + 5 {
            log.push(3, i % 256, EV_HOT, 0.5, Some((4, 7)));
        }
        let r = log.recent(60.0, 3);
        assert_eq!(r.iter().map(|e| e.seq).collect::<Vec<_>>(), [EventLog::CAP as u64 + 3, EventLog::CAP as u64 + 4, EventLog::CAP as u64 + 5]);
        assert_eq!((r[2].layer, r[2].kind, r[2].victim), (3, EV_HOT, Some((4, 7))));
        assert_eq!(log.recent(60.0, usize::MAX).len(), EventLog::CAP);
    }
}
