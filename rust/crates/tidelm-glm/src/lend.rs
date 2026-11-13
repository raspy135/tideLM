//! Lending expert slots: the arena's memory, while an image is encoded, holds the vision tower
//! instead of some of the least-wanted experts (GLM-5.3-Flash's vision is loaded on demand, so a
//! text-only server keeps all of its memory for experts). Nothing is allocated: the lent slots'
//! owners leave routing (as a swap's victims do), their trellis ranges are handed out, and on
//! return the slots are refilled from the pack - with the same experts, or where adaptation has
//! placed one elsewhere meanwhile, with the class's next most wanted non-resident one.
//!
//! The lent slots are the last ones of the class with the most slots. A swap or a kept hot read only
//! ever takes a resident's slot, and a lent slot has none, so nothing writes into it while lent.

use std::os::raw::c_void;

use crate::forward::{load_slots, Engine};
use crate::Result;

/// Slots lent out: their class, range and former owners, and the memory they give.
pub struct Lent {
    pub class: usize,
    pub first: usize,
    pub n: usize,
    owners: Vec<Option<(usize, usize)>>,
    /// the class's gate, up and down trellis ranges of the lent slots
    pub ranges: [(*mut c_void, usize); 3],
}

impl Lent {
    pub fn bytes(&self) -> usize {
        self.ranges.iter().map(|r| r.1).sum()
    }
}

impl Engine {
    /// Lend at least `bytes` of expert slots (one rank: the pair would have to agree on them).
    pub fn lend_slots(&mut self, bytes: usize) -> Result<Lent> {
        if self.world != 1 {
            return Err("lending slots on a pair is not implemented".into());
        }
        let (c, cl) = self.arena.classes.iter().enumerate().max_by_key(|(_, c)| c.n).ok_or("arena: no classes")?;
        let per: usize = cl.sizes[..3].iter().sum();
        let n = bytes.div_ceil(per);
        // the class's last slots, but not all of them: a class keeps most of its residents
        if n * 4 > cl.n {
            return Err(format!("lend {bytes} bytes: {n} slots of class {c}'s {}", cl.n).into());
        }
        let first = cl.first + cl.n - n;
        let e = self.cfg.experts;
        let mut owners = Vec::with_capacity(n);
        for s in first..first + n {
            let o = self.res.owner[s].take();
            if let Some((l, x)) = o {
                self.lut_h[l * e + x] = -1;
                self.allowed_h[l * e + x] = 0;
            }
            owners.push(o);
        }
        self.lut_upload()?;
        self.st.synchronize()?;
        let ranges = self.arena.class_ranges(c, first, n)?;
        Ok(Lent { class: c, first, n, owners, ranges })
    }

    /// Refill lent slots from the pack: their former experts, or for one adaptation re-placed
    /// meanwhile, the class's most wanted non-resident expert (the routing history's mass).
    pub fn return_slots(&mut self, lent: Lent) -> Result<()> {
        let e = self.cfg.experts;
        let bits = self.arena.classes[lent.class].bits;
        let mass = self.res.db_mass.clone().unwrap_or_default();
        let mut taken = std::collections::HashSet::new();
        let mut spare: Vec<(usize, usize)> = vec![];
        if !mass.is_empty() {
            let mut cand: Vec<(f64, usize, usize)> = vec![];
            for &l in &self.res.swap_layers {
                for x in 0..e {
                    if self.lut_h[l * e + x] < 0 && self.pack.record(l, x).map(|r| r.bits == bits).unwrap_or(false) {
                        cand.push((mass[l * e + x], l, x));
                    }
                }
            }
            cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            spare = cand.into_iter().map(|(_, l, x)| (l, x)).collect();
        }
        let mut spare = spare.into_iter();
        let mut items = vec![];
        for (k, o) in lent.owners.iter().enumerate() {
            let slot = lent.first + k;
            let pick = match o {
                Some((l, x)) if self.lut_h[l * e + x] < 0 && taken.insert((*l, *x)) => Some((*l, *x)),
                _ => spare.by_ref().find(|&(l, x)| self.lut_h[l * e + x] < 0 && taken.insert((l, x))),
            };
            if let Some((l, x)) = pick {
                items.push((slot, l, x));
            }
        }
        load_slots(&self.pack, &self.arena, &items)?;
        for &(slot, l, x) in &items {
            self.lut_h[l * e + x] = slot as i32;
            self.allowed_h[l * e + x] = 1;
            self.res.owner[slot] = Some((l, x));
        }
        self.lut_upload()?;
        self.st.synchronize()?;
        Ok(())
    }
}
