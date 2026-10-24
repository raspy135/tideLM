//! Lossless random-access UE8M0 scales: adjacent exponents plus exact exceptions.
//! From `engine/scale_layout.py` (`tidelm-expert-scales-v1`).
//!
//! Every 64-byte block has a base, 64 code bits, 64 exception bits and an exception
//! offset. Fixed slot strides use the largest exception payload in a checkpoint pack,
//! so every audited expert remains replaceable without allocation in graphs.
//!
//! Byte-for-byte compatible with the Python packer: the same plane encodes to the same
//! record, and either side decodes the other's output.

pub const FORMAT: &str = "tidelm-expert-scales-v1";
pub const BLOCK: usize = 64;

#[derive(Debug, PartialEq, Eq)]
pub struct ScaleError(pub &'static str);

impl std::fmt::Display for ScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ScaleError {}

pub fn align(value: usize, boundary: usize) -> usize {
    value.div_ceil(boundary) * boundary
}

/// Field offsets of one packed record, all in bytes. Identical to the Python dict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub size: usize,
    pub capacity: usize,
    pub blocks: usize,
    pub codes: usize,
    pub escaped: usize,
    pub offsets: usize,
    pub payload: usize,
    pub stride: usize,
}

pub fn layout(size: usize, exceptions: usize) -> Result<Layout, ScaleError> {
    if size < 1 || exceptions > size {
        return Err(ScaleError("invalid scale plane size or exception capacity"));
    }
    let blocks = size.div_ceil(BLOCK);
    let codes = align(blocks, 8);
    let escaped = codes + blocks * 8;
    let offsets = escaped + blocks * 8;
    let payload = align(offsets + blocks * 4, 8);
    Ok(Layout {
        size,
        capacity: exceptions,
        blocks,
        codes,
        escaped,
        offsets,
        payload,
        stride: align(payload + exceptions, 256),
    })
}

/// One encoded scale plane before packing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub size: usize,
    pub bases: Vec<u8>,
    /// Per block, bit j set when `value[j] == base + 1` (little bit order, as u64 LE).
    pub codes: Vec<u64>,
    /// Per block, bit j set when `value[j]` is an exact exception.
    pub escaped: Vec<u64>,
    /// Exclusive prefix sum of per-block exception counts.
    pub offsets: Vec<i32>,
    pub exceptions: Vec<u8>,
}

impl Plane {
    /// Pack into a fixed-stride record. From `scale_layout.Plane.pack`.
    pub fn pack(&self, capacity: usize) -> Result<Vec<u8>, ScaleError> {
        if self.exceptions.len() > capacity {
            return Err(ScaleError("scale exceptions exceed the audited slot capacity"));
        }
        let spec = layout(self.size, capacity)?;
        let mut out = vec![0u8; spec.stride];
        out[..spec.blocks].copy_from_slice(&self.bases);
        for (dst, words) in [(spec.codes, &self.codes), (spec.escaped, &self.escaped)] {
            for (i, w) in words.iter().enumerate() {
                out[dst + i * 8..dst + i * 8 + 8].copy_from_slice(&w.to_le_bytes());
            }
        }
        for (i, o) in self.offsets.iter().enumerate() {
            out[spec.offsets + i * 4..spec.offsets + i * 4 + 4].copy_from_slice(&o.to_le_bytes());
        }
        out[spec.payload..spec.payload + self.exceptions.len()].copy_from_slice(&self.exceptions);
        Ok(out)
    }
}

/// Encode a native UE8M0 byte plane. From `scale_layout.encode`.
///
/// Per 64-byte block the base is the byte value b maximizing `hist[b] + hist[b+1]`
/// (first maximum on ties), so values equal to the base cost nothing and values equal
/// to base+1 cost one code bit; everything else is an exact exception byte.
pub fn encode(source: &[u8]) -> Result<Plane, ScaleError> {
    if source.is_empty() {
        return Err(ScaleError("expected a nonempty native UE8M0 byte plane"));
    }
    let blocks = source.len().div_ceil(BLOCK);
    let mut bases = Vec::with_capacity(blocks);
    let mut codes = Vec::with_capacity(blocks);
    let mut escaped = Vec::with_capacity(blocks);
    let mut counts = Vec::with_capacity(blocks);
    let mut exceptions = Vec::new();
    for b in 0..blocks {
        let lo = b * BLOCK;
        let hi = (lo + BLOCK).min(source.len());
        let block = &source[lo..hi];
        let mut hist = [0u32; 256];
        for &v in block {
            hist[v as usize] += 1;
        }
        let mut base = 0usize;
        let mut best = 0u32;
        for v in 0..255 {
            let pair = hist[v] + hist[v + 1];
            if pair > best {
                best = pair;
                base = v;
            }
        }
        let base = base as u8;
        let mut code_bits = 0u64;
        let mut escape_bits = 0u64;
        let mut n_escaped = 0i32;
        for (j, &v) in block.iter().enumerate() {
            if v == base.wrapping_add(1) && base != 255 {
                code_bits |= 1 << j;
            } else if v != base {
                escape_bits |= 1 << j;
                exceptions.push(v);
                n_escaped += 1;
            }
        }
        bases.push(base);
        codes.push(code_bits);
        escaped.push(escape_bits);
        counts.push(n_escaped);
    }
    let mut offsets = Vec::with_capacity(blocks);
    let mut acc = 0i32;
    for c in counts {
        offsets.push(acc);
        acc += c;
    }
    Ok(Plane { size: source.len(), bases, codes, escaped, offsets, exceptions })
}

/// Decode a packed record. From `scale_layout.decode`, including every validation:
/// a record whose bitmaps, bases or offsets are inconsistent is refused, never
/// best-effort read — exceptions and padding must not be able to address other records.
pub fn decode(record: &[u8], spec: &Layout) -> Result<Vec<u8>, ScaleError> {
    if *spec != layout(spec.size, spec.capacity)? {
        return Err(ScaleError("invalid scale layout"));
    }
    if record.len() != spec.stride {
        return Err(ScaleError("invalid packed scale record"));
    }
    let blocks = spec.blocks;
    let word = |base: usize, i: usize| -> u64 {
        u64::from_le_bytes(record[base + i * 8..base + i * 8 + 8].try_into().unwrap())
    };
    let mut counts = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let escape_bits = word(spec.escaped, b);
        let valid = spec.size.saturating_sub(b * BLOCK).min(BLOCK);
        if valid < BLOCK && (escape_bits >> valid) != 0 {
            return Err(ScaleError("invalid scale bitmap or base"));
        }
        if record[b] == 255 {
            return Err(ScaleError("invalid scale bitmap or base"));
        }
        counts.push(escape_bits.count_ones() as i64);
    }
    let mut acc = 0i64;
    let mut total = 0i64;
    for (b, &c) in counts.iter().enumerate() {
        let got = i32::from_le_bytes(
            record[spec.offsets + b * 4..spec.offsets + b * 4 + 4].try_into().unwrap(),
        ) as i64;
        if got != acc {
            return Err(ScaleError("invalid scale exception offsets"));
        }
        acc += c;
        total += c;
    }
    if total as usize > spec.capacity {
        return Err(ScaleError("invalid scale exception offsets"));
    }
    let mut out = Vec::with_capacity(spec.size);
    let mut next_exception = spec.payload;
    for b in 0..blocks {
        let base = record[b];
        let code_bits = word(spec.codes, b);
        let escape_bits = word(spec.escaped, b);
        let valid = spec.size.saturating_sub(b * BLOCK).min(BLOCK);
        for j in 0..valid {
            let v = if escape_bits >> j & 1 == 1 {
                let v = record[next_exception];
                next_exception += 1;
                v
            } else {
                base + ((code_bits >> j & 1) as u8)
            };
            out.push(v);
        }
        // exceptions in padding positions were refused above, so no cursor drift here
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[u8]) {
        let plane = encode(values).unwrap();
        let spec = layout(values.len(), plane.exceptions.len()).unwrap();
        let packed = plane.pack(spec.capacity).unwrap();
        assert_eq!(decode(&packed, &spec).unwrap(), values, "len {}", values.len());
    }

    // engine/test_scale_layout.py::test_every_byte_and_partial_blocks_roundtrip
    #[test]
    fn every_byte_and_partial_blocks_roundtrip() {
        roundtrip(&(0..=255u8).collect::<Vec<_>>());
        roundtrip(&[0, 255, 127]);
        roundtrip(&[0u8; 129]);
        roundtrip(&vec![255u8; 257]);
        // arbitrary bytes, deterministic: a simple LCG stands in for np.random
        let mut state = 93u64;
        let noise: Vec<u8> = (0..13 * 9)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();
        roundtrip(&noise);
    }

    // engine/test_scale_layout.py::test_exceptions_and_padding_cannot_address_other_records
    #[test]
    fn exceptions_and_padding_cannot_address_other_records() {
        let values: Vec<u8> = (0..192u8).collect();
        let plane = encode(&values).unwrap();
        let count = plane.exceptions.len();
        assert!(count > 0);
        assert_eq!(plane.pack(count - 1).unwrap_err().0, "scale exceptions exceed the audited slot capacity");
        let spec = layout(plane.size, count).unwrap();
        let mut packed = plane.pack(count).unwrap();
        for b in packed[spec.offsets..spec.offsets + 4].iter_mut() {
            *b = 255;
        }
        assert!(decode(&packed, &spec).is_err());
    }

    #[test]
    fn mostly_uniform_plane_stores_few_exceptions() {
        // base+code coverage: a plane of {7, 8} should need no exception bytes at all
        let values: Vec<u8> = (0..128).map(|i| if i % 3 == 0 { 8 } else { 7 }).collect();
        let plane = encode(&values).unwrap();
        assert!(plane.exceptions.is_empty());
        roundtrip(&values);
    }
}
