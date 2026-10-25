//! NumPy `.npz` files as `np.savez` writes them (a zip of `.npy` members, stored, no
//! compression): the Python engine's expert demand database (`DSV41_PRUNE_DB`, written by
//! `v41_engine.save_prune_db`) and seed profiles, read and written in place so both engines
//! keep one history. Little-endian numeric arrays only; a compressed member
//! (`np.savez_compressed`) or a big-endian / object dtype is refused, never guessed at.

use std::collections::BTreeMap;

#[derive(Debug)]
pub struct NpzError(pub String);

impl std::fmt::Display for NpzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NpzError {}

fn err<T>(msg: impl Into<String>) -> Result<T, NpzError> {
    Err(NpzError(msg.into()))
}

/// One array: its npy dtype (`<f8`, `<i8`, `<f4`, `<i4`, `|u1`, `|b1`), shape and raw
/// little-endian bytes in C order.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Array {
    pub fn f64s(shape: Vec<usize>, v: &[f64]) -> Self {
        Array { descr: "<f8".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }

    pub fn i64s(shape: Vec<usize>, v: &[i64]) -> Self {
        Array { descr: "<i8".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values as f64 (from `<f8`, `<f4`, `<i8`, `<i4`).
    pub fn to_f64(&self) -> Result<Vec<f64>, NpzError> {
        let d = &self.data;
        Ok(match self.descr.as_str() {
            "<f8" => d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
            "<f4" => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "<i8" => d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "<i4" => d.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            other => return err(format!("npz: dtype {other} is not numeric")),
        })
    }

    /// The values as i64 (from `<i8`, `<i4`, `|u1`, `|b1`).
    pub fn to_i64(&self) -> Result<Vec<i64>, NpzError> {
        let d = &self.data;
        Ok(match self.descr.as_str() {
            "<i8" => d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "<i4" => d.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            "|u1" | "|b1" => d.iter().map(|&b| b as i64).collect(),
            other => return err(format!("npz: dtype {other} is not an integer")),
        })
    }

    fn itemsize(descr: &str) -> Result<usize, NpzError> {
        match descr {
            "<f8" | "<i8" => Ok(8),
            "<f4" | "<i4" => Ok(4),
            "|u1" | "|b1" => Ok(1),
            other => err(format!("npz: unsupported dtype {other}")),
        }
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xffff_ffff
}

/// An `.npy` member: magic, version 1-3, header dict, data.
fn parse_npy(name: &str, b: &[u8]) -> Result<Array, NpzError> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return err(format!("npz: {name} is not an npy array"));
    }
    let (hlen, at) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 if b.len() >= 12 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => return err(format!("npz: {name}: npy version {v}")),
    };
    let header = std::str::from_utf8(b.get(at..at + hlen).ok_or_else(|| NpzError(format!("npz: {name}: short header")))?)
        .map_err(|_| NpzError(format!("npz: {name}: header is not text")))?;
    let field = |key: &str| -> Result<&str, NpzError> {
        let i = header.find(&format!("'{key}':")).ok_or_else(|| NpzError(format!("npz: {name}: no {key}")))?;
        Ok(header[i + key.len() + 3..].trim_start())
    };
    let descr = field("descr")?;
    let descr = descr.strip_prefix('\'').and_then(|d| d.split('\'').next()).ok_or_else(|| NpzError(format!("npz: {name}: descr")))?;
    if !field("fortran_order")?.starts_with("False") {
        return err(format!("npz: {name}: Fortran order"));
    }
    let shape = field("shape")?;
    let shape = shape.strip_prefix('(').and_then(|s| s.split(')').next()).ok_or_else(|| NpzError(format!("npz: {name}: shape")))?;
    let shape: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| NpzError(format!("npz: {name}: shape {shape}"))))
        .collect::<Result<_, _>>()?;
    let n: usize = shape.iter().product::<usize>() * Array::itemsize(descr)?;
    let data = b.get(at + hlen..at + hlen + n).ok_or_else(|| NpzError(format!("npz: {name}: {n} data bytes missing")))?;
    Ok(Array { descr: descr.to_string(), shape, data: data.to_vec() })
}

fn npy_bytes(a: &Array) -> Vec<u8> {
    let shape = match a.shape.len() {
        1 => format!("({},)", a.shape[0]),
        _ => format!("({})", a.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", a.descr, shape);
    // numpy pads the header with spaces to a multiple of 64 bytes, newline-terminated
    let total = (10 + header.len() + 1).div_ceil(64) * 64;
    header.push_str(&" ".repeat(total - 10 - header.len() - 1));
    header.push('\n');
    let mut out = Vec::with_capacity(total + a.data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&a.data);
    out
}

/// Every member of an uncompressed `.npz`, by name without the `.npy` suffix.
pub fn parse(bytes: &[u8]) -> Result<BTreeMap<String, Array>, NpzError> {
    // the central directory, from the end-of-central-directory record
    let eocd = (0..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| NpzError("npz: no end of central directory".into()))?;
    let u16at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;
    let u32at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
    let (n, mut at) = (u16at(eocd + 10), u32at(eocd + 16));
    let mut out = BTreeMap::new();
    for _ in 0..n {
        if at + 46 > bytes.len() || u32at(at) != 0x0201_4b50 {
            return err("npz: bad central directory entry");
        }
        let method = u16at(at + 10);
        let (csize, usize_) = (u32at(at + 20), u32at(at + 24));
        let (nlen, xlen, clen) = (u16at(at + 28), u16at(at + 30), u16at(at + 32));
        let local = u32at(at + 42);
        let name = String::from_utf8_lossy(&bytes[at + 46..at + 46 + nlen]).to_string();
        at += 46 + nlen + xlen + clen;
        if method != 0 || csize != usize_ {
            return err(format!("npz: {name} is compressed (method {method}); only np.savez files are read"));
        }
        if local + 30 > bytes.len() || u32at(local) != 0x0403_4b50 {
            return err(format!("npz: {name}: bad local header"));
        }
        let data_at = local + 30 + u16at(local + 26) + u16at(local + 28);
        let member = bytes.get(data_at..data_at + csize).ok_or_else(|| NpzError(format!("npz: {name}: truncated")))?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(key.clone(), parse_npy(&name, member)?);
    }
    Ok(out)
}

/// The `.npz` bytes of `arrays` (members `<name>.npy`, stored, in name order).
pub fn to_bytes(arrays: &BTreeMap<String, Array>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, a) in arrays {
        let fname = format!("{name}.npy");
        let data = npy_bytes(a);
        let crc = crc32(&data);
        let offset = out.len() as u32;
        let mut local = Vec::new();
        local.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        local.extend_from_slice(&20u16.to_le_bytes()); // version needed
        local.extend_from_slice(&0u16.to_le_bytes()); // flags
        local.extend_from_slice(&0u16.to_le_bytes()); // stored
        local.extend_from_slice(&0u16.to_le_bytes()); // time
        local.extend_from_slice(&0x21u16.to_le_bytes()); // date 1980-01-01
        local.extend_from_slice(&crc.to_le_bytes());
        local.extend_from_slice(&(data.len() as u32).to_le_bytes());
        local.extend_from_slice(&(data.len() as u32).to_le_bytes());
        local.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes());
        local.extend_from_slice(fname.as_bytes());
        out.extend_from_slice(&local);
        out.extend_from_slice(&data);
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // made by
        central.extend_from_slice(&local[4..30]); // version needed .. extra length
        central.extend_from_slice(&0u16.to_le_bytes()); // comment length
        central.extend_from_slice(&0u16.to_le_bytes()); // disk
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(fname.as_bytes());
    }
    let (cd_at, cd_len, n) = (out.len() as u32, central.len() as u32, arrays.len() as u16);
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&cd_len.to_le_bytes());
    out.extend_from_slice(&cd_at.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

pub fn read(path: &std::path::Path) -> Result<BTreeMap<String, Array>, NpzError> {
    let bytes = std::fs::read(path).map_err(|e| NpzError(format!("{}: {e}", path.display())))?;
    parse(&bytes).map_err(|e| NpzError(format!("{}: {}", path.display(), e.0)))
}

/// Written beside `path` and renamed over it: a reader never sees half a file
/// (`v41_engine.save_prune_db` does the same).
pub fn write(path: &std::path::Path, arrays: &BTreeMap<String, Array>) -> Result<(), NpzError> {
    let tmp = path.with_extension("npz.tmp");
    std::fs::write(&tmp, to_bytes(arrays)).map_err(|e| NpzError(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| NpzError(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_of_the_check_string() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn round_trips_the_demand_database_layout() {
        let mut m = BTreeMap::new();
        let counts: Vec<f64> = (0..40 * 384).map(|i| i as f64 * 0.25).collect();
        m.insert("counts".to_string(), Array::f64s(vec![40, 384], &counts));
        m.insert("unit".to_string(), Array::i64s(vec![1], &[1]));
        m.insert("score_history_version".to_string(), Array::i64s(vec![1], &[2]));
        let back = parse(&to_bytes(&m)).unwrap();
        assert_eq!(back, m);
        assert_eq!(back["counts"].to_f64().unwrap(), counts);
        assert_eq!(back["unit"].to_i64().unwrap(), vec![1]);
    }

    #[test]
    fn reads_numpys_own_header_layout() {
        // np.save(np.arange(3, dtype='<i8')) header, as numpy 1.x/2.x writes it
        let mut npy = b"\x93NUMPY\x01\x00".to_vec();
        let mut h = "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }".to_string();
        let total = (10 + h.len() + 1).div_ceil(64) * 64;
        h.push_str(&" ".repeat(total - 10 - h.len() - 1));
        h.push('\n');
        npy.extend_from_slice(&(h.len() as u16).to_le_bytes());
        npy.extend_from_slice(h.as_bytes());
        for v in [0i64, 1, 2] {
            npy.extend_from_slice(&v.to_le_bytes());
        }
        let a = parse_npy("x.npy", &npy).unwrap();
        assert_eq!((a.descr.as_str(), a.shape.clone(), a.to_i64().unwrap()), ("<i8", vec![3], vec![0, 1, 2]));
    }

    /// The Python server's live database, when this checkout has one: every member parses,
    /// and writing it back gives the same arrays.
    #[test]
    fn reads_the_python_engines_database() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../results/prune_demand_req_score_1017.npz");
        let Ok(bytes) = std::fs::read(&path) else { return };
        let m = parse(&bytes).unwrap();
        assert_eq!(m["counts"].shape, vec![40, 384]);
        assert_eq!(m["mass"].descr, "<f8");
        assert_eq!(m["score_history_version"].to_i64().unwrap(), vec![2]);
        assert_eq!(parse(&to_bytes(&m)).unwrap(), m);
    }

    #[test]
    fn refuses_a_compressed_member() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), Array::i64s(vec![1], &[7]));
        let mut b = to_bytes(&m);
        // flip the central directory's method to deflate
        let cd = b.windows(4).position(|w| w == [0x50, 0x4b, 0x01, 0x02]).unwrap();
        b[cd + 10] = 8;
        assert!(parse(&b).unwrap_err().0.contains("compressed"));
    }
}
