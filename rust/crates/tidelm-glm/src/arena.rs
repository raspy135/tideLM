//! Routed experts: the per-rank EXL3 pack (tools/exl3_ref.py's PackReader, written by
//! tools/glm_pack_exl3.py) and the arena of resident slots the EXL3 kernels index
//! (tools/exl3_moe.py's Exl3Arena): structure of arrays, one device pool per record tensor,
//! trellis reached through per-slot pointer tables, scales dense [slots][len] (the kernels index
//! `base + e * len`). Unloaded slots have zero scales, so they decode to an exact 0.

use std::collections::HashMap;
use std::os::raw::c_void;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_model::direct_io::{AlignedBuf, DirectFile, ALIGN};

use crate::Result;

/// The nine tensors of a record, in pack order: gate (1), up (3), down (2) trellis, then suh,
/// then svh (exl3_ref.RECORD_TENSORS).
pub const RECORD_TENSORS: [&str; 9] = ["t1", "t3", "t2", "suh1", "suh3", "suh2", "svh1", "svh3", "svh2"];

/// One record's tensor sizes in bytes for this rank of `world` at `bits`, with the model's
/// hidden and intermediate sizes: gate/up keep half the intermediate's N tiles, down half the
/// hidden's; suh whole.
pub fn record_sizes(bits: usize, world: usize, hidden: usize, inter: usize) -> [usize; 9] {
    let il = inter / world;
    let hl = hidden / world;
    let tile = 16 * 16 * bits / 8; // bytes a 16x16 tile
    [
        (hidden / 16) * (il / 16) * tile,
        (hidden / 16) * (il / 16) * tile,
        (inter / 16) * (hl / 16) * tile,
        hidden * 2,
        hidden * 2,
        inter * 2,
        il * 2,
        il * 2,
        hl * 2,
    ]
}

pub fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

#[derive(Clone, Copy, Debug)]
pub struct Record {
    pub off: u64,
    pub nbytes: usize,
    pub bits: usize,
}

/// One rank's pack: the header, and O_DIRECT reads of whole records.
pub struct Pack {
    pub path: std::path::PathBuf,
    file: DirectFile,
    pub data0: u64,
    pub records: HashMap<(usize, usize), Record>,
    pub rank: usize,
    pub world: usize,
    pub source_sha256: String,
    pub header_sha256: String,
    pub max_record: usize,
}

impl Pack {
    pub fn open(path: &Path, hidden: usize, inter: usize) -> Result<Self> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        use std::io::Read;
        let mut n8 = [0u8; 8];
        f.read_exact(&mut n8)?;
        let n = u64::from_le_bytes(n8) as usize;
        let mut hb = vec![0u8; n];
        f.read_exact(&mut hb)?;
        let header: serde_json::Value = serde_json::from_slice(&hb).map_err(|e| format!("{}: header: {e}", path.display()))?;
        let field = |k: &str| header.get(k).ok_or_else(|| format!("{}: header has no {k}", path.display()));
        if field("format")?.as_str() != Some("exl3-experts") || field("codebook")?.as_str() != Some("mul1") {
            return Err(format!("{}: not a mul1 EXL3 expert pack", path.display()).into());
        }
        let dims: Vec<u64> = field("dims")?.as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).collect()).unwrap_or_default();
        if dims != [hidden as u64, inter as u64] {
            return Err(format!("{}: pack dims {dims:?}, model [{hidden}, {inter}]", path.display()).into());
        }
        let tensors: Vec<&str> = field("tensors")?.as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
        if tensors != RECORD_TENSORS {
            return Err(format!("{}: record tensors {tensors:?}", path.display()).into());
        }
        let rank = field("rank")?.as_u64().ok_or("pack: rank")? as usize;
        let world = field("world")?.as_u64().ok_or("pack: world")? as usize;
        let data0 = align_up(8 + n) as u64;
        let mut records = HashMap::new();
        let mut max_record = 0;
        for (k, v) in field("records")?.as_object().ok_or("pack: records")? {
            let (l, e) = k.split_once(',').ok_or("pack: record key")?;
            let r = Record {
                off: v.get("off").and_then(|x| x.as_u64()).ok_or("pack: off")?,
                nbytes: v.get("nbytes").and_then(|x| x.as_u64()).ok_or("pack: nbytes")? as usize,
                bits: v.get("bits").and_then(|x| x.as_f64()).ok_or("pack: bits")? as usize,
            };
            if r.off as usize % ALIGN != 0 {
                return Err(format!("{}: records not {ALIGN}-aligned", path.display()).into());
            }
            let want: usize = record_sizes(r.bits, world, hidden, inter).iter().sum();
            if want != r.nbytes {
                return Err(format!("{}: record {k} is {} bytes, the layout says {want}", path.display(), r.nbytes).into());
            }
            max_record = max_record.max(r.nbytes);
            records.insert((l.parse()?, e.parse()?), r);
        }
        use sha2::Digest;
        let header_sha256 = format!("{:x}", sha2::Sha256::digest(&hb));
        Ok(Pack {
            path: path.to_path_buf(),
            file: DirectFile::open(path)?,
            data0,
            records,
            rank,
            world,
            source_sha256: field("source_sha256")?.as_str().unwrap_or("").to_string(),
            header_sha256,
            max_record,
        })
    }

    pub fn record(&self, l: usize, e: usize) -> Result<Record> {
        self.records.get(&(l, e)).copied().ok_or_else(|| format!("pack: no record ({l}, {e})").into())
    }

    /// One O_DIRECT read of a whole record into `buf` (aligned, at least align_up(max_record)).
    pub fn read(&self, l: usize, e: usize, buf: &mut [u8]) -> Result<Record> {
        let r = self.record(l, e)?;
        let need = align_up(r.nbytes);
        self.file.pread_exact(&mut buf[..need], self.data0 + r.off, r.nbytes)?;
        Ok(r)
    }

    pub fn buffer(&self) -> Result<AlignedBuf> {
        Ok(AlignedBuf::new(align_up(self.max_record))?)
    }
}

/// Resident expert slots for one width: per record tensor a device pool [slots][bytes].
pub struct Arena {
    pub slots: usize,
    pub bits: usize,
    pub sizes: [usize; 9],
    pub pools: Vec<DeviceBuffer<u8>>,
    /// per-slot trellis addresses (t1, t3, t2) and 2 * bits
    pub t1p: DeviceBuffer<i64>,
    pub t3p: DeviceBuffer<i64>,
    pub t2p: DeviceBuffer<i64>,
    pub k2: DeviceBuffer<i32>,
    pub hidden: usize,
    pub inter_l: usize,
    pub down_n: usize,
}

impl Arena {
    pub fn new(slots: usize, bits: usize, world: usize, hidden: usize, inter: usize) -> Result<Self> {
        let sizes = record_sizes(bits, world, hidden, inter);
        let mut pools = Vec::with_capacity(9);
        for (i, &b) in sizes.iter().enumerate() {
            // trellis left uninitialized (a slot is read only once loaded); scales zeroed
            pools.push(if i < 3 { DeviceBuffer::<u8>::new(slots * b)? } else { DeviceBuffer::<u8>::zeroed(slots * b)? });
        }
        let ptrs = |i: usize| -> Result<DeviceBuffer<i64>> {
            let base = pools[i].raw() as i64;
            Ok(DeviceBuffer::from_slice(&(0..slots).map(|s| base + (s * sizes[i]) as i64).collect::<Vec<_>>())?)
        };
        let (t1p, t3p, t2p) = (ptrs(0)?, ptrs(1)?, ptrs(2)?);
        let k2 = DeviceBuffer::from_slice(&vec![2 * bits as i32; slots])?;
        Ok(Arena { slots, bits, sizes, pools, t1p, t3p, t2p, k2, hidden, inter_l: inter / world, down_n: hidden / world })
    }

    pub fn slot_bytes(&self) -> usize {
        self.sizes.iter().sum()
    }

    /// Copy a record (host, in pack order) into `slot` on `st` (the host buffer must outlive the
    /// copy: callers synchronize before reusing it).
    pub fn load(&self, slot: usize, rec: &[u8], st: &Stream) -> Result<()> {
        if slot >= self.slots {
            return Err(format!("arena: slot {slot} of {}", self.slots).into());
        }
        let mut off = 0;
        for (i, &b) in self.sizes.iter().enumerate() {
            unsafe {
                tidelm_cuda::memcpy_async(self.pools[i].ptr_at(slot * b), rec[off..].as_ptr() as *const c_void, b,
                                          tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
            }
            off += b;
        }
        Ok(())
    }

    /// Device copy of slot `src` over slot `dst` (a kept hot read leaving its transient slot).
    pub fn copy_slot(&self, dst: usize, src: usize, st: &Stream) -> Result<()> {
        for (i, &b) in self.sizes.iter().enumerate() {
            self.pools[i].copy_from(dst * b, &self.pools[i], src * b, b, st)?;
        }
        Ok(())
    }

    pub fn pool(&self, i: usize) -> *mut c_void {
        self.pools[i].raw()
    }
}
