//! Routed experts: the per-rank EXL3 pack (tools/exl3_ref.py's PackReader, written by
//! tools/glm_pack_exl3.py) and the arena of resident slots the EXL3 kernels index
//! (tools/exl3_moe.py's Exl3Arena): structure of arrays, trellis reached through per-slot pointer
//! tables and per-slot widths, scales dense [slots][len] (the kernels index `base + e * len`).
//! Unloaded slots have zero scales, so they decode to an exact 0.
//!
//! A pack may give every expert its own gate / up / down widths (version 2, Mia's AI Lab's
//! 2-4-bit checkpoint). The arena then holds one class of slots for each width triple, each class
//! its own trellis pools; an expert only ever takes a slot of its own class (adapt.rs), so a slot
//! never changes size.

use std::collections::HashMap;
use std::os::raw::c_void;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_model::direct_io::{AlignedBuf, DirectFile, ALIGN};

use crate::Result;

/// The nine tensors of a record, in pack order: gate (1), up (3), down (2) trellis, then suh,
/// then svh (exl3_ref.RECORD_TENSORS).
pub const RECORD_TENSORS: [&str; 9] = ["t1", "t3", "t2", "suh1", "suh3", "suh2", "svh1", "svh3", "svh2"];

/// Gate, up and down widths in bits (each 2..=8).
pub type Bits = [usize; 3];

/// One record's tensor sizes in bytes for this rank of `world` at widths `bits`, with the model's
/// hidden and intermediate sizes: gate/up keep half the intermediate's N tiles, down half the
/// hidden's; suh whole.
pub fn record_sizes(bits: Bits, world: usize, hidden: usize, inter: usize) -> [usize; 9] {
    let il = inter / world;
    let hl = hidden / world;
    let tile = |b: usize| 16 * 16 * b / 8; // bytes a 16x16 tile
    [
        (hidden / 16) * (il / 16) * tile(bits[0]),
        (hidden / 16) * (il / 16) * tile(bits[1]),
        (inter / 16) * (hl / 16) * tile(bits[2]),
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
    pub bits: Bits,
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
    /// the trellis codebook, as the kernels take it (exl3::CB_MUL1 / CB_MCG)
    pub cb: i32,
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
        let cb = match (field("format")?.as_str(), field("codebook")?.as_str()) {
            (Some("exl3-experts"), Some("mul1")) => tidelm_kernels::exl3::CB_MUL1,
            (Some("exl3-experts"), Some("mcg")) => tidelm_kernels::exl3::CB_MCG,
            _ => return Err(format!("{}: not a mul1 / mcg EXL3 expert pack", path.display()).into()),
        };
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
            // one width for the record (version 1), or [gate, up, down] (version 2)
            let width = |x: &serde_json::Value| -> Result<usize> {
                let b = x.as_f64().ok_or("pack: bits")?;
                if b.fract() != 0.0 || !(2.0..=8.0).contains(&b) {
                    return Err(format!("{}: record {k}: {b}-bit trellis (whole 2-8 bits only)", path.display()).into());
                }
                Ok(b as usize)
            };
            let bits = match v.get("bits").ok_or("pack: bits")? {
                serde_json::Value::Array(a) if a.len() == 3 => [width(&a[0])?, width(&a[1])?, width(&a[2])?],
                x => [width(x)?; 3],
            };
            let r = Record {
                off: v.get("off").and_then(|x| x.as_u64()).ok_or("pack: off")?,
                nbytes: v.get("nbytes").and_then(|x| x.as_u64()).ok_or("pack: nbytes")? as usize,
                bits,
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
            cb,
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

/// One class of slots: experts of one width triple, its own trellis pools [n][bytes].
pub struct Class {
    pub bits: Bits,
    pub first: usize,
    pub n: usize,
    pub sizes: [usize; 9],
    trellis: Vec<DeviceBuffer<u8>>,
}

/// Resident expert slots: per class the trellis pools (gate, up, down), and over every slot the
/// scale pools, the trellis pointer tables and the widths (2 x bits a matrix) the kernels read.
pub struct Arena {
    pub slots: usize,
    /// the trellis codebook (exl3::CB_MUL1 / CB_MCG)
    pub cb: i32,
    pub classes: Vec<Class>,
    /// per slot: its class
    pub class_of: Vec<usize>,
    /// the scale pools [slots][bytes]: suh1, suh3, suh2, svh1, svh3, svh2 (the same at any width)
    scales: Vec<DeviceBuffer<u8>>,
    /// per-slot trellis addresses (t1, t3, t2) and 2 * bits of gate, up and down
    pub t1p: DeviceBuffer<i64>,
    pub t3p: DeviceBuffer<i64>,
    pub t2p: DeviceBuffer<i64>,
    pub k2g: DeviceBuffer<i32>,
    pub k2u: DeviceBuffer<i32>,
    pub k2d: DeviceBuffer<i32>,
    /// the widest 2 * bits any slot holds
    pub max_k2: i32,
    /// prefill runs the smem tile (the decoder's arena), not the grouped GEMV (the MTP layer's)
    pub prefill_tiled: bool,
    pub hidden: usize,
    pub inter_l: usize,
    pub down_n: usize,
}

impl Arena {
    /// `classes`: (widths, slots) in slot order; `cb` the codebook.
    pub fn new(classes: &[(Bits, usize)], cb: i32, world: usize, hidden: usize, inter: usize) -> Result<Self> {
        let slots: usize = classes.iter().map(|c| c.1).sum();
        let mut cls = Vec::with_capacity(classes.len());
        let mut class_of = Vec::with_capacity(slots);
        let (mut t1, mut t3, mut t2, mut k2) = (vec![], vec![], vec![], [vec![], vec![], vec![]]);
        for (c, &(bits, n)) in classes.iter().enumerate() {
            let sizes = record_sizes(bits, world, hidden, inter);
            // trellis left uninitialized: a slot is read only once loaded
            let trellis = (0..3).map(|i| DeviceBuffer::<u8>::new((n * sizes[i]).max(1))).collect::<std::result::Result<Vec<_>, _>>()?;
            for s in 0..n {
                class_of.push(c);
                t1.push(trellis[0].raw() as i64 + (s * sizes[0]) as i64);
                t3.push(trellis[1].raw() as i64 + (s * sizes[1]) as i64);
                t2.push(trellis[2].raw() as i64 + (s * sizes[2]) as i64);
                for i in 0..3 {
                    k2[i].push(2 * bits[i] as i32);
                }
            }
            cls.push(Class { bits, first: class_of.len() - n, n, sizes, trellis });
        }
        let ssz = record_sizes([2; 3], world, hidden, inter);
        let scales = (3..9).map(|i| DeviceBuffer::<u8>::zeroed(slots * ssz[i])).collect::<std::result::Result<Vec<_>, _>>()?;
        let max_k2 = k2.iter().flatten().copied().max().unwrap_or(0);
        Ok(Arena {
            slots,
            cb,
            classes: cls,
            class_of,
            scales,
            t1p: DeviceBuffer::from_slice(&t1)?,
            t3p: DeviceBuffer::from_slice(&t3)?,
            t2p: DeviceBuffer::from_slice(&t2)?,
            k2g: DeviceBuffer::from_slice(&k2[0])?,
            k2u: DeviceBuffer::from_slice(&k2[1])?,
            k2d: DeviceBuffer::from_slice(&k2[2])?,
            max_k2,
            prefill_tiled: false,
            hidden,
            inter_l: inter / world,
            down_n: hidden / world,
        })
    }

    /// Every slot at one width (the MTP layer's arena, a version-1 pack's).
    pub fn uniform(slots: usize, bits: Bits, cb: i32, world: usize, hidden: usize, inter: usize) -> Result<Self> {
        Self::new(&[(bits, slots)], cb, world, hidden, inter)
    }

    /// Slot `slot`'s record tensor sizes.
    pub fn sizes(&self, slot: usize) -> [usize; 9] {
        self.classes[self.class_of[slot]].sizes
    }

    pub fn bits(&self, slot: usize) -> Bits {
        self.classes[self.class_of[slot]].bits
    }

    /// Bytes of every slot.
    pub fn bytes(&self) -> usize {
        self.classes.iter().map(|c| c.n * c.sizes.iter().sum::<usize>()).sum()
    }

    /// Where slot `slot`'s record tensors go: (device address, bytes) in pack order.
    pub fn slot_dsts(&self, slot: usize) -> [(usize, usize); 9] {
        let c = &self.classes[self.class_of[slot]];
        let i = slot - c.first;
        std::array::from_fn(|t| {
            let p = if t < 3 { c.trellis[t].ptr_at(i * c.sizes[t]) } else { self.scales[t - 3].ptr_at(slot * c.sizes[t]) };
            (p as usize, c.sizes[t])
        })
    }

    /// Copy a record (host, in pack order, of the slot's widths) into `slot` on `st` (the host
    /// buffer must outlive the copy: callers synchronize before reusing it).
    pub fn load(&self, slot: usize, rec: &[u8], st: &Stream) -> Result<()> {
        if slot >= self.slots {
            return Err(format!("arena: slot {slot} of {}", self.slots).into());
        }
        let dsts = self.slot_dsts(slot);
        if rec.len() < dsts.iter().map(|d| d.1).sum::<usize>() {
            return Err(format!("arena: a record of {} bytes for slot {slot} of {:?}", rec.len(), self.bits(slot)).into());
        }
        let mut off = 0;
        for (dst, b) in dsts {
            unsafe {
                tidelm_cuda::memcpy_async(dst as *mut c_void, rec[off..].as_ptr() as *const c_void, b,
                                          tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
            }
            off += b;
        }
        Ok(())
    }

    /// Device copy of slot `src`'s record, of `dst`'s widths, over slot `dst` (a kept hot read
    /// leaving its transient slot, which is as wide as any class).
    pub fn copy_slot(&self, dst: usize, src: usize, st: &Stream) -> Result<()> {
        let (d, s) = (self.slot_dsts(dst), self.slot_dsts(src));
        for t in 0..9 {
            if s[t].1 < d[t].1 {
                return Err(format!("arena: slot {src} cannot hold slot {dst}'s record").into());
            }
            unsafe {
                tidelm_cuda::memcpy_async(d[t].0 as *mut c_void, s[t].0 as *const c_void, d[t].1,
                                          tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, st)?;
            }
        }
        Ok(())
    }

    /// Slot `slot`'s widths for the kernels (the transient slot takes each read's).
    pub fn set_widths(&self, slot: usize, bits: Bits, st: &Stream) -> Result<()> {
        for (k, b) in [(&self.k2g, bits[0]), (&self.k2u, bits[1]), (&self.k2d, bits[2])] {
            k.upload_async(slot, &[2 * b as i32], st)?;
        }
        Ok(())
    }

    /// A scale pool (record tensors 3..9: suh1, suh3, suh2, svh1, svh3, svh2).
    pub fn pool(&self, i: usize) -> *mut c_void {
        self.scales[i - 3].raw()
    }
}
