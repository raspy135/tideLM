"""Offline packer: a GLM checkpoint's EXL3 routed experts -> one aligned file per rank (GLM_EXPERT_FORMAT=exl3).

Reads `<prefix>{L}.mlp.experts.{E}.{gate,up,down}_proj.{trellis,suh,svh,<codebook>}` (prefix
`model.layers.` for GLM-5.3, `model.language_model.layers.` for GLM-5.3-Flash) and writes each
expert's TP-output rank slice with tools/exl3_ref.py's pack writer -- the same record layout and
4096-aligned O_DIRECT records as the V4.1 pack, at the checkpoint's own dims (GLM-5.3: 6,144
hidden, 2,048 intermediate, 256 experts a layer; GLM-5.3-Flash: 4,096, 2,048, 288). gate/up keep
this rank's share of the intermediate columns, down this rank's share of the hidden columns, as
engine/glm/tp.py splits the MXFP4 experts; at --world 1 a record is the whole expert. Sources:

  --source DIR    a checkpoint on disk;
  --repo ID       a checkpoint on the Hub, read by HTTP range requests: one request an expert, no
                  shard lands on disk (a one-Spark setup needs the pack's bytes, not the 156 GB
                  checkpoint beside it). Same pack bytes and source identity as --source on a
                  download of the same revision. Stdlib and numpy only.
  --from-pack P   re-slice a --world 1 pack into one rank's pack of a larger world (the TP=2 packs
                  of GLM-5.3-Flash from its one-box pack, at disk speed, without the Hub).

Checkpoints packed:
  bertholomus/GLM-5.3-EXL3-3.0bpw: mul1, 3-bit routed experts on layers 3-77, 4-bit on the MTP
    layer 78 (a version-1 pack, one width a layer);
  Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold: mcg, each expert's gate / up / down at its own width
    of 2-4 bits (a version-2 pack, the widths in every record);
  Mia-AiLab/GLM-5.3-Flash-EXL3-3.5bpw: mcg, down 4-bit, gate / up 3- or 4-bit a layer, MTP layer
    45 at 4 (a version-2 pack).

    python tools/glm_pack_exl3.py --source /models-exl3 --out /packs/glm-exl3-experts-r0of2.bin --rank 0
    uv run --no-project --with numpy python tools/glm_pack_exl3.py --world 1 \\
        --repo Mia-AiLab/GLM-5.3-Flash-EXL3-3.5bpw --out /packs/glm-flash-35-r0of1.bin --resume --verify 64

Run it on each node for its own rank; nothing here runs in serving.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402

PARTS = (("1", "gate_proj"), ("3", "up_proj"), ("2", "down_proj"))
EXPERT_RE = re.compile(r"^(.*layers\.)(\d+)\.mlp\.experts\.(\d+)\.")
DTYPES = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "F32": np.float32, "BF16": np.uint16, "U8": np.uint8}


def check_quant(q: dict, where: str) -> str:
    codebook = q.get("codebook", "mul1")
    if str(q.get("quant_method", "")).lower() != "exl3" or codebook not in ("mul1", "mcg"):
        raise SystemExit(f"{where}: not an EXL3 mul1 / mcg checkpoint ({q})")
    return codebook


def expert_names(weight_map) -> tuple[str, list[int], int]:
    """(name prefix up to the layer number, layers with routed experts, experts a layer)."""

    prefixes, layers, experts = set(), set(), set()
    for k in weight_map:
        m = EXPERT_RE.match(k)
        if m:
            prefixes.add(m[1])
            layers.add(int(m[2]))
            experts.add(int(m[3]))
    if len(prefixes) != 1:
        raise SystemExit(f"routed expert names under {sorted(prefixes) or 'no'} prefix")
    return prefixes.pop(), sorted(layers), max(experts) + 1


def rank_slice(full: dict[str, np.ndarray], rank: int, world: int) -> dict[str, np.ndarray]:
    """One rank's record from a whole expert: N tiles [rank*NT/world, +NT/world) of each trellis and
    the matching svh columns; suh (the input side) whole."""

    if world == 1:
        return full
    out = {}
    for tag, _ in PARTS:
        t, svh = full["t" + tag], full["svh" + tag]
        nt, n = t.shape[1] // world, svh.shape[0] // world
        out["t" + tag] = np.ascontiguousarray(t[:, rank * nt:(rank + 1) * nt, :])
        out["suh" + tag] = full["suh" + tag]
        out["svh" + tag] = np.ascontiguousarray(svh[rank * n:(rank + 1) * n])
    return out


class GlmExl3Source:
    """A checkpoint on disk."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open  # only this source needs it
        self._safe_open = safe_open
        self.dir = model_dir
        self.where = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        raw = json.load(open(os.path.join(model_dir, "config.json")))
        cfg = raw.get("text_config", raw)
        q = dict(raw.get("quantization_config") or cfg.get("quantization_config") or {})
        qpath = os.path.join(model_dir, "quantization_config.json")   # Mia's: the codebook and widths
        if os.path.exists(qpath):
            q.update({k: v for k, v in json.load(open(qpath)).items() if k in ("quant_method", "codebook")})
        self.codebook = check_quant(q, model_dir)
        self.marker = {"mul1": R.MUL1_MUL, "mcg": R.MCG_MUL}[self.codebook]
        self.hidden, self.inter = int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
        self.prefix, self.layers, n = expert_names(self.map)
        self.n_experts = int(cfg["n_routed_experts"])
        assert n == self.n_experts, (n, self.n_experts)
        R.set_dims(self.hidden, self.inter)
        self.sha = R.manifest_sha256(model_dir)
        self._open = {}

    def _file(self, name):
        shard = self.map[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(os.path.join(self.dir, shard), framework="numpy")
        return self._open[shard]

    def shape(self, name):
        return tuple(self._file(name).get_slice(name).get_shape())

    def array(self, name):
        return np.ascontiguousarray(self._file(name).get_tensor(name))

    def expert_bits(self, L, E) -> tuple[float, float, float]:
        """(gate, up, down) widths, from the trellis shapes."""
        return tuple(R.bits_of_shape(self.shape(f"{self.prefix}{L}.mlp.experts.{E}.{p}.trellis")) for _, p in PARTS)

    def marker_ok(self, L, E) -> bool:
        return all((int(self.array(f"{self.prefix}{L}.mlp.experts.{E}.{p}.{self.codebook}").reshape(-1)[0]) & 0xFFFFFFFF)
                   == self.marker for _, p in PARTS)

    def full_record(self, L, E) -> dict[str, np.ndarray]:
        out = {}
        for tag, part in PARTS:
            p = f"{self.prefix}{L}.mlp.experts.{E}.{part}."
            for t in ("trellis", "suh", "svh"):
                out[("t" if t == "trellis" else t) + tag] = self.array(p + t)
        return out


class HubSource(GlmExl3Source):
    """A checkpoint on the Hub, read by HTTP range requests. The identity is manifest_sha256's over
    the same bytes (each shard's header, config.json, the index), so a pack from here and one from a
    download of the same revision are the same file."""

    def __init__(self, repo: str, revision: str):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.where = f"{repo}@{revision}"
        self._url: dict[str, str] = {}           # a shard's resolved (CDN) URL, skipping the redirect
        cfg_b = self.get_url(self.base + "config.json")
        idx_b = self.get_url(self.base + "model.safetensors.index.json")
        self.map = json.loads(idx_b)["weight_map"]
        raw = json.loads(cfg_b)
        cfg = raw.get("text_config", raw)
        q = dict(raw.get("quantization_config") or cfg.get("quantization_config") or {})
        if "codebook" not in q:                   # Mia's GLM-5.3 cards keep it in quantization_config.json
            q.update({k: v for k, v in json.loads(self.get_url(self.base + "quantization_config.json")).items()
                      if k in ("quant_method", "codebook")})
        self.codebook = check_quant(q, self.where)
        self.marker = {"mul1": R.MUL1_MUL, "mcg": R.MCG_MUL}[self.codebook]
        self.hidden, self.inter = int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
        self.prefix, self.layers, n = expert_names(self.map)
        self.n_experts = int(cfg["n_routed_experts"])
        assert n == self.n_experts, (n, self.n_experts)
        R.set_dims(self.hidden, self.inter)
        shards = sorted(set(self.map.values()))
        with cf.ThreadPoolExecutor(16) as ex:
            heads = dict(zip(shards, ex.map(self._header, shards)))
        self.headers = {s: json.loads(b) for s, b in heads.items()}
        self.data0 = {s: 8 + len(b) for s, b in heads.items()}
        h = hashlib.sha256()                      # = R.manifest_sha256 on a download
        for s in shards:
            h.update(s.encode())
            h.update(heads[s])
        for name, b in (("config.json", cfg_b), ("model.safetensors.index.json", idx_b)):
            h.update(name.encode())
            h.update(b)
        self.sha = h.hexdigest()

    @staticmethod
    def get_url(url: str, rng: tuple[int, int] | None = None, final: list | None = None) -> bytes:
        hd = {"User-Agent": "tidelm-glm-pack"}
        if os.environ.get("HF_TOKEN") and "huggingface.co/" in url:
            hd["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        if rng:
            hd["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=hd), timeout=300) as r:
            data = r.read()
            if final is not None:
                final.append(r.geturl())
        if rng and len(data) != rng[1] - rng[0]:
            raise OSError(f"short read {len(data)} of {rng[1] - rng[0]}")
        return data

    def fetch(self, shard: str, rng: tuple[int, int]) -> bytes:
        for attempt in range(8):
            url = self._url.get(shard)
            try:
                if url is None:
                    final = []
                    data = self.get_url(self.base + shard, rng, final)
                    self._url[shard] = final[0]
                    return data
                return self.get_url(url, rng)
            except (OSError, urllib.error.URLError) as e:
                self._url.pop(shard, None)        # an expired signed URL (403) or a dropped connection
                if attempt == 7:
                    raise
                print(f"  retry {shard} {rng}: {e}", file=sys.stderr, flush=True)
                time.sleep(min(30, 2 ** attempt))
        raise AssertionError

    def _header(self, shard: str) -> bytes:
        n = struct.unpack("<Q", self.fetch(shard, (0, 8)))[0]
        return self.fetch(shard, (8, 8 + n))

    def _meta(self, name):
        shard = self.map[name]
        return shard, self.headers[shard][name]

    def shape(self, name):
        return tuple(self._meta(name)[1]["shape"])

    def array(self, name):
        shard, m = self._meta(name)
        a, b = m["data_offsets"]
        d0 = self.data0[shard]
        return np.frombuffer(self.fetch(shard, (d0 + a, d0 + b)), dtype=DTYPES[m["dtype"]]).reshape(m["shape"])

    def full_record(self, L, E) -> dict[str, np.ndarray]:
        """The expert's tensors in one request when they sit together in one shard (they do in
        Mia's checkpoints), else one request a tensor."""

        p = f"{self.prefix}{L}.mlp.experts.{E}."
        names = [f"{p}{part}.{t}" for _, part in PARTS for t in ("trellis", "suh", "svh", self.codebook)]
        shards = {self.map[n] for n in names}
        if len(shards) != 1:
            return super().full_record(L, E)
        shard = shards.pop()
        metas = {n: self.headers[shard][n] for n in names}
        lo = min(m["data_offsets"][0] for m in metas.values())
        hi = max(m["data_offsets"][1] for m in metas.values())
        if hi - lo != sum(m["data_offsets"][1] - m["data_offsets"][0] for m in metas.values()):
            return super().full_record(L, E)
        d0 = self.data0[shard]
        blob = self.fetch(shard, (d0 + lo, d0 + hi))
        out = {}
        for tag, part in PARTS:
            for t in ("trellis", "suh", "svh"):
                m = metas[f"{p}{part}.{t}"]
                a, b = m["data_offsets"]
                dt = np.dtype(DTYPES[m["dtype"]])
                out[("t" if t == "trellis" else t) + tag] = np.frombuffer(
                    blob, dtype=dt, count=(b - a) // dt.itemsize, offset=a - lo).reshape(m["shape"])
        return out


class PackSource:
    """A --world 1 pack, re-sliced into a rank of a larger world."""

    def __init__(self, path: str):
        self.reader = R.PackReader(path)          # applies the pack's dims
        if self.reader.world != 1:
            raise SystemExit(f"{path}: a pack of world {self.reader.world}; --from-pack takes a --world 1 pack")
        self.codebook = self.reader.codebook
        self.sha = self.reader.header["source_sha256"]
        self.hidden, self.inter = self.reader.header["dims"]
        self.layers = sorted(self.reader.bits_map)
        self.n_experts = max(E for _, E in self.reader.records) + 1
        self.where = path

    def expert_bits(self, L, E):
        b = self.reader.records[(L, E)]["bits"]
        return tuple(float(v) for v in b) if isinstance(b, list) else (float(b),) * 3

    def marker_ok(self, L, E) -> bool:
        return True                               # checked when the --world 1 pack was built

    def full_record(self, L, E):
        return self.reader.read_expert(L, E)


def prefetched(keys, fn, workers: int):
    """fn over keys on a pool, yielded in key order with at most 2 x workers in flight."""

    if workers <= 1:
        for k in keys:
            yield k, fn(k)
        return
    it = iter(keys)
    with cf.ThreadPoolExecutor(workers) as ex:
        q = collections.deque()
        for k in it:
            q.append((k, ex.submit(fn, k)))
            if len(q) >= 2 * workers:
                break
        while q:
            k, f = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(fn, nxt)))
            yield k, f.result()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--source", help="a checkpoint directory")
    src_g.add_argument("--repo", help="a Hub repo, read by range requests")
    src_g.add_argument("--from-pack", help="a --world 1 pack to re-slice")
    ap.add_argument("--revision", default="main", help="with --repo: a branch, tag or commit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--workers", type=int, default=32, help="experts read at once (--repo: 32 reached ~70 MB/s)")
    ap.add_argument("--resume", action="store_true", help="keep the records an interrupted run of the same pack wrote")
    ap.add_argument("--verify", type=int, default=0, help="re-read N random records from the source, one tensor at a time")
    ap.add_argument("--experts-per-layer", type=int, default=0, help="a small test pack (0: all)")
    ap.add_argument("--layers", default="", help="comma-separated subset, for a test pack")
    a = ap.parse_args(argv)
    if a.source:
        src = GlmExl3Source(a.source)
    elif a.repo:
        src = HubSource(a.repo, a.revision)
    else:
        src = PackSource(a.from_pack)
    layers = [int(L) for L in a.layers.split(",")] if a.layers else src.layers
    n = a.experts_per_layer or src.n_experts
    for L in layers:
        if not src.marker_ok(L, 0):
            raise SystemExit(f"layer {L} expert 0: {src.codebook} marker mismatch")
    ebits = {(L, E): src.expert_bits(L, E) for L in layers for E in range(n)}
    mixed = any(len(set(b)) > 1 for b in ebits.values()) or any(
        ebits[(L, E)] != ebits[(L, 0)] for L in layers for E in range(n))
    # the header's per-layer width: the layer's own, or its average for a mixed pack
    bits = {L: sum(sum(ebits[(L, E)]) for E in range(n)) / (3 * n) for L in layers}
    widths = sorted({b for v in ebits.values() for b in v})
    print(f"[pack] {len(layers)} layers x {n} experts, {src.codebook}, widths {widths}{' (mixed)' if mixed else ''}, "
          f"rank {a.rank}/{a.world}, dims {src.hidden}x{src.inter}, source {src.sha[:16]}", flush=True)
    keys = [(L, E) for L in sorted(bits) for E in range(n)]
    if a.resume:
        done_keys = R.written_records(a.out)
        keys = [k for k in keys if k not in done_keys]
        if done_keys:
            print(f"[pack] resuming: {len(done_keys)} records already written (kept only if the header matches)", flush=True)
    t0, done, nbytes = time.time(), [0], [0]

    def records():
        for key, full in prefetched(keys, lambda k: src.full_record(*k), 1 if a.source else a.workers):
            yield key, rank_slice(full, a.rank, a.world)
            done[0] += 1
            nbytes[0] += sum(v.nbytes for v in full.values())
            if done[0] % 512 == 0:
                dt = time.time() - t0
                print(f"[pack]   {done[0]}/{len(keys)} experts, {dt:.0f}s, {nbytes[0] / dt / 1e6:.0f} MB/s read", flush=True)

    R.write_pack(a.out, bits, a.rank, a.world, src.codebook, src.sha, records(), n_experts=n,
                 expert_bits=ebits if mixed else None, resume=a.resume)
    r = R.PackReader(a.out)
    print(f"[pack] wrote {a.out}: {os.path.getsize(a.out) / 1e9:.1f} GB, {len(r.records)} records, "
          f"header {r.header_sha256()[:16]}, source {r.header['source_sha256'][:16]}, {time.time() - t0:.0f}s", flush=True)
    if a.verify:
        missing = sorted(set(r.records) - R.written_records(a.out))
        if missing:
            raise SystemExit(f"[pack] {len(missing)} records hold no data, first {missing[:4]}")
        rnd = random.Random(0)
        n_check = min(a.verify, len(r.records))
        for L, E in rnd.sample(sorted(r.records), n_check):
            if isinstance(src, PackSource):
                full = src.full_record(L, E)
            else:   # another path than the pack's: a tensor at a time
                full = GlmExl3Source.full_record(src, L, E)
            want, got = rank_slice(full, a.rank, a.world), r.read_expert(L, E)
            for name in R.RECORD_TENSORS:
                if not np.array_equal(want[name].view(np.uint16), got[name].view(np.uint16)):
                    raise SystemExit(f"[pack] verify: layer {L} expert {E} {name} differs from the source")
        print(f"[pack] verified {n_check} records against the source, every record written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
