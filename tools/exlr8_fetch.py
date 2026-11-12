"""Fetch what one exlr8 profile needs from kindlingai/glm-5.3-exlr8-k2-k3-k4, by HTTP Range.

    python3 tools/exlr8_fetch.py --manifest bit260 --out ~/models/glm-5.3-exlr8 [--jobs 8] [--layers 3-78]

The release is 838 GB because every routed expert is stored at K2, K3 and K4; a profile (manifest)
names one width a unit, so it needs a third of it or less (bit260: 240 GB). This writes a local
mirror in the release's own layout - the small files whole (config, tokenizer, manifests, a layer's
index file), each trellis file as a sparse file of the remote's size holding only the units the
manifest names, at their own offsets - so tools/exlr8_ref.py's SafeFile, the packer and the
publisher's loader read it as they would the full download, and another profile's fetch adds to it.

Safe beside a serving engine (stdlib only): each of --jobs connections streams 1 MB at a time into
pwrite, and every finished piece is fsync'ed and dropped from the page cache, so the process holds
tens of MB and the dirty pages stay under jobs x PIECE. It stops if MemAvailable falls under
--min-free-mb. Resumable: finished pieces are journalled in OUT/.fetch/, a rerun skips them.
Checksums are the packer's job (it reads every unit anyway).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/resolve/main/"
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "README.md")
PIECE = 32 << 20
PROJS = ("gate", "up", "down")
UNITS = 16


def http(path: str, off: int | None = None, n: int | None = None, tries: int = 8):
    """An open response for REPO/path (a byte range when off is given), retried with backoff."""
    for t in range(tries):
        try:
            req = urllib.request.Request(REPO + path)
            if off is not None:
                req.add_header("Range", f"bytes={off}-{off + n - 1}")
            return urllib.request.urlopen(req, timeout=120)
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                raise
            time.sleep(min(60, 2 ** t))
            print(f"  retry {t + 1} {path}: {e}", flush=True)


def fetch_small(path: str, out: str) -> str:
    dst = os.path.join(out, path)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with http(path) as r, open(dst + ".part", "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(dst + ".part", dst)
    return dst


def header(path: str):
    """A safetensors file's (header dict, data start) from the hub."""
    with http(path, 0, 8) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    with http(path, 8, n) as r:
        h = json.loads(r.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def mem_available_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 1 << 30


class Mirror:
    """One remote file's sparse local copy, its journal of finished pieces."""

    def __init__(self, out: str, path: str):
        self.path = path
        self.local = os.path.join(out, path)
        self.jpath = os.path.join(out, ".fetch", path.replace("/", "__") + ".json")
        os.makedirs(os.path.dirname(self.local), exist_ok=True)
        os.makedirs(os.path.dirname(self.jpath), exist_ok=True)
        self.head, self.data0 = header(path)
        self.size = self.data0 + max(v["data_offsets"][1] for v in self.head.values())
        self.done = set(tuple(p) for p in json.load(open(self.jpath))) if os.path.exists(self.jpath) else set()
        self.lock = threading.Lock()
        self.fd = os.open(self.local, os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(self.fd).st_size != self.size:
            os.ftruncate(self.fd, self.size)
        if (0, self.data0) not in self.done:           # the header, so the file opens as a safetensors
            with http(path, 0, self.data0) as r:
                os.pwrite(self.fd, r.read(self.data0), 0)
            self.mark(0, self.data0)

    def tensor(self, name: str):
        a, b = self.head[name]["data_offsets"]
        return self.data0 + a, b - a

    def mark(self, off: int, n: int):
        with self.lock:
            self.done.add((off, n))
            with open(self.jpath + ".tmp", "w") as f:
                json.dump(sorted(self.done), f)
            os.replace(self.jpath + ".tmp", self.jpath)

    def piece(self, off: int, n: int) -> int:
        if (off, n) in self.done:
            return 0
        got = 0
        with http(self.path, off, n) as r:
            while got < n:
                b = r.read(min(1 << 20, n - got))
                if not b:
                    raise IOError(f"{self.path}: short read at {off + got}")
                os.pwrite(self.fd, b, off + got)
                got += len(b)
        os.fsync(self.fd)
        os.posix_fadvise(self.fd, off, n, os.POSIX_FADV_DONTNEED)
        self.mark(off, n)
        return n


def runs(units: list[int], stride: int, base: int):
    """Byte ranges of the unit indices (sorted), consecutive ones merged, cut into PIECE-sized pieces."""
    out, i = [], 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and units[j + 1] == units[j] + 1:
            j += 1
        a, b = base + units[i] * stride, base + (units[j] + 1) * stride
        while a < b:
            out.append((a, min(PIECE, b - a)))
            a += PIECE
        i = j + 1
    return out


def missing(pieces, done):
    """The parts of the pieces that no journalled range holds, as pieces of their own.

    Another profile's fetch cut the same files at other offsets (bit260 took every down at K3, home
    takes 192 of a layer's 256), so a piece is compared with the bytes already here, not with the
    journal's entries: without this home after bit260 planned 292.9 GB to fetch where 172.1 GB were new."""
    have = []
    for o, n in sorted(done):
        if have and o <= have[-1][1]:
            have[-1][1] = max(have[-1][1], o + n)
        else:
            have.append([o, o + n])
    out = []
    for o, n in pieces:
        a, b = o, o + n
        for s, e in have:
            if e <= a:
                continue
            if s >= b:
                break
            if s > a:
                out.append((a, s - a))
            a = max(a, e)
        if a < b:
            out.append((a, b - a))
    return out


def plan_layer(out: str, layer: int, lay: dict, hidden: int = 6144):
    """(Mirror, [(off, n)]) a trellis file the manifest's layer reads: the units at each width."""
    need: dict[int, dict[str, list[int]]] = {}
    experts = len(lay["down"])
    for pi, p in enumerate(PROJS):
        for e in range(experts):
            ks = [lay["down"][e]] * UNITS if p == "down" else lay[p][e]
            for u, k in enumerate(ks):
                need.setdefault(int(k), {}).setdefault(p, []).append(e * UNITS + u)
    sh = lay.get("shared")
    plans = []
    for k in sorted(set(need) | ({int(sh["down"])} | {int(v) for v in sh["gate"] + sh["up"]} if sh else set())):
        m = Mirror(out, f"experts/exlr8-layer-{layer:03d}.k{k}.safetensors")
        stride = hidden * 16 * k
        pieces = []
        for p, units in need.get(k, {}).items():
            base, _ = m.tensor(f"trellis.{p}")
            pieces += runs(units, stride, base)
        if sh:
            for p in PROJS:
                ks = [sh["down"]] * UNITS if p == "down" else sh[p]
                units = [u for u, kk in enumerate(ks) if int(kk) == k]
                if units:
                    base, _ = m.tensor(f"shared.trellis.{p}")
                    pieces += runs(units, stride, base)
        plans.append((m, pieces))
    return plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="bit260")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--layers", default="3-78")
    ap.add_argument("--min-free-mb", type=int, default=4600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dense", action="store_true", help="the dense layer files (8.8 GB) instead of a profile's experts")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    for s in SMALL:
        fetch_small(s, out)
    if a.dense:
        # dense/exlr8-dense-LLL.safetensors, whole (0.1-0.23 GB each); the vocabulary file (embedding and
        # head in FP8, 1.9 GB) is left: nothing here reads it yet
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda L: fetch_small(f"dense/exlr8-dense-{L:03d}.safetensors", out), range(79)))
        print("dense: 79 layer files", flush=True)
        return
    man = json.load(open(fetch_small(f"manifests/exlr8-manifest-{a.manifest}.json", out)))
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = [int(l) for l in sorted(man["layers"], key=int) if lo <= int(l) <= hi]
    total = todo = 0
    work = []
    for L in layers:
        fetch_small(f"experts/exlr8-layer-{L:03d}.index.safetensors", out)
        for m, pieces in plan_layer(out, L, man["layers"][str(L)]):
            total += sum(n for _, n in pieces)
            left = [(m, o, n) for o, n in missing(pieces, m.done)]
            todo += sum(n for _, _, n in left)
            work += left
    free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
    print(f"{a.manifest}: layers {layers[0]}-{layers[-1]}: {total / 1e9:.1f} GB of trellis, {todo / 1e9:.1f} GB to fetch in {len(work)} pieces; "
          f"{free / 1e9:.0f} GB free at {out}", flush=True)
    if a.dry_run or not work:
        return
    if todo > free - (40 << 30):
        sys.exit("not enough disk: the fetch would leave under 40 GB free")
    t0, got, lock, stop = time.time(), [0, 0.0], threading.Lock(), threading.Event()

    def one(job):
        if stop.is_set():
            return
        if mem_available_mb() < a.min_free_mb:
            stop.set()
            return
        m, o, n = job
        b = m.piece(o, n)
        with lock:
            got[0] += b
            if time.time() - got[1] > 60:
                got[1] = time.time()
                dt = time.time() - t0
                print(f"  {got[0] / 1e9:7.1f} / {todo / 1e9:.1f} GB, {got[0] / 1e6 / dt:.0f} MB/s, "
                      f"{(todo - got[0]) / max(got[0], 1) * dt / 60:.0f} min left, MemAvailable {mem_available_mb()} MB", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, work))
    if stop.is_set():
        sys.exit(f"stopped: MemAvailable under {a.min_free_mb} MB; rerun to resume")
    print(f"done: {got[0] / 1e9:.1f} GB in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
