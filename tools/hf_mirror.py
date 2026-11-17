"""Mirror a whole Hugging Face model repository at one revision, safe beside a serving engine.

    python3 tools/hf_mirror.py --repo 0xSero/Inkling-2-Small-EXL3-4bpw --revision <sha> \
        --out ~/models/Inkling-2-Small-EXL3-4bpw [--jobs 8] [--min-free-mb 4600] [--verify]

tools/exlr8_fetch.py's writer for files taken whole (stdlib only, no token): each of --jobs
connections streams 1 MB at a time into pwrite, every finished 32 MB piece is fsync'ed and dropped
from the page cache, so the process holds tens of MB and the dirty pages stay under jobs x PIECE
(`hf download` held 3.5 GB and cost the server its memory guard, 2026-10-25). It stops if
MemAvailable falls under --min-free-mb. Resumable: finished pieces are journalled in OUT/.fetch/.
--verify reads every LFS file back against the sha256 the Hub lists, dropping the cache as it goes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PIECE = 32 << 20


def http(url: str, off: int | None = None, n: int | None = None, tries: int = 8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if off is not None:
                req.add_header("Range", f"bytes={off}-{off + n - 1}")
            return urllib.request.urlopen(req, timeout=120)
        except (urllib.error.URLError, OSError) as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
            last = e
    raise last


def mem_available_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 1 << 30


class File:
    """One remote file's local copy and its journal of finished pieces."""

    def __init__(self, out: str, base: str, path: str, size: int):
        self.path, self.size, self.url = path, size, base + path
        self.local = os.path.join(out, path)
        self.jpath = os.path.join(out, ".fetch", path.replace("/", "__") + ".json")
        os.makedirs(os.path.dirname(self.local) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.jpath), exist_ok=True)
        self.done = set(tuple(p) for p in json.load(open(self.jpath))) if os.path.exists(self.jpath) else set()
        self.lock = threading.Lock()
        self.fd = None

    def pieces(self):
        return [(o, min(PIECE, self.size - o)) for o in range(0, self.size, PIECE) if (o, min(PIECE, self.size - o)) not in self.done]

    def piece(self, off: int, n: int) -> int:
        with self.lock:
            if self.fd is None:
                self.fd = os.open(self.local, os.O_RDWR | os.O_CREAT, 0o644)
                if os.fstat(self.fd).st_size != self.size:
                    os.ftruncate(self.fd, self.size)
        got = 0
        for attempt in range(6):                        # a connection dropped mid-piece: the piece again
            try:
                with http(self.url, off + got, n - got) as r:
                    while got < n:
                        b = r.read(min(1 << 20, n - got))
                        if not b:
                            raise IOError("short read")
                        os.pwrite(self.fd, b, off + got)
                        got += len(b)
                break
            except (OSError, urllib.error.URLError):
                if attempt == 5:
                    raise
                time.sleep(3 * (attempt + 1))
        os.fsync(self.fd)
        os.posix_fadvise(self.fd, off, n, os.POSIX_FADV_DONTNEED)
        with self.lock:
            self.done.add((off, n))
            with open(self.jpath + ".tmp", "w") as f:
                json.dump(sorted(self.done), f)
            os.replace(self.jpath + ".tmp", self.jpath)
        return n


def sha256_file(path: str) -> str:
    h, fd, off = hashlib.sha256(), os.open(path, os.O_RDONLY), 0
    try:
        while True:
            b = os.pread(fd, 8 << 20, off)
            if not b:
                break
            h.update(b)
            off += len(b)
            if off % (256 << 20) == 0:
                os.posix_fadvise(fd, 0, off, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="a commit sha: the mirror is of one revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--min-free-mb", type=int, default=4600)
    ap.add_argument("--include", action="append", help="only paths with this prefix (repeatable)")
    ap.add_argument("--verify", action="store_true", help="check every LFS file's sha256 instead of fetching")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.join(out, ".fetch"), exist_ok=True)
    with http(f"https://huggingface.co/api/models/{a.repo}/revision/{a.revision}?blobs=true") as r:
        info = json.load(r)
    if info.get("sha") != a.revision:
        sys.exit(f"{a.repo}: revision {a.revision} resolves to {info.get('sha')}; name the full commit sha")
    sib = [s for s in info["siblings"] if not a.include or any(s["rfilename"].startswith(p) for p in a.include)]
    base = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/"
    with open(os.path.join(out, ".fetch", "files.json"), "w") as f:
        json.dump({"repo": a.repo, "revision": a.revision,
                   "files": {s["rfilename"]: {"size": s["size"], "sha256": (s.get("lfs") or {}).get("sha256")} for s in sib}}, f, indent=1)

    if a.verify:
        bad = 0
        for s in sib:
            p, want = os.path.join(out, s["rfilename"]), (s.get("lfs") or {}).get("sha256")
            if not os.path.exists(p) or os.path.getsize(p) != s["size"]:
                print(f"MISSING or wrong size: {s['rfilename']}", flush=True)
                bad += 1
            elif want:
                if mem_available_mb() < a.min_free_mb:
                    sys.exit(f"stopped: MemAvailable under {a.min_free_mb} MB; rerun to resume")
                ok = os.path.join(out, ".fetch", s["rfilename"].replace("/", "__") + ".ok")
                if os.path.exists(ok) and open(ok).read().strip() == want:
                    continue
                got = sha256_file(p)
                if got == want:
                    open(ok, "w").write(want)
                else:
                    print(f"SHA256 MISMATCH: {s['rfilename']}: {got} != {want}", flush=True)
                    bad += 1
        print(f"verify: {len(sib)} files, {bad} bad", flush=True)
        sys.exit(1 if bad else 0)

    files = [File(out, base, s["rfilename"], s["size"]) for s in sib]
    work = [(f, o, n) for f in files for (o, n) in f.pieces()]
    total, todo = sum(f.size for f in files), sum(n for _, _, n in work)
    free = shutil.disk_usage(out).free
    print(f"{a.repo}@{a.revision[:8]}: {len(files)} files, {total / 1e9:.1f} GB, {todo / 1e9:.1f} GB to fetch in {len(work)} pieces; "
          f"{free / 1e9:.0f} GB free", flush=True)
    for f in files:                                     # empty files have no piece
        if f.size == 0:
            open(f.local, "a").close()
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
        f, o, n = job
        b = f.piece(o, n)
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
