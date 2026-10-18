"""Prompt prefixes kept on disk across requests, conversations and restarts (GLM_PREFIX_DISK=1).

GLM's state after a prompt is its KV rows -- the MLA latent, the RoPE keys, the indexer keys, per
layer -- plus the MTP layer's input hidden at the last position. The in-memory cache (GlmTP.
reusable_prefix) reuses one sequence: another conversation, or one of Open WebUI's background
requests (title, tags, follow-ups), replaces it, and the next turn of a long chat prefills from
scratch -- ~4 minutes at 100K tokens, where reading its rows back takes well under a second.

Layout. A segment holds the rows of positions [start, end) of one prompt and names the segment it
continues (parent: the rows of [0, start)), so a growing conversation writes only each turn's new
tokens -- DS V4.1's bundles rewrite the whole prompt on every save, its documented limitation. A
segment is two files in this rank's directory: <id>.kv (a JSON header, then one 4 KB-aligned block
of rows per cache tensor and the hidden at end - 1; O_DIRECT both ways, so a restore does not fill
the page cache, which on GB10 is the GPU's memory) and <id>.tok (the tokens of [start, end), int32).
index.json lists the segments: parent, start, end, the SHA-256 of tokens [0, end), bytes, last use.

Lookup (rank 0): the deepest segment whose hash matches the prompt's prefix at its end, then the
longest token match into one of its children (or, with none, into a root -- another chat with the
same system prompt): a partial restore, without the hidden. Every restore compares the chain's
tokens with the prompt before use. Rank 0 decides what to restore, save and evict and broadcasts
it; each rank checks and reads its own files, and a miss or failure on either rank is a miss on
both (GlmTP._disk_restore). Restores copy into the existing cache tensors, so captured graphs stay
valid.

The directory is namespaced by everything that changes the rows' meaning (engine code digest, KV
format, split, expert format, model, rank): a code change starts a new namespace, and old ones are
the first evicted. Rows depend on the resident experts when they were computed; like DS V4.1's
default, a restored prefix is not recomputed because adaptation changed them since.

Token ids and KV reveal the prompt: files are 0600 in 0700 directories, under results/ (ignored
by Git). Never share a rank directory between servers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid

import numpy as np
import torch

from .direct_io import ALIGN, BufPool, align_up, pread_direct

HEADER = 64 * 1024          # bytes reserved for a segment file's JSON header


def tok_bytes(tokens) -> bytes:
    return np.asarray(tokens, dtype=np.int32).tobytes()


class PrefixStore:
    """One rank's segments. The engine's layout lists the cache tensors as (name, tensor [rows,
    ...], mode): "split" -- row j holds position j * world + rank (GLM_KV_SPLIT); "full" -- row p
    holds position p; "mtp" -- the MTP layer's, whose row p is made with token p + 1, so a segment
    of positions [start, end) holds its rows [start - 1, end - 1): the ones its tokens determine.
    Its last row is redone at a restore that comes with the hidden (GlmTP.prefill)."""

    def __init__(self, root: str, namespace: str, rank: int, world: int, budget_bytes: int,
                 free_floor_bytes: int, log=print, gpu_lock=None):
        self.root, self.ns, self.rank, self.world = root, namespace, rank, world
        self.dir = os.path.join(root, namespace, f"rank-{rank}")
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        for d in (root, os.path.join(root, namespace)):
            os.chmod(d, 0o700)
        self.budget, self.floor, self.log = budget_bytes, free_floor_bytes, log
        self.lock = threading.Lock()
        self.pending: threading.Thread | None = None
        self.pending_err: list = []
        self.pool = BufPool(64 << 20, 2)
        self.stream = None
        # The engine's graph-capture lock: the save thread's GPU work waits out a capture.
        self.gpu_lock = gpu_lock or threading.Lock()
        self.idx = self._load_index()

    # ---------------------------------------------------------------- index
    def _index_path(self):
        return os.path.join(self.dir, "index.json")

    def _load_index(self) -> dict:
        try:
            with open(self._index_path()) as f:
                idx = json.load(f)
        except (OSError, ValueError):
            idx = {"segments": {}}
        segs = idx["segments"]
        for sid in [s for s, m in segs.items() if not os.path.exists(self._kv(s))]:
            segs.pop(sid)                                   # a write that never published
        return idx

    def _save_index(self):
        tmp = self._index_path() + ".tmp"
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(self.idx, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._index_path())

    def _kv(self, sid):
        return os.path.join(self.dir, sid + ".kv")

    def _tok(self, sid):
        return os.path.join(self.dir, sid + ".tok")

    def segs(self) -> dict:
        return self.idx["segments"]

    def chain(self, sid) -> list[str]:
        """Root .. sid."""
        out = []
        while sid is not None:
            out.append(sid)
            sid = self.segs()[sid]["parent"]
        return out[::-1]

    def tokens(self, sid) -> np.ndarray:
        return np.fromfile(self._tok(sid), dtype=np.int32)

    # ---------------------------------------------------------------- lookup (rank 0)
    def match(self, prompt, limit: int):
        """(deepest segment whose end <= limit matches the prompt at its end, or None; and the
        longest extension into one of its children -- or a root -- as (child, end), or None)."""
        segs = self.segs()
        by_end: dict[int, list[str]] = {}
        for sid, m in segs.items():
            if m["end"] <= limit:
                by_end.setdefault(m["end"], []).append(sid)
        h, prev, best = hashlib.sha256(), 0, None
        for e in sorted(by_end):
            h.update(tok_bytes(prompt[prev:e]))
            prev = e
            d = h.hexdigest()
            hit = [s for s in by_end[e] if segs[s]["hash"] == d]
            if hit:
                best = max(hit, key=lambda s: segs[s]["used"])
        base = segs[best]["end"] if best else 0
        kids = sorted((s for s, m in segs.items() if m["parent"] == best and m["start"] == base),
                      key=lambda s: -segs[s]["used"])[:8]
        ext = None
        want = np.asarray(prompt[base:limit], dtype=np.int32)
        for s in kids:
            try:
                t = self.tokens(s)[:len(want)]
            except OSError:
                continue
            diff = np.nonzero(t != want[:len(t)])[0]
            m = int(diff[0]) if len(diff) else len(t)
            if m and (ext is None or base + m > ext[1]):
                ext = (s, base + m)
        return best, ext

    def plan_restore(self, prompt, have: int, min_gain: int):
        """What to read for `prompt` when positions [0, have) are already cached: a dict with the
        chain [(sid, start, end)], the restored length n and whether the hidden at n - 1 comes
        with it -- or None."""
        limit = len(prompt) - 1                    # at least one token is prefilled (the logits)
        best, ext = self.match(prompt, limit)
        segs = self.segs()
        n = segs[best]["end"] if best else 0
        ids = self.chain(best) if best else []
        h_ok = bool(best)
        if ext is not None and ext[1] > n:
            ids, n, h_ok = ids + [ext[0]], ext[1], ext[1] == segs[ext[0]]["end"]
        if n - have < min_gain:
            return None
        return {"chain": [(s, segs[s]["start"], segs[s]["end"]) for s in ids], "n": n, "h": h_ok}

    def plan_save(self, prompt, min_tokens: int):
        """A new segment for `prompt`'s rows past its deepest saved prefix -- or None."""
        L = len(prompt)
        best, _ = self.match(prompt, L)
        segs = self.segs()
        start = segs[best]["end"] if best else 0
        if L - start < min_tokens:
            return None
        return {"id": uuid.uuid4().hex[:16], "parent": best, "start": start, "end": L,
                "hash": hashlib.sha256(tok_bytes(prompt[:L])).hexdigest()}

    def plan_evict(self, incoming: int, keep=()) -> dict | None:
        """Least recently used first: other namespaces whole, then this one's leaves (never one in
        `keep`), until the published bytes plus `incoming` fit the budget; None if they cannot."""
        others = []
        for ns in os.listdir(self.root):
            p = os.path.join(self.root, ns)
            if ns != self.ns and os.path.isdir(p):
                others.append((os.path.getmtime(p), ns, _du(p)))
        segs = {s: dict(m) for s, m in self.segs().items()}
        total = sum(b for _, _, b in others) + sum(m["bytes"] for m in segs.values()) + incoming
        drop_ns, drop = [], []
        for _, ns, b in sorted(others):
            if total <= self.budget:
                break
            drop_ns.append(ns)
            total -= b
        while total > self.budget:
            parents = {m["parent"] for m in segs.values()}
            leaves = [s for s in segs if s not in parents and s not in keep]
            if not leaves:
                return None
            s = min(leaves, key=lambda s: segs[s]["used"])
            total -= segs.pop(s)["bytes"]
            drop.append(s)
        return {"ns": drop_ns, "segs": drop}

    # ---------------------------------------------------------------- both ranks
    def touch(self, sids):
        now = time.time()
        with self.lock:
            for s in sids:
                if s in self.segs():
                    self.segs()[s]["used"] = now
            self._save_index()

    def missing(self, plan) -> list[str]:
        """The plan's segments this rank cannot read."""
        out = []
        for sid, _, _ in plan["chain"]:
            if sid not in self.segs() or not os.path.exists(self._kv(sid)) or not os.path.exists(self._tok(sid)):
                out.append(sid)
        return out

    def drop(self, sids, namespaces=()):
        """Delete segments (and their descendants) and whole other namespaces."""
        for ns in namespaces:
            if ns != self.ns:
                shutil.rmtree(os.path.join(self.root, ns), ignore_errors=True)
        with self.lock:
            segs = self.segs()
            gone, todo = set(), [s for s in sids if s in segs]
            while todo:
                s = todo.pop()
                if s in gone:
                    continue
                gone.add(s)
                todo += [c for c, m in segs.items() if m["parent"] == s]
            for s in gone:
                segs.pop(s, None)
                for p in (self._kv(s), self._tok(s)):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            if gone:
                self._save_index()

    def room(self, nbytes: int) -> bool:
        st = os.statvfs(self.dir)
        return st.f_bavail * st.f_frsize - nbytes >= self.floor

    def verify_tokens(self, plan, prompt) -> bool:
        for sid, a, b in plan["chain"]:
            hi = min(b, plan["n"])
            t = self.tokens(sid)[:hi - a]
            if len(t) != hi - a or not np.array_equal(t, np.asarray(prompt[a:hi], dtype=np.int32)):
                return False
        return True

    # ---------------------------------------------------------------- rows
    def _rows(self, mode: str, p: int) -> int:
        """The first row a segment starting at position p holds (see the class docstring)."""
        if mode == "split":
            return max(0, (p - self.rank + self.world - 1) // self.world)
        return max(0, p - 1) if mode == "mtp" else p

    def _blocks(self, layout, start, end, h_parts):
        blocks, off = [], HEADER
        for name, t, mode in layout:
            lo, hi = self._rows(mode, start), self._rows(mode, end)
            rb = t[0].numel() * t.element_size()
            blocks.append([name, lo, hi, rb, off])
            off += align_up((hi - lo) * rb)
        hb = []
        for name, t in h_parts:
            rb = t.numel() * t.element_size()
            hb.append([name, rb, off])
            off += align_up(rb)
        return blocks, hb, off

    def estimate(self, layout, start, end) -> int:
        return self._blocks(layout, start, end, [])[2]

    def write_async(self, plan, prompt, layout, h_parts, after: torch.cuda.Event | None):
        """Write the plan's segment from the live caches on a side stream, then publish it. The
        caller must not change rows of [start, end) until join() returns."""
        self.join()

        def run():
            try:
                self._write(plan, prompt, layout, h_parts, after)
            except Exception as e:                  # noqa: BLE001  -- a failed save is a miss later
                self.pending_err.append(e)
                self.log(f"[prefix] save {plan['id']} failed: {e!r}")
        self.pending = threading.Thread(target=run, name="prefix-save", daemon=True)
        self.pending.start()

    def join(self):
        if self.pending is not None:
            self.pending.join()
            self.pending = None

    def _write(self, plan, prompt, layout, h_parts, after):
        t0 = time.perf_counter()
        sid, start, end = plan["id"], plan["start"], plan["end"]
        blocks, hb, size = self._blocks(layout, start, end, h_parts)
        tensors = {name: t for name, t, _ in layout}
        hts = dict(h_parts)
        with self.gpu_lock:
            if self.stream is None and torch.cuda.is_available():
                self.stream = torch.cuda.Stream()
            if after is not None:
                self.stream.wait_event(after)
        tmp = self._kv(sid) + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_DIRECT", 0), 0o600)
        try:
            with torch.cuda.stream(self.stream) if self.stream is not None else _null():
                head = json.dumps({"v": 1, "start": start, "end": end, "rank": self.rank, "world": self.world,
                                   "blocks": blocks, "h": hb}).encode()
                assert len(head) < HEADER, len(head)
                lease = self.pool.lease(HEADER)
                buf = np.frombuffer(lease, dtype=np.uint8, count=HEADER)
                buf[:] = 0
                buf[:len(head)] = np.frombuffer(head, dtype=np.uint8)
                _pwrite(fd, memoryview(lease)[:HEADER], 0)
                del buf, lease
                for name, lo, hi, rb, off in blocks:
                    self._put(fd, tensors[name][lo:hi], off)
                for name, rb, off in hb:
                    self._put(fd, hts[name], off)
            os.fsync(fd)
        finally:
            os.close(fd)
        with open(os.open(self._tok(sid), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
            f.write(tok_bytes(prompt[start:end]))
        os.replace(tmp, self._kv(sid))
        with self.lock:
            self.segs()[sid] = {"parent": plan["parent"], "start": start, "end": end, "hash": plan["hash"],
                                "bytes": size + 4 * (end - start), "used": time.time()}
            self._save_index()
        self.log(f"[prefix] saved [{start}, {end}) as {sid}: {size / 1e6:.1f} MB in "
                 f"{time.perf_counter() - t0:.2f}s")

    def _put(self, fd, t: torch.Tensor, off: int):
        """Write a tensor's bytes at aligned offset `off`, through the pooled buffers."""
        flat = t.contiguous().view(torch.uint8).reshape(-1)
        n, step = flat.numel(), self.pool.size
        for a in range(0, n, step):
            b = min(n, a + step)
            m = align_up(b - a)
            lease = self.pool.lease(m)
            host = torch.from_numpy(np.frombuffer(lease, dtype=np.uint8, count=m))
            with self.gpu_lock:
                host[:b - a].copy_(flat[a:b])
            if m > b - a:
                host[b - a:].zero_()
            del host
            _pwrite(fd, memoryview(lease)[:m], off + a)

    def load(self, plan, have: int, layout, h_parts):
        """Read positions [have, n) of the plan's chain into the caches (and the hidden at n - 1)."""
        tensors = {name: (t, mode) for name, t, mode in layout}
        hts = dict(h_parts)
        n = plan["n"]
        for i, (sid, start, end) in enumerate(plan["chain"]):
            a, b = max(have, start), min(n, end)
            last = i == len(plan["chain"]) - 1
            if a >= b and not (last and plan["h"]):
                continue
            fd = os.open(self._kv(sid), os.O_RDONLY | getattr(os, "O_DIRECT", 0))
            try:
                hdr = torch.empty(HEADER, dtype=torch.uint8)
                self._read_into(fd, 0, HEADER, hdr)
                head = json.loads(hdr.numpy().tobytes().rstrip(b"\0"))
                assert head["start"] == start and head["end"] == end and head["rank"] == self.rank, head
                for name, lo, hi, rb, off in head["blocks"]:
                    t, mode = tensors[name]
                    assert rb == t[0].numel() * t.element_size(), (name, rb)
                    ra, rz = max(lo, self._rows(mode, a)), min(hi, self._rows(mode, b))
                    if rz > ra:
                        self._read_into(fd, off + (ra - lo) * rb, (rz - ra) * rb,
                                        t[ra:rz].view(torch.uint8).reshape(-1))
                if last and plan["h"]:
                    for name, rb, off in head["h"]:
                        self._read_into(fd, off, rb, hts[name].view(torch.uint8).reshape(-1))
            finally:
                os.close(fd)

    def _read_into(self, fd, off: int, n: int, dst: torch.Tensor):
        """O_DIRECT-read bytes [off, off + n) of the file into the flat uint8 tensor dst, a pooled
        buffer at a time."""
        step = self.pool.size - ALIGN
        for a in range(0, n, step):
            b = min(n, a + step)
            o = off + a
            al = o - o % ALIGN
            m = align_up(o + b - a) - al
            lease = self.pool.lease(m)
            got = pread_direct(fd, memoryview(lease)[:m], al, o + b - a - al)
            if got < o + b - a - al:
                raise OSError(f"short read: {got} of {o + b - a - al}")
            src = torch.from_numpy(np.frombuffer(lease, dtype=np.uint8, count=m))
            dst[a:b].copy_(src[o - al:o - al + b - a])
            del src, lease


def _pwrite(fd, view, off):
    done = 0
    while done < len(view):
        done += os.pwrite(fd, view[done:], off + done)


def _du(path) -> int:
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
