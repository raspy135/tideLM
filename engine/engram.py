"""
engram.py -- Engram table rows at serve time: 24 random 264-byte reads per token per engram
layer, straight from the two 101 GB safetensors shards on NVMe. By default through the page cache
(memmap faults); DSV41_ENGRAM_DIRECT=1 reads the covering 512-byte sectors with O_DIRECT instead,
which matters for what it does to the REST of the box, not for the reads themselves (see
EngramTable.__init__). Hash ids come from the reference `NgramHashState`, which depends on the
token ids only.
"""

from __future__ import annotations

import json
import os
import mmap
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager

import numpy as np
import torch


class EngramTable:
    def __init__(self, model_dir: str, index: dict, layer: int, device: str, threads: int = 32,
                 cache_bytes: int = 0):
        wm = index["weight_map"]
        self.path = os.path.join(model_dir, wm[f"layers.{layer}.engram.embed.weight"])
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        w = hdr[f"layers.{layer}.engram.embed.weight"]
        s = hdr[f"layers.{layer}.engram.embed.scale"]
        assert w["shape"][1] == 256 and s["shape"][1] == 8
        self.w_off = base + w["data_offsets"][0]
        self.s_off = base + s["data_offsets"][0]
        self.n_rows = w["shape"][0]
        self.row_cache = None
        if cache_bytes:
            from engine.engram_cache import PackedRowCache
            self.row_cache = PackedRowCache(cache_bytes, self.n_rows)
        self.fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        # Row gather over a memmap instead of a Python loop of preads. The loop cost ~12 us/row
        # even fully warm in page cache (two syscalls plus ~10 bytecode ops per row, and the GIL
        # kept the thread pool from helping): 83k rows/s warm, 47k cold. Fancy-indexing a memmap
        # does the whole gather in C and lets page faults do the I/O -- 8.9M rows/s warm, 891k
        # cold, byte-identical output. DSV41_ENGRAM_MMAP=0 restores the pread path.
        self.mmap_rows = os.environ.get("DSV41_ENGRAM_MMAP", "1") == "1"
        # DSV41_ENGRAM_DIRECT=1: the native workers pread the 512-byte sectors covering each row's
        # weights and scale from an O_DIRECT descriptor -- ~1 KB of I/O a row instead of two 4 KB
        # page faults, nothing added to the page cache, nothing mapped. Every gather size goes
        # there, decode's and prefill's. The reason is the box, not the read: with ~1 GB free per
        # node, a cold 8K prompt's ~1.5M faults force reclaim and compaction (700-1,500 compaction
        # stalls, 0.35-0.83M pages migrated a prompt), and while that runs the MoE kernels stall
        # inside the kernel -- single calls of 20-176 ms instead of ~5 ms. Same prompt with its
        # rows already cached: MoE phase 12.9 -> 5.3 s, prefill 19.6 -> 11.4 s (gate,
        # results/prefill-trace-20261008). Unrelated page-cache churn (cat of other files) slowed a
        # hot prefill 2.7x by itself. Same bytes either way, so tokens cannot change.
        # "prefill" (the serving setting) keeps decode-sized gathers (<= 384 rows) on the memmap:
        # there a page-cache hit beats an O_DIRECT round trip, and all-direct measured +0.89 +-
        # 0.43 ms/step over 22 paired prompts (results/prefill-trace-20261008/decode-engram-ab).
        mode = os.environ.get("DSV41_ENGRAM_DIRECT", "0")
        if mode not in ("0", "1", "prefill"):
            raise ValueError("DSV41_ENGRAM_DIRECT must be 0, 1 or prefill")
        self.direct = mode == "1"
        self.direct_large = mode != "0"
        self.direct_gather = None
        self.w_mm = self.s_mm = None
        if self.mmap_rows and not self.direct:
            self.w_mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                                  offset=self.w_off, shape=(self.n_rows, 256))
            self.s_mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                                  offset=self.s_off, shape=(self.n_rows, 8))
            for mm in (self.w_mm, self.s_mm):   # no readahead: every row lands on its own page
                try:
                    mm._mmap.madvise(mmap.MADV_RANDOM)
                except (AttributeError, OSError):
                    pass
        self.gather_threads = int(os.environ.get("DSV41_ENGRAM_GATHER_THREADS", "64"))
        # Rows below which the gather stays on the calling thread (see _gather_rows).
        self.gather_min_parallel = int(os.environ.get("DSV41_ENGRAM_GATHER_MIN", "32"))
        # Decode-sized gathers only; large prefill batches keep their existing path.
        self.native_gather = None
        if self.direct_large:
            from engine.engram_native import NativeGather
            if not self.mmap_rows:
                raise ValueError('DSV41_ENGRAM_DIRECT replaces the mmap row path; leave DSV41_ENGRAM_MMAP=1')
            self.direct_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            self.direct_gather = NativeGather(
                None, None, workers=int(os.environ.get('DSV41_ENGRAM_NATIVE_THREADS', '64')),
                direct=(self.direct_fd, self.w_off, self.s_off, self.n_rows))
        if self.direct:
            self.native_gather = self.direct_gather
        elif os.environ.get('DSV41_ENGRAM_NATIVE', '0') == '1':
            if not self.mmap_rows:
                raise ValueError('DSV41_ENGRAM_NATIVE requires the mmap row path')
            from engine.engram_native import NativeGather
            self.native_gather = NativeGather(self.w_mm, self.s_mm,
                workers=int(os.environ.get('DSV41_ENGRAM_NATIVE_THREADS', '64')))
        # EP2 row split: both ranks hash the same tokens, so both were reading the SAME rows off
        # their own NVMe -- the work was duplicated, and it is the prefill bottleneck. With the
        # split each rank reads only `uniq % world == rank` into a zero-filled full array and one
        # all-reduce (the halves are disjoint, so SUM is exactly the union) rebuilds it on both.
        # That halves per-box read volume AND halves the footprint each box's page cache has to
        # cover. Default off: a collective that only one rank reaches wedges the pair.
        # Set by the engine after construction; None means single-box.
        self.ep = None
        self.row_split = os.environ.get("DSV41_ENGRAM_ROW_SPLIT", "0") == "1"
        # ...but only for gathers big enough to pay for it. The split buys HALF THE ROWS at the
        # price of a stream-synchronising H2D and a cross-rank all-reduce, per layer. A prefill
        # chunk is 2048 x 24 ~ 49k rows, where halving is worth ~100 ms and the collective is
        # noise. A decode verify block is 6 x 24 = 144 rows, where it saves about a millisecond of
        # reading and pays a collective plus a sync that blocks the host until every graph queued
        # so far has finished -- the exact overlap this pipeline exists to preserve.
        self.split_min_rows = int(os.environ.get("DSV41_ENGRAM_SPLIT_MIN", "4096"))
        self.pool = ThreadPoolExecutor(max(threads, self.gather_threads if self.mmap_rows else 0))
        self.device = device
        self.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        # small process-local row cache (exact n-gram repeats inside a conversation hit here)
        self.cache: dict[int, bytes] = {}
        self.cache_max = 200_000
        # Pinned staging for the H2D (see to_device), OFF by default: it does make `to_device`
        # itself ~10x cheaper (0.27 s vs 3.16 s of a 200-token run, because the pageable copy
        # synchronises the stream and absorbs the queued graph work) but it does NOT make the run
        # faster -- measured 12.04-13.57 s of decode against 12.03-12.14 s pageable, i.e. the host
        # simply blocks somewhere else instead. Kept behind the switch rather than deleted.
        # Re-measured 2026-10-08 (ABBA, bit-identical output): 85.4 -> 85.0 ms/step prose,
        # 86.0 -> 86.1 code -- still no gain.
        self.pinned = os.environ.get("DSV41_ENGRAM_PINNED", "0") == "1"
        self._stage = [None, None]
        self._inv_stage = [None, None]
        self._ev = [None, None]
        self._cur = 0

    def _read_rows(self, ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), 264), np.uint8)
        for i, r in enumerate(ids):
            r = int(r)
            b = self.cache.get(r)
            if b is None:
                b = os.pread(self.fd, 256, self.w_off + r * 256) + os.pread(self.fd, 8, self.s_off + r * 8)
                if len(self.cache) < self.cache_max:
                    self.cache[r] = b
            out[i] = np.frombuffer(b, np.uint8)
        return out

    def _gather_rows(self, ids: np.ndarray) -> np.ndarray:
        cache = getattr(self, 'row_cache', None)
        if cache is not None and cache.enabled:
            return cache.gather(ids, self._gather_rows_uncached)
        return self._gather_rows_uncached(ids)

    def _gather_rows_uncached(self, ids: np.ndarray) -> np.ndarray:
        """Parallel memmap gather of the unique rows. Page faults do the I/O, so the threads are
        blocked in the kernel rather than fighting over the GIL.

        Throttling this was tried and measured harmful, twice: pacing it (a short yield every 16
        rows, per worker) gave 434 tok/s and deferring the whole read 400 ms gave 217, against
        1135 with neither. The read is not background work with slack to spare -- the next chunk
        cannot start its layer-1 engram until its rows land, so slowing the read by any mechanism
        directly delays the chunk waiting on it. What paid was removing work: splitting the rows
        across ranks halved per-box volume and took prefill 509 -> 1045 tok/s.
        """
        if self.direct_gather is not None and (self.direct or len(ids) > 384):
            return self.direct_gather.gather(np.ascontiguousarray(ids, dtype=np.int64))
        if self.native_gather is not None and len(ids) <= 384:
            return self.native_gather.gather(ids)
        out = np.empty((len(ids), 264), np.uint8)
        nt = max(1, self.gather_threads)
        # Task size has to follow the row count. This was `max(1024, len(ids) // nt + 1)`, and the
        # 1024 floor silently turned off the parallelism at exactly the size decode uses: a verify
        # block is T_VERIFY x 24 hashes ~ 144 unique rows, 144 <= 1024, so the whole gather ran
        # SERIALLY in the calling thread -- one synchronous page fault at a time, with the GPU
        # draining behind it. It cost 83 ms of a 196 ms decode step (42%). The floor also capped
        # medium gathers at 4 tasks, since it takes 65k rows before len//64 reaches 1024.
        # Measured on layer 1 (384M rows), random ids, cold:
        #     n=  144   serial  93-105 ms   64 threads  4.5-6.0 ms
        #     n= 1024   serial 459-675 ms   64 threads 10.1-13.2 ms
        #     n=49152   48 tasks    345 ms  64 threads      223 ms   (prefill: still faster)
        # MADV_WILLNEED on each row's page, then a serial gather, was also tried: fastest of all
        # at 144 rows (1.8 ms) but it is one syscall per page, so it collapses where prefill lives
        # -- 1367 ms at 49152 rows against 223. Not worth a size-dependent second path.
        step = max(1, (len(ids) + nt - 1) // nt)

        def one(i):
            sl = slice(i, min(i + step, len(ids)))
            rows = ids[sl]
            out[sl, :256] = self.w_mm[rows]
            out[sl, 256:] = self.s_mm[rows]

        starts = range(0, len(ids), step)
        # Below this the pool costs more than the faults it overlaps (~15 us to hand off a task).
        if len(ids) <= self.gather_min_parallel:
            for i in starts:
                one(i)
        else:
            list(self.pool.map(one, starts))
        return out

    def _splitting(self) -> bool:
        """Both ranks must agree, so this reads only constants -- never per-call state."""
        return bool(self.row_split and self.ep is not None and self.ep.active)

    def _split_call(self, n: int) -> bool:
        """Whether THIS gather of `n` unique rows is split across the ranks.

        Per-call, but still safe for the collective: `n` comes from np.unique over hashes both
        ranks compute identically, so both ranks always take the same branch. That is the whole
        invariant -- a collective one rank reaches and the other does not wedges the pair -- and
        it is why the threshold may only ever look at `n`, never at timing, load, or rank.
        """
        return self._splitting() and n >= self.split_min_rows

    def read_raw(self, hashes_np: np.ndarray):
        """Host-only part (safe in a background thread: no CUDA calls): NVMe reads of the unique rows.
        hashes_np: int64 [T, 24]. Returns (raw uint8 [n, 264], inv, shape).

        Under the EP2 row split this reads only this rank's share; the rest stays zero and
        to_device's all-reduce fills it in."""
        t1 = time.perf_counter()
        flat = hashes_np.reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        if self._split_call(n):
            # np.unique sorts, so both ranks see identical `uniq` and pick disjoint halves.
            mine = (uniq % self.ep.world) == self.ep.rank
            raw = np.zeros((n, 264), np.uint8)
            sel = uniq[mine]
            if len(sel):
                raw[mine] = (self._gather_rows(sel) if self.mmap_rows
                             else self._read_rows(sel))
            self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
            self.stats["rows"] += int(len(sel)); self.stats["calls"] += 1
            return raw, inv, hashes_np.shape
        if self.mmap_rows:
            raw = self._gather_rows(uniq)
        else:
            chunk = max(8, n // (self.pool._max_workers * 2) + 1)
            parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
            raw = np.concatenate(parts) if parts else np.empty((0, 264), np.uint8)
        self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
        self.stats["rows"] += int(n); self.stats["calls"] += 1
        return raw, inv, hashes_np.shape

    def to_device(self, raw, inv, shape) -> torch.Tensor:
        """GPU part (main thread): dequantize the rows and expand to [T, 24, 256] float32.

        The raw rows and the row-index vector go through PINNED staging buffers and are copied
        non-blocking. A plain `.to(device)` from pageable numpy memory synchronises the calling
        stream, which inside a decode step means blocking the host until every graph queued so far
        has finished -- exactly the overlap this pipeline exists to avoid. The buffers are reused
        across steps, so the previous step's copy out of them is waited on first (`_ev`); a whole
        step elapses in between, so that wait is free.
        """
        t0 = time.perf_counter()
        n = raw.shape[0]
        # Pinned staging and the row split are mutually exclusive per call, not per config: the
        # split needs an all-reduce on the device tensor, which the non_blocking path has not
        # queued yet. Since the split is now size-gated, decode (144 rows, never split) can take
        # the pinned path while prefill (49k rows, split) keeps the pageable one -- and decode is
        # exactly where the synchronising copy hurts, because there it blocks the host mid-step
        # with 40 layers of graphs already queued.
        staged = self.pinned and not self._split_call(n)
        if not staged:
            rawt = torch.from_numpy(raw).to(self.device)
            invt = torch.from_numpy(inv.reshape(-1)).to(self.device)
            if self._split_call(n):
                # Disjoint halves over a zero background: SUM is the union, bit-exact, no
                # overflow. Must run on every rank for the same (chunk, layer) or the pair wedges
                # -- `n` is derived from the hashes, which both ranks compute identically, so the
                # skip-on-empty is taken on both or neither.
                import torch.distributed as _dist
                from engine.collective_rails import group as collective_group
                _dist.all_reduce(rawt, op=_dist.ReduceOp.SUM, group=collective_group())
        else:
            i = self._cur
            self._cur ^= 1  # two buffers: the wait is on the copy from two calls ago
            if self._stage[i] is None or self._stage[i].shape[0] < n:
                cap = max(n, 4096)
                # inference_mode(False) around the ALLOCATION, not the copy. A tensor created
                # inside inference mode is an "inference tensor" and can never be mutated outside
                # it -- and this runs inside generate()'s inference_mode while the copy below can
                # reach it from elsewhere, so allocating here raised
                #   RuntimeError: Inplace update to inference tensor outside InferenceMode
                # on the first real use. A normal tensor can be written from either context, which
                # is what a buffer reused across calls has to be.
                with torch.inference_mode(False):
                    self._stage[i] = torch.empty(cap, 264, dtype=torch.uint8, pin_memory=True)
                    self._inv_stage[i] = torch.empty(cap * 64, dtype=torch.int64, pin_memory=True)
                self._ev[i] = torch.cuda.Event()
                self._ev[i].record()
            self._ev[i].synchronize()  # the last H2D out of these buffers is done
            self._stage[i][:n].copy_(torch.from_numpy(raw))
            ni = inv.size if hasattr(inv, "size") else len(inv)
            self._inv_stage[i][:ni].copy_(torch.from_numpy(inv.reshape(-1).astype(np.int64)))
            rawt = self._stage[i][:n].to(self.device, non_blocking=True)
            invt = self._inv_stage[i][:ni].to(self.device, non_blocking=True)
        vals = rawt[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(rawt[:, 256:].float() - 127.0)
        deq = (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2)
        out = deq[invt].view(shape[0], shape[1], 256)
        if staged:      # not `self.pinned`: a split call takes the pageable branch and has no `i`
            self._ev[i].record()
        self.stats["seconds"] += time.perf_counter() - t0
        return out

    def rows(self, hashes: torch.Tensor) -> torch.Tensor:
        """hashes: int64 [T, 24] -> float32 [T, 24, 256] dequantized rows."""
        t0 = time.perf_counter()
        flat = hashes.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        t1 = time.perf_counter()
        if self.mmap_rows:
            raw = self._gather_rows(uniq)
        else:
            chunk = max(8, n // (self.pool._max_workers * 2) + 1)
            parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
            raw = np.concatenate(parts) if parts else np.empty((0, 264), np.uint8)
        self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
        raw = torch.from_numpy(raw).to(self.device)
        vals = raw[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(raw[:, 256:].float() - 127.0)
        deq = (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2)  # [n, 256]
        out = deq[torch.from_numpy(inv).to(self.device)].view(hashes.shape[0], hashes.shape[1], 256)
        self.stats["rows"] += int(n); self.stats["seconds"] += time.perf_counter() - t0; self.stats["calls"] += 1
        return out


@contextmanager
def prefetch_rows(tables, pool, hashes, layer_ids):
    """Overlap a chunk's host row reads with its GPU layers, joining on every exit."""
    host_hashes = hashes.cpu().numpy()
    futures = {}
    try:
        for li, layer in enumerate(layer_ids):
            futures[layer] = pool.submit(tables[layer].read_raw, host_hashes[:, li, :])

        def rows(layer, _hashes):
            return tables[layer].to_device(*futures[layer].result())

        yield rows
    finally:
        # Workers must finish before the next request resets counters/caches,
        # including when a forward fails before reaching the second table.
        wait(list(futures.values()))


class EngramReadAhead:
    """Cross-chunk read-ahead for the engram tables.

    `prefetch_rows` submits a chunk's reads at the top of that chunk's OWN forward, and
    `engram_layer_ids` starts at layer 1 -- so only layer 0's GPU work, ~25 ms of a ~1.5 s
    chunk, is available to hide what can be ~1 s of random NVMe reads. Prefill over text
    with novel n-grams therefore alternates roughly 1 s busy / 1 s idle, and the effect is
    invisible to any benchmark whose prompt repeats (a repeated template collapses to a
    handful of unique hashes, which the row cache serves for free).

    The hashes depend on the token ids alone, so the whole prompt can be hashed up front and
    chunk k+1's reads submitted while chunk k is still on the GPU. `depth` chunks are kept in
    flight, which both hides the latency and keeps the NVMe queue deep enough to be worth the
    32-thread pool.
    """

    def __init__(self, tables, pool, layer_ids, hashes_np, spans, depth: int = 2):
        self.tables, self.pool, self.layer_ids = tables, pool, list(layer_ids)
        self.hashes = hashes_np
        self.depth = max(1, depth)
        self.total = hashes_np.shape[0]
        self.spans = [(int(s), int(e)) for s, e in spans]
        self.start = self.spans[0][0] if self.spans else 0
        # A normal prefill tiles [0, total). Exact-prefix reuse deliberately supplies only the
        # uncached suffix, so its first span may be > 0 while hashes still contains the full
        # sequence (n-grams at the boundary need the prefix tokens). In both cases the supplied
        # spans must be one contiguous tiling through ``total``.
        if (not self.spans or self.start < 0 or self.spans[-1][1] != self.total
                or any(s >= e for s, e in self.spans)
                or any(e != ns for (_s, e), (ns, _ne) in zip(self.spans, self.spans[1:]))):
            raise ValueError(f"read-ahead spans must tile [{self.start}, {self.total}): {self.spans}")
        self.span_index = {s: i for i, (s, _e) in enumerate(self.spans)}
        self.futures: dict[int, dict] = {}

    def _submit(self, span_i: int):
        if span_i >= len(self.spans):
            return
        s, e = self.spans[span_i]
        if s in self.futures:
            return
        hs = self.hashes[s:e]
        self.futures[s] = {L: self.pool.submit(self.tables[L].read_raw, hs[:, li, :])
                           for li, L in enumerate(self.layer_ids)}

    def rows_for(self, s: int):
        """Ensure span `s` and the next `depth-1` spans are in flight.

        Image spans move prefill boundaries away from fixed multiples of MAX_CHUNK. Scheduling by
        those real spans is essential: otherwise a short image-bearing chunk can receive the rows
        for a full ordinary chunk and fail when engram_forward reshapes them.
        """
        try:
            span_i = self.span_index[s]
        except KeyError as exc:
            raise ValueError(f"{s} is not a configured read-ahead span start") from exc
        for k in range(self.depth):
            self._submit(span_i + k)
        futs = self.futures[s]
        expected = self.spans[span_i][1] - s

        def get_rows(layer, hashes):
            if hashes.shape[0] != expected:
                raise RuntimeError(
                    f"engram read-ahead span {s} has {expected} rows, forward requested "
                    f"{hashes.shape[0]}")
            rows = self.tables[layer].to_device(*futs[layer].result())
            if rows.shape[0] != expected:
                raise RuntimeError(
                    f"engram read-ahead span {s} produced {rows.shape[0]} rows, expected {expected}")
            return rows

        return get_rows

    def done(self, s: int):
        """Release chunk `s`'s futures once its forward has consumed them."""
        self.futures.pop(s, None)

    def close(self):
        # Workers must finish before the next request resets counters/caches, including when a
        # forward raises before reaching the second table.
        for futs in self.futures.values():
            wait(list(futs.values()))
        self.futures.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_hash_state(model_dir: str, tokenizer, max_seq: int, device: str):
    """The reference NgramHashState (engram.py from the checkpoint's inference/ folder)."""
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import engram as E  # noqa: E402
    cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))

    class A:
        engram_layer_ids = tuple(cfg["engram_layer_ids"])
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]
        engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
        max_batch_size = 1
        max_seq_len = max_seq + 16

    layout = E.EngramLayout.from_args(A)
    st = E.NgramHashState(A, layout, tokenizer)
    return st.to(device)
