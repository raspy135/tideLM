"""Bounded CPU reads for an already agreed expert-swap plan.

This helper neither chooses experts nor touches arena slots, CUDA, routing masks or
LRU state. A caller may stage a frozen plan while the old resident set keeps serving,
then install its records at the normal agreed boundary. Each installed record's
release admits one more read, keeping the complete pipeline bounded.

Records own their bytes. ``take`` transfers ownership to the caller, which must
release each record *after* installation has finished reading it. Ready, running,
cancelled and borrowed records all share one byte budget, including across plans.

Experimental only; not connected to serving. On 32 real checkpoint experts with
four retained payloads and two readers, owning-copy staging took 104.23 ms versus
77.53 ms for the existing serial leased reader (+34.4%, three samples per arm,
no simulated install delay).
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field

VERSION = 1


class StaleSwapPlan(RuntimeError):
    """A staged plan cannot be applied to the caller's current resident set."""


def _plan(swaps):
    # Include the complete plan, order and gains: staging must not quietly serve a
    # different planner decision merely because the incoming expert keys match.
    return tuple((int(layer), int(out), int(incoming), float(gain))
                 for layer, out, incoming, gain in swaps)


def _payload_bytes(views):
    if len(views) != 6:
        raise ValueError("staged expert must contain six packed tensors")
    return sum(int(v.numel()) * int(v.element_size()) if hasattr(v, "numel")
               else memoryview(v).nbytes for v in views)


class StagedExpert:
    """A CPU expert payload and the reservation that keeps its lifetime bounded."""

    def __init__(self, key, views, release):
        self.key = key
        self._views = tuple(views)
        self._release = release
        self._lock = threading.Lock()

    @property
    def views(self):
        with self._lock:
            if self._views is None:
                raise RuntimeError("staged expert has been released")
            return self._views

    def release(self):
        with self._lock:
            if self._views is None:
                return
            self._views = None
            release, self._release = self._release, None
        release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


@dataclass
class _Entry:
    key: tuple
    cancelled: threading.Event
    future: object = None
    claimed: bool = False
    on_release: object = None
    lock: object = field(default_factory=threading.Lock)

    def discard(self, _future=None):
        # Future completion and cancellation can race with take(). Claimed records
        # belong to the installer and must outlive its final H2D synchronization.
        with self.lock:
            if (not self.cancelled.is_set() or self.claimed or
                    self.future is None or not self.future.done()):
                return
            try:
                record = self.future.result()
            except BaseException:
                return
            self.claimed = True
        record.release()


class StagedSwapPlan:
    """Immutable plan identity plus a bounded set of local pending CPU reads."""

    def __init__(self, swaps, generation, entries, cancelled, submit, capacity):
        self.swaps = swaps
        self.generation = int(generation)
        self._entries = entries
        self._cancelled = cancelled
        self._submit = submit
        self._lock = threading.Lock()
        self.initial_keys = tuple(entries)[:capacity]
        self._pending = deque(tuple(entries)[capacity:])

    @property
    def keys(self):
        return tuple(self._entries)

    def validate(self, swaps, generation):
        if (self._cancelled.is_set() or int(generation) != self.generation or
                _plan(swaps) != self.swaps):
            self.cancel()
            raise StaleSwapPlan("staged swap plan or resident generation is stale")

    def wait(self, swaps, generation):
        """Surface initial-window read failures before any arena slot is changed.

        The engine must exchange success/failure on both ranks unconditionally
        before installation. This class deliberately performs no collectives.
        """
        self.validate(swaps, generation)
        error = None
        for key in self.initial_keys:
            entry = self._entries[key]
            try:
                entry.future.result()
            except Exception as exc:
                if error is None:
                    error = exc
        self.validate(swaps, generation)
        if error is not None:
            raise RuntimeError("expert swap staging read failed") from error
        return len(self.initial_keys)

    def take(self, key, swaps, generation):
        """Claim an owned expert in plan order; peers' keys return None."""
        self.validate(swaps, generation)
        entry = self._entries.get(tuple(key))
        if entry is None:
            return None
        if entry.future is None:
            raise RuntimeError("consume staged experts in plan order and release installed records")
        record = entry.future.result()
        with entry.lock:
            self.validate(swaps, generation)
            if entry.claimed:
                raise RuntimeError(f"staged expert already consumed: {key}")
            entry.claimed = True
        return record

    def advance(self):
        """Release one payload before admitting the next read in plan order."""
        with self._lock:
            if self._cancelled.is_set() or not self._pending:
                return
            entry = self._entries[self._pending.popleft()]
        self._submit(entry)

    def cancel(self):
        """Drop unclaimed records; running reads release themselves on completion."""
        self._cancelled.set()
        for entry in self._entries.values():
            if entry.future is not None:
                entry.future.cancel()
            entry.discard()


class SwapStager:
    """Pipeline a frozen local plan under a strict payload-byte budget.

    ``read_expert(layer, expert)`` must return six owning CPU byte
    tensors, as ExpertStore.read_expert does. No leased staging-buffer aliases
    may escape the reader. The budget covers retained expert payloads, not the
    ExpertStore's existing temporary I/O buffers or Python object bookkeeping.
    """

    def __init__(self, read_expert, *, bytes_per_expert, max_bytes, workers=2):
        self.bytes_per_expert = int(bytes_per_expert)
        self.max_bytes = int(max_bytes)
        if self.bytes_per_expert <= 0 or self.max_bytes < self.bytes_per_expert:
            raise ValueError("swap staging budget must fit at least one expert")
        if int(workers) <= 0:
            raise ValueError("swap staging needs at least one worker")
        self.capacity = self.max_bytes // self.bytes_per_expert
        self._reader = read_expert
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._pool = ThreadPoolExecutor(int(workers), thread_name_prefix="swap-stage")
        self._lock = threading.Lock()
        self._active = None
        self._closed = False
        self._reserved = 0
        self.peak_reserved_bytes = 0

    @property
    def reserved_bytes(self):
        with self._lock:
            return self._reserved

    def _release(self):
        with self._lock:
            self._reserved -= self.bytes_per_expert
        self._slots.release()

    def _read(self, entry):
        while not entry.cancelled.is_set():
            if self._slots.acquire(timeout=0.02):
                break
        else:
            raise CancelledError()
        with self._lock:
            self._reserved += self.bytes_per_expert
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, self._reserved)
        try:
            if entry.cancelled.is_set():
                raise CancelledError()
            views = self._reader(*entry.key)
            if _payload_bytes(views) != self.bytes_per_expert:
                raise ValueError("expert reader returned an unexpected payload size")
            def release():
                self._release()
                entry.on_release()
            return StagedExpert(entry.key, views, release)
        except BaseException:
            self._release()
            raise

    def start(self, swaps, generation, *, world=1, rank=0):
        """Stage only owned incoming experts from the exact agreed plan prefix."""
        swaps = _plan(swaps)
        world, rank = int(world), int(rank)
        if world < 1 or not 0 <= rank < world:
            raise ValueError("invalid swap-staging ownership")
        keys = []
        seen = set()
        for layer, outgoing, incoming, _gain in swaps:
            if outgoing % world != incoming % world:
                raise ValueError("swap staging plan crosses ownership classes")
            key = layer, incoming
            if key in seen:
                raise ValueError("swap staging plan promotes an expert twice")
            seen.add(key)
            if incoming % world == rank:
                keys.append(key)
        # No lock is held across read completion. Cancellation also covers already
        # running reads from a superseded plan, whose reservations remain counted.
        with self._lock:
            if self._closed:
                raise RuntimeError("swap stager is closed")
            old = self._active
            cancelled = threading.Event()
            entries = {key: _Entry(key, cancelled) for key in keys}
            plan = StagedSwapPlan(swaps, generation, entries, cancelled, self._submit, self.capacity)
            for entry in entries.values():
                entry.on_release = plan.advance
            self._active = plan
        if old is not None:
            old.cancel()
        for key in plan.initial_keys:
            self._submit(entries[key])
        return plan

    def _submit(self, entry):
        with entry.lock:
            if entry.cancelled.is_set():
                return
            entry.future = self._pool.submit(self._read, entry)
        entry.future.add_done_callback(entry.discard)

    def cancel(self):
        with self._lock:
            active, self._active = self._active, None
        if active is not None:
            active.cancel()

    def close(self):
        with self._lock:
            self._closed = True
        self.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
