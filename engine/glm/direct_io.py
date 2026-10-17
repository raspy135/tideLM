"""O_DIRECT reads into pooled, page-aligned buffers lent to the tensors that view them.

One 7.1 MB O_DIRECT preadv took 0.83 ms (8.5 GB/s) on this box where a buffered read in 16
parallel 256 KB pieces took 3.1 ms; a fresh 7 MB buffer costs ~3.7 ms of page faults, so buffers
are pooled. A buffer is lent (Lease) to the numpy/torch views of one read and goes back to the
pool only when the last view dies -- the buffer protocol keeps it referenced until then -- so it is
never refilled under a live view, however long a caller holds the tensors.
"""

from __future__ import annotations

import mmap
import os
import threading

ALIGN = 4096


def align_up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class Lease:
    def __init__(self, mm, pool):
        self.mm, self.pool = mm, pool

    def __buffer__(self, flags):
        return memoryview(self.mm)

    def __release_buffer__(self, view):
        view.release()

    def __del__(self):
        try:
            self.pool.give(self.mm)
        except Exception:               # interpreter shutdown
            pass


class BufPool:
    """Up to `keep` idle `size`-byte buffers (faulted in) for reuse; larger requests get a buffer
    of their own, freed with its lease."""

    def __init__(self, size: int, keep: int):
        self.size, self.keep, self.free, self.lock = size, keep, [], threading.Lock()

    def lease(self, n: int) -> Lease:
        mm = None
        if n <= self.size:
            with self.lock:
                if self.free:
                    mm = self.free.pop()
        return Lease(mm if mm is not None else mmap.mmap(-1, max(n, self.size if n <= self.size else n)), self)

    def give(self, mm):
        if len(mm) == self.size:
            with self.lock:
                if len(self.free) < self.keep:
                    self.free.append(mm)
                    return
        try:
            mm.close()
        except BufferError:             # a view still alive somewhere: let it be collected
            pass


def pread_direct(fd: int, view: memoryview, off: int, need: int) -> int:
    """O_DIRECT-read the aligned `view` at aligned file offset `off` until `need` bytes have come
    (the aligned length may run past the end of the file); returns the bytes read."""
    got = 0
    while got < need:
        r = os.preadv(fd, [view[got:]], off + got)
        if r <= 0:
            break
        got += r
    return got
