"""dist.py -- EP2 glue: expert-parallel across two ranks, nothing else.

The design is in docs/dual-spark-plan.md ("EP2 with replicated attention"): every weight and
every computation EXCEPT the routed experts is bit-identically replicated on both nodes, and
the only new distributed operation in the whole engine is one fp32 all-reduce of the routed
MoE partial sum per main-model layer.

Ownership is `expert_id % world_size == rank` -- interleaved, not contiguous, because the
warm-start ranking (results/trace-*/stats/coverage.json) is a global order and parity
splitting keeps both arenas equally good without either rank knowing the other's set.

world_size <= 1  ->  EPDistributed() is inert: `active` is False, no process group is
created, and every call site below is skipped. The single-box path is untouched by this
module, on purpose.

NOT bit-identical to single-node by construction: the 6-term routed sum becomes
(owned subset) + (complement) added once in fp32, which rounds differently than one 6-term
sum. The correctness bar for dual mode is therefore the repo's existing gates -- spec
losslessness (drafted vs greedy identical ON the same config) and held-out NLL within
+/-0.01 nats of the single-node run -- not bit equality. See docs/dual-spark-plan.md Phase 2.
"""

from __future__ import annotations

import datetime
import os

import torch
from engine import collective_rails

try:
    import torch.distributed as dist
    _HAVE_DIST = True
except Exception:  # torch built without distributed
    dist = None
    _HAVE_DIST = False


def _apply_default_net_env() -> None:
    """Pin NCCL/Gloo to the CX7 200G link when the caller did not already choose.

    Without this, NCCL may bind the WiFi/LAN NIC (192.168.x) or pick the wrong RoCE
    port and hang at init_process_group with no useful log -- which is exactly the
    Gate G0 failure mode we hit on first bring-up.
    """
    # The interface name is site-specific: `enp1s0f1np1` is the direct-attach CX7 port on the
    # pair this was developed on, and hardcoding it as a default meant any other machine silently
    # got a name it does not have -- which NCCL reports as a hang at init, not as a bad interface.
    # Set NCCL_SOCKET_IFNAME in .env (scripts/dual-up.sh forwards it to both ranks); the default
    # is only applied when an interface by that name actually exists here.
    _iface = os.environ.get("NCCL_SOCKET_IFNAME") or os.environ.get("GLOO_SOCKET_IFNAME")
    if not _iface and os.path.isdir("/sys/class/net/enp1s0f1np1"):
        _iface = "enp1s0f1np1"
    if _iface:
        os.environ.setdefault("NCCL_SOCKET_IFNAME", _iface)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _iface)
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    # Don't force NCCL_NET=IB: on some DGX OS builds the RoCE path needs explicit HCA;
    # leaving it unset lets NCCL probe. Set NCCL_IB_HCA=mlx5_0 if probe fails.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")


class EPDistributed:
    """Rank/world bookkeeping + the one collective the EP2 design needs.

    Constructed once per engine; read-only afterwards. `owns(layer, expert)` is the single
    source of truth for who loads, keeps and computes what; ExpertStore consults it in
    resolve() and warm_start() (non-owned experts get the arena's zero-filled null slot and
    never enter the LRU, the transient ring, or the hit/miss stats).
    """

    def __init__(self, rank: int | None = None, world_size: int | None = None):
        # torchrun-style env is the default; explicit args (or launcher flags) win.
        self.rank = int(os.environ.get("RANK", 0)) if rank is None else int(rank)
        self.world = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else int(world_size)
        self.active = self.world > 1
        self.control_value = 0   # rank 0's per-step values from the last control() broadcast
        self.control_flag = 0
        self.tensor_parallel = os.environ.get('DSV41_TP_EXPERTS', '0') == '1'
        self.prefill_dual_rail = os.environ.get('DSV41_PREFILL_DUAL_RAIL', '0') == '1'
        if self.tensor_parallel and self.world != 2:
            raise ValueError('DSV41_TP_EXPERTS requires two ranks')
        if self.active and not _HAVE_DIST:
            raise RuntimeError("WORLD_SIZE > 1 but torch.distributed is unavailable in this torch build")
        if self.active:
            assert 0 <= self.rank < self.world, f"RANK {self.rank} outside 0..{self.world - 1}"

    # ------------------------------------------------------------- lifecycle
    def init(self, device: str | torch.device = "cuda"):
        """Init the process group. Call BEFORE the arena is allocated: NCCL pins its buffers
        out of the same unified pool, so sizing an arena around a not-yet-initialized PG
        overcounts what the arena will actually have.

        The mixed backend gives us NCCL for the CUDA all-reduce and Gloo for object
        broadcast (request metadata from rank 0 to rank 1) without a second network stack.
        """
        if not self.active:
            return
        if dist.is_initialized():
            if self.prefill_dual_rail and collective_rails._prefill_group is None:
                raise RuntimeError('prefill dual rail must initialize before any external CUDA process group')
            return
        _apply_default_net_env()
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev.index or 0)
        # Mixed backend: CUDA tensors -> NCCL, CPU/object collectives -> Gloo.
        # broadcast_object_list MUST have a Gloo group; pure NCCL hangs or errors on it.
        backend = "cpu:gloo,cuda:nccl" if dev.type == "cuda" else "gloo"
        timeout_s = float(os.environ.get("DSV41_DIST_TIMEOUT_S", "600"))
        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.world,
            timeout=datetime.timedelta(seconds=timeout_s),
        )
        collective_rails.initialize(self.prefill_dual_rail, dev,
                                   datetime.timedelta(seconds=timeout_s))

    def destroy(self):
        if self.active and _HAVE_DIST and dist.is_initialized():
            collective_rails.destroy()
            dist.destroy_process_group()

    # ------------------------------------------------------------- ownership
    def owns(self, layer: int, expert: int) -> bool:
        return self.tensor_parallel or expert % self.world == self.rank

    def owned_mask(self, expert_ids):
        """numpy or tensor of expert ids -> bool mask of the ones THIS rank computes."""
        if isinstance(expert_ids, torch.Tensor):
            if self.tensor_parallel:
                return torch.ones_like(expert_ids, dtype=torch.bool)
            return expert_ids.remainder(self.world) == self.rank
        import numpy as np
        if self.tensor_parallel:
            return np.ones_like(np.asarray(expert_ids), dtype=bool)
        return (np.asarray(expert_ids) % self.world) == self.rank

    # ------------------------------------------------------------- combine
    def combine(self, partial: torch.Tensor) -> torch.Tensor:
        """In-place all-reduce (SUM) of this rank's fp32 routed partial. Both ranks leave
        with identical bits: NCCL guarantees the same reduction result on every rank for a
        given collective, which is what keeps their logits (and therefore their accept/reject
        and sampling decisions) in lockstep without any per-step token broadcast."""
        assert partial.dtype == torch.float32, "combine in fp32 -- the single-node k-sum is fp32 (model.moe)"
        if not partial.is_cuda:
            # Should not happen on the serving path; keep the assert loud in debug.
            pass
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=collective_rails.group())
        return partial

    def broadcast_obj(self, payload):
        """rank 0 -> all, for any picklable object. Used for state that MUST be identical on both
        ranks and is derived from something only rank 0 can see."""
        if not self.active:
            return payload
        obj = [payload if self.rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def gather_objects(self, payload):
        """All-rank agreement for optional local resources (e.g. persistent prefix files)."""
        if not self.active:
            return [payload]
        gathered = [None] * self.world
        dist.all_gather_object(gathered, payload)
        return gathered

    # ------------------------------------------------------- request control
    def broadcast_request(self, payload):
        """rank 0 -> all: the request metadata that starts a replicated generate().
        Everything after this point runs from the shared parameters -- except the decode
        loop's stop decision, which rank 0 owns and hands over step by step in control().

        Uses broadcast_object_list over the Gloo half of the mixed process group.
        """
        if not self.active:
            raise RuntimeError("broadcast_request on an inert EPDistributed")
        obj = [payload if self.rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    # ------------------------------------------------------- step control
    def control(self, keep_going: bool, value: int = 0, flag: int = 0) -> bool:
        """rank 0 -> all, once per decode step: does the loop run another iteration?

        The lockstep argument in this module's docstring covers *what* the two ranks compute
        (identical logits => identical tokens => identical collective shapes). It does NOT cover
        *when rank 0 stops*, and that is not a theoretical gap:

          * a text stop sequence ("\\n\\nUser:") is detected by server/app.py on the detokenized
            stream, which rank 1 never sees;
          * a client disconnect or any server-side error closes the generator, raising
            GeneratorExit into the decode loop at its pending `yield`;
          * max_tokens is re-clamped server-side against the burst the engine already emitted.

        In every one of those, rank 0 leaves the loop while rank 1's own `while` is still true.
        Rank 1 then issues the next step's 40 all-reduces into a communicator whose peer has
        moved on to the next request -- which is not a wrong answer but a wedged pair, and the
        damage lands on whoever's request comes next.

        So the loop condition itself is broadcast: rank 0 evaluates it, rank 1 obeys it and never
        evaluates its own. A CPU tensor keeps this on the Gloo half of the mixed group -- no CUDA
        sync, no interleaving with the in-flight NCCL combines, just the host bool the loop needs.

        The same message carries rank 0's per-step decisions: `value` (the speculative depth,
        DSV41_BLOCK_DYNAMIC) and `flag` (run a decode-time expert adaptation pass before this
        step); every rank reads them back from `self.control_value` / `self.control_flag`. It is
        always three ints, for every caller including release_peer, so a rank blocked in a
        decode loop's control() can never be paired with a broadcast of a different size.
        """
        if not self.active:
            self.control_value, self.control_flag = int(value), int(flag)
            return keep_going
        msg = torch.tensor([1 if keep_going else 0, int(value), int(flag)], dtype=torch.int32)
        dist.broadcast(msg, src=0)
        self.control_value, self.control_flag = int(msg[1]), int(msg[2])
        return bool(msg[0])

    def release_peer(self) -> None:
        """Best-effort 'stop' for the abort paths (GeneratorExit, exceptions, shutdown).

        Called from an `except`/`finally` on rank 0, where rank 1 is already blocked in the
        matching control() of the step that will never run. Failures are swallowed on purpose:
        this runs while another exception is propagating, and a dead peer is the launcher's
        problem, not this frame's.
        """
        if not self.active or self.rank != 0:
            return
        try:
            self.control(False)
        except Exception:
            pass
