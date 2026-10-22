"""Dense TP primitives. Keep the checkpoint's rounding AFTER each complete linear.

Attention shards heads/groups, not queries or shared compressed KV. Output projections
gather inputs and disjoint results by default. The legacy reduction path remains for A/B
tests. The vocabulary head gathers disjoint logits, without sums.
"""
import os
import torch
import torch.distributed as dist
from engine.collective_rails import group as collective_group
from engine import comm


TP_DRAFT_HEAD_VERSION = 1


def tp_draft_head_enabled():
    """Opt in separately: DRAFT_HEAD_FMT historically had no effect with vocab TP."""
    value = os.environ.get('DSV41_TP_DRAFT_HEAD', '0')
    if value not in ('0', '1'):
        raise ValueError('DSV41_TP_DRAFT_HEAD must be 0 or 1')
    return value == '1'


def make_tp_draft_head(head, *, load_mtp=True, source=None):
    """Quantize this rank's draft-only vocab shard; retain the verifier and gather layout.

    The separate opt-in preserves the old TP behavior when DRAFT_HEAD_FMT is left
    at its low-level FP8 default. The format decision remains R.make_draft_head's:
    off, an already quantized verifier and the FP32 reference do not allocate.
    No collective runs while constructing the copy. At decode, both heads use
    VocabParallelHead.tp_logits, including its unconditional disjoint all-gather.
    """
    if not tp_draft_head_enabled() or not load_mtp:
        return None
    if not isinstance(head, VocabParallelHead):
        raise TypeError('TP draft head requires a VocabParallelHead verifier')
    import v41_ref as R
    if source is not None and not isinstance(head.local, torch.Tensor):
        # The verifier shard is packed (DSV41_HEAD_KERNEL=packed) and has no tensor to quantize;
        # build the draft copy from the BF16 shard it was packed from.
        fmt = R.draft_head_fmt()
        local = R._head_in_format(source, fmt) if fmt in ('fp8', 'fp4') else None
    else:
        local = R.make_draft_head(head.local)
    return None if local is None else VocabParallelHead(local, head.world)


def draft_head_bytes(head):
    """Additional resident tensor bytes of a separate head, including quantization scales."""
    if head is None:
        return 0
    local = head.local if isinstance(head, VocabParallelHead) else head
    if hasattr(local, 'stored_bytes'):
        return local.stored_bytes
    tensors = ((local,) if isinstance(local, torch.Tensor)
               else (local.w, local.s))
    return sum(t.numel() * t.element_size() for t in tensors)


def shard(weight, dim, rank, world):
    if isinstance(weight, torch.Tensor):
        if weight.shape[dim] % world:
            raise ValueError('unaligned tensor shard')
        return weight.chunk(world, dim=dim)[rank].clone().contiguous()
    return weight.shard(dim, rank, world)


class RowParallelWeight:
    def __init__(self, local):
        self.local = local
        self.shape = local.shape

    def tp_linear_act_qdq(self, x):
        """qlinear(x, self) with act_qdq_fp8 applied inside the fp8 GEMM. This rank's x is the
        K shard the unfused path quantizes, and shard widths are multiples of 32, so the
        quantization groups are the same ones. Decode-sized bf16 x only (v41_ref gates it)."""
        return self.tp_linear(x, act_qdq=True)

    def tp_linear(self, x, act_qdq: bool = False):
        import decode_rows
        import v41_ref as R
        from fp8_linear import FP8Weight, fp8_linear
        x = x.to(torch.bfloat16)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if act_qdq:
            partial = fp8_linear(flat, self.local, out_dtype=torch.float32, act_qdq=True)
        elif isinstance(self.local, FP8Weight) and flat.size(0) <= decode_rows.limit():
            partial = fp8_linear(flat, self.local, out_dtype=torch.float32)
        else:
            weight = self.local if isinstance(self.local, torch.Tensor) else self.local.dequant()
            # Pad small BF16 tensor GEMMs just as R.mm does, preserving decode block invariance.
            rows = flat.size(0)
            if 0 < rows < R.MM_TILE:
                flat = torch.cat((flat, flat.new_zeros(R.MM_TILE - rows, flat.size(1))))
            partial = torch.mm(flat, weight.t(), out_dtype=torch.float32)[:rows].contiguous()
        dist.all_reduce(partial, group=collective_group())
        return partial.to(torch.bfloat16).view(*shape[:-1], self.shape[0])


class OutputParallelWeight:
    """Gather BF16 inputs, compute complete dot products for disjoint output rows.

    Keeps half the weight bytes without reducing separately accumulated K partitions.
    Both collectives concatenate values; neither introduces a floating-point sum.
    """
    def __init__(self, local, world=2):
        if world != 2 or local.shape[1] % world:
            raise ValueError('output-parallel linear requires two aligned input shards')
        self.local, self.world = local, world
        self.shape = (local.shape[0] * world, local.shape[1] // world)

    def tp_linear_act_qdq(self, x):
        """qlinear(x, self) with act_qdq_fp8 applied inside the fp8 GEMM, after the gather. The
        quantization is per row and per 32-wide K group, and each rank's shard is a whole number
        of groups, so quantizing the gathered row equals gathering quantized shards."""
        return self.tp_linear(x, act_qdq=True)

    def tp_linear(self, x, act_qdq: bool = False):
        import v41_ref as R
        from fp8_linear import fp8_linear
        shape = x.shape
        flat = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
        rows, width = flat.shape
        gathered = torch.empty((self.world * rows, width), dtype=flat.dtype, device=flat.device)
        comm.all_gather_fast(gathered, flat, group=collective_group())
        full = gathered.view(self.world, rows, width).transpose(0, 1).reshape(rows, self.world * width)
        local = (fp8_linear(full, self.local, act_qdq=True) if act_qdq
                 else R.mm(full, self.local)).contiguous()
        output = torch.empty((self.world * rows, local.shape[-1]), dtype=local.dtype, device=local.device)
        comm.all_gather_fast(output, local, group=collective_group())
        return output.view(self.world, rows, local.shape[-1]).transpose(0, 1).reshape(
            *shape[:-1], self.shape[0])


class ColumnParallelWeight:
    """Replicated input, disjoint output rows a rank, the outputs gathered.

    No sum anywhere: fp8_linear partitions output columns only, so each gathered value is the
    dot product the unsplit weight computes. Decode-sized calls only. At prefill sizes cuBLAS
    picks a different algorithm for the narrower N (17 and 64 rows of a 5120x15360 GEMM differ
    in their bits split vs whole; 256+ rows matched, 2026-10-22), so prefill keeps the full weight.
    """
    def __init__(self, local, world=2):
        if world != 2:
            raise ValueError('column-parallel linear supports two ranks')
        self.local, self.world = local, world
        self.shape = (local.shape[0] * world, local.shape[1])

    def tp_linear_act_qdq(self, x):
        return self.tp_linear(x, act_qdq=True)

    def tp_linear(self, x, act_qdq: bool = False):
        import decode_rows
        from fp8_linear import fp8_linear
        shape = x.shape
        flat = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
        rows = flat.size(0)
        if rows > decode_rows.limit():
            raise ValueError(f'column-parallel linear is decode-only, got {rows} rows')
        local = fp8_linear(flat, self.local, act_qdq=act_qdq).contiguous()
        output = torch.empty((self.world * rows, local.shape[-1]), dtype=local.dtype, device=local.device)
        comm.all_gather_fast(output, local, group=collective_group())
        return output.view(self.world, rows, local.shape[-1]).transpose(0, 1).reshape(
            *shape[:-1], self.shape[0])


def tp_draft_dense_enabled():
    value = os.environ.get('DSV41_TP_DRAFT_DENSE', '0')
    if value not in ('0', '1'):
        raise ValueError('DSV41_TP_DRAFT_DENSE must be 0 or 1')
    return value == '1'


def shard_draft_dense(weight, rank, world):
    """DSV41_TP_DRAFT_DENSE for one DSpark block: its shared expert in the backbone's output
    layout (w1/w3 output rows, w2 an OutputParallelWeight), and block 0's main_proj given a
    decode-only ColumnParallelWeight over a row VIEW of its rows -- the full matrix stays for
    the prefill seed, so this adds no memory there and frees 17.7 MB a block of shared expert.

    The draft read both in full on both ranks: per step, 3 x 35 MB of shared expert and 79 MB
    of main_proj (fastdecode._draft, _seed_draft_rings). Every value stays a complete dot
    product; only which rank computes it changes."""
    from fp8_linear import FP8Weight
    if world != 2 or not 0 <= rank < world:
        raise ValueError('draft dense TP requires two ranks')
    if os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output') != 'output':
        raise ValueError('draft dense TP requires DSV41_TP_LINEAR_LAYOUT=output')
    if any(type(getattr(weight, n)) is not FP8Weight for n in ('sh_w1', 'sh_w2', 'sh_w3')):
        raise ValueError('draft dense TP requires the native FP8 shared expert (DSV41_DENSE_FP4 '
                         'without "shared")')
    weight.sh_w1 = shard(weight.sh_w1, 0, rank, world)
    weight.sh_w3 = shard(weight.sh_w3, 0, rank, world)
    weight.sh_w2 = OutputParallelWeight(shard(weight.sh_w2, 0, rank, world), world)
    main = getattr(weight, 'main_proj', None)
    if main is not None:
        if type(main) is not FP8Weight or main.N % (32 * world):
            raise ValueError('draft dense TP requires the native FP8 main_proj')
        n = main.N // world
        weight._main_proj_tp = ColumnParallelWeight(
            FP8Weight(main.w[rank * n:(rank + 1) * n], main.s[rank * n // 32:(rank + 1) * n // 32]), world)


def shard_attention(weight, args, rank, world):
    from fp8_linear import FP8GroupedWeight
    from fp4_linear import FP4GroupedWeight
    if world != 2 or not 0 <= rank < world:
        raise ValueError('attention TP currently requires two ranks')
    if args.n_heads % world or args.o_groups % world:
        raise ValueError('attention heads/groups must divide TP world')
    weight.tp_heads, weight.tp_groups = args.n_heads // world, args.o_groups // world
    weight.wq_b = shard(weight.wq_b, 0, rank, world)
    weight.attn_sink = shard(weight.attn_sink, 0, rank, world)
    wo = weight.wo_a
    if isinstance(wo, FP8GroupedWeight):
        rows = wo.G * wo.R // world
        lo, hi = rank * rows, (rank + 1) * rows
        weight.wo_a = FP8GroupedWeight(wo.w[lo:hi].clone(), wo.s[lo//32:hi//32].clone(),
                                      wo.G // world, wo.R)
    elif isinstance(wo, FP4GroupedWeight):
        rows = wo.G * wo.R // world
        lo, hi = rank * rows, (rank + 1) * rows
        # fp4 scales are one per row, not a 32x32 block, so both tables slice the row range
        weight.wo_a = FP4GroupedWeight(wo.w[lo:hi].clone(), wo.s[lo:hi].clone(),
                                      wo.G // world, wo.R, wo.K)
    elif isinstance(wo, torch.Tensor):
        weight.wo_a = shard(wo, 0, rank, world)
    else:
        raise ValueError('attention TP supports native FP8, FP4 or BF16 wo_a only')
    layout = os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output')
    if layout == 'output':
        weight.wo_b = OutputParallelWeight(shard(weight.wo_b, 0, rank, world), world)
    elif layout == 'intermediate':
        weight.wo_b = RowParallelWeight(shard(weight.wo_b, 1, rank, world))
    else:
        raise ValueError('TP linear layout must be intermediate or output')


class FeatureParallelEmbedding:
    """Keep disjoint feature columns; concatenate lookups without rounding or sums.

    All ranks must index the same IDs, including draft and CUDA-graph paths.
    The caller slices on CPU before uploading to avoid a full GPU allocation.
    """
    def __init__(self, local, world):
        if world != 2 or local.ndim != 2:
            raise ValueError('embedding TP requires two feature shards')
        self.local, self.world = local, world
        self.shape = (local.shape[0], local.shape[1] * world)
        self.device, self.dtype = local.device, local.dtype

    def __getitem__(self, ids):
        local = self.local[ids].contiguous()
        flat = local.reshape(-1, local.shape[-1])
        rows, width = flat.shape
        gathered = torch.empty((self.world * rows, width), dtype=flat.dtype, device=flat.device)
        comm.all_gather_fast(gathered, flat, group=collective_group())
        return gathered.view(self.world, rows, width).transpose(0, 1).reshape(
            *local.shape[:-1], self.shape[1])


class VocabParallelHead:
    def __init__(self, local, world):
        self.local, self.world = local, world
        self.shape = (local.shape[0] * world, local.shape[1])

    def tp_logits(self, x):
        import v41_ref as R
        local = R.head_logits(x, self.local).contiguous()
        flat = local.reshape(-1, local.shape[-1])
        gathered = torch.empty((self.world * flat.shape[0], flat.shape[1]),
                               dtype=flat.dtype, device=flat.device)
        comm.all_gather_fast(gathered, flat, group=collective_group())
        return gathered.view(self.world, flat.shape[0], flat.shape[1]).transpose(0, 1).reshape(
            *x.shape[:-1], self.shape[0])

    def tp_markov_greedy(self, x, embed, head_local, anchor, linear, rank):
        """Greedy DSpark Markov chain without gathering logits: each rank biases its own vocabulary
        half (head_local = this rank's rows of the Markov head) and only (max, global id) pairs cross
        the network, one tiny all-gather per draft position. Ties resolve to the lower global id,
        as a full-vocabulary argmax does. Returns (proposals [T] int64, previous-token embeddings)."""
        import v41_ref as R
        local = R.head_logits(x, self.local).reshape(-1, self.local.shape[0])
        prev = anchor.reshape(1)
        proposals, embeddings = [], []
        for i in range(local.shape[0]):
            e = embed[prev]
            embeddings.append(e)
            lg = local[i] + linear(e, head_local).float()[0]
            best = lg.argmax().view(1)
            packed = torch.cat([lg.gather(0, best),
                                self._column_id(best, rank).view(torch.float32)]).contiguous()
            gathered = torch.empty(self.world * 2, dtype=torch.float32, device=lg.device)
            comm.all_gather_fast(gathered, packed, group=collective_group())
            pairs = gathered.view(self.world, 2)
            winner = pairs[:, 0].argmax().view(1)
            prev = pairs[:, 1].contiguous().view(torch.int32).gather(0, winner).to(torch.int64)
            proposals.append(prev)
        return torch.cat(proposals), embeddings

    def tp_candidates(self, x, k_local):
        """Each rank's top-k_local logits per row, gathered: (global ids int64 [rows, world*k],
        values fp32 [rows, world*k]), ids ascending per row so ties resolve to the lower id like a
        full-vocabulary argmax. Only the candidates cross the network (rows*world*k*8 bytes)
        instead of the full fp32 logits. Both ranks receive identical tensors."""
        import v41_ref as R
        local = R.head_logits(x, self.local).reshape(-1, self.local.shape[0])
        rows = local.shape[0]
        values, ids = local.topk(k_local, dim=-1)
        packed = torch.cat([values, ids.to(torch.int32).view(torch.float32)], dim=-1).contiguous()
        gathered = torch.empty((self.world * rows, 2 * k_local), dtype=torch.float32, device=local.device)
        comm.all_gather_fast(gathered, packed, group=collective_group())
        g = gathered.view(self.world, rows, 2 * k_local)
        gids = self._gathered_ids(g[..., k_local:].contiguous().view(torch.int32).to(torch.int64))
        gids = gids.permute(1, 0, 2).reshape(rows, self.world * k_local)
        gvals = g[..., :k_local].permute(1, 0, 2).reshape(rows, self.world * k_local)
        gids, order = gids.sort(dim=-1)
        return gids, gvals.gather(1, order)

    def _column_id(self, col, rank):
        """Global token id (int32) of this rank's head column `col` (int64 [1])."""
        return col.to(torch.int32) + rank * self.local.shape[0]

    def _gathered_ids(self, cols):
        """Global token ids (int64) of gathered columns `cols` [world, rows, k], rank-major."""
        offset = (torch.arange(self.world, device=cols.device, dtype=torch.int64)
                  * self.local.shape[0]).view(self.world, 1, 1)
        return cols + offset


class ListedVocabHead(VocabParallelHead):
    """The drafter's vocabulary head over listed ids (engine/draft_vocab.py), one contiguous run of
    them a rank. Draft-only: the verifier keeps its own VocabParallelHead over the full vocabulary.

    `local` holds this rank's listed rows of `head.weight` in column order (ascending ids);
    `ids_all` is every rank's list, so each rank maps any gathered column to its token id. The
    greedy chain and the candidate exchange are the base class's with that mapping; `tp_logits`
    scatters the gathered listed logits into a full-vocabulary row of -inf, so the sampling draft,
    the global shortlist and the tree probe read the same [T, V] shape as before and an unlisted
    token gets probability zero (rejection sampling stays exact: the residual can still emit it)."""

    def __init__(self, local, world, ids_all, vocab, rank):
        super().__init__(local, world)
        if ids_all.numel() != world * local.shape[0]:
            raise ValueError('listed vocabulary: one run of local.shape[0] ids a rank')
        self.vocab, self.rank = vocab, rank
        self.shape = (vocab, local.shape[1])
        self.ids_all = ids_all.to(torch.int64)
        self.ids_by_rank = self.ids_all.view(world, local.shape[0])
        self.ids_local = self.ids_by_rank[rank].to(torch.int32).contiguous()

    def tp_logits(self, x):
        import v41_ref as R
        local = R.head_logits(x, self.local).contiguous()
        flat = local.reshape(-1, local.shape[-1])
        rows, n = flat.shape
        gathered = torch.empty((self.world * rows, n), dtype=flat.dtype, device=flat.device)
        comm.all_gather_fast(gathered, flat, group=collective_group())
        listed = gathered.view(self.world, rows, n).transpose(0, 1).reshape(rows, self.world * n)
        out = torch.full((rows, self.vocab), float('-inf'), dtype=listed.dtype, device=listed.device)
        out.index_copy_(1, self.ids_all, listed)
        return out.reshape(*x.shape[:-1], self.vocab)

    def _column_id(self, col, rank):
        return self.ids_local.gather(0, col)

    def _gathered_ids(self, cols):
        w, rows, k = cols.shape
        return self.ids_by_rank.gather(1, cols.reshape(w, rows * k)).reshape(w, rows, k)


def make_listed_draft_head(head, ids_all, rank, world, device):
    """The draft's ListedVocabHead from the full bf16 `head.weight` (every rank loads it before
    taking its verifier shard), in DSV41_DRAFT_HEAD_FMT. No collective runs here."""
    import v41_ref as R
    from engine import draft_vocab
    fmt = R.draft_head_fmt()
    if fmt not in ('fp8', 'fp4', 'bf16'):
        raise ValueError('DSV41_DRAFT_VOCAB builds its own head: set DSV41_DRAFT_HEAD_FMT to fp8, fp4 or bf16')
    mine = draft_vocab.rank_ids(ids_all, rank, world)
    rows = head.index_select(0, mine.to(head.device)).to(device)
    local = R._head_in_format(rows, fmt)
    del rows
    return ListedVocabHead(local, world, ids_all.to(device), head.shape[0], rank)
