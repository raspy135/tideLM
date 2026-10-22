"""Dense and MTP weights from the same EXL3 checkpoint as the routed experts.

No Quark tensors are read on this path. Published trellises stay EXL3; raw
norms, routers, embeddings and absorbed MLA weights come from that checkpoint.
"""
import torch

from .exl3_attn import Exl3Source
from .weights import P


class FixedMLP:
    def __init__(self, source, prefix, eng):
        from engine.exl3_dense import MAX_ROWS, Exl3Matrix
        from .exl3_attn import Exl3Lin, _members
        r, w, d = eng.rank, eng.world, eng.dev
        gate = source.lin(prefix + 'gate_proj', d, 4, r, w, 'n')
        up = source.lin(prefix + 'up_proj', d, 4, r, w, 'n')
        self.down = source.lin(prefix + 'down_proj', d, 2, r, w, 'n')
        self.comm = eng.comm
        # Decode: gate and up as one G=2 Exl3Matrix over the same input rows (engine/exl3_draft's
        # shared expert): one rotation/GEMV/combine set instead of two. Same split, so bitwise
        # equal to the separate calls; 16 shared experts L2-cold, rank-0 shapes, 1/4/8 rows:
        # 67.3/69.2/70.2 -> 60.0/61.5/61.3 us a layer (2026-10-22). Split 8 gave 57-60 but sums K
        # in another order. Prefill keeps per-matrix unpacking, on views of the stacked trellis
        # (Exl3Matrix copies G > 1 trellises; separate tensors would hold them twice).
        self.rows = MAX_ROWS
        self.n = gate.N
        self.gate_up = None
        if gate.bits == up.bits:
            self.gate_up = Exl3Matrix([gate.tr, up.tr], torch.stack([gate.suh, up.suh]),
                                      torch.stack([gate.svh, up.svh]), [gate.bits, up.bits], 4, _members(str(d)))
            gate = Exl3Lin(self.gate_up.w[0], gate.suh, gate.svh, 4)
            up = Exl3Lin(self.gate_up.w[1], up.suh, up.svh, 4)
        self.gate, self.up = gate, up

    @property
    def nbytes(self):
        return self.gate.nbytes + self.up.nbytes + self.down.nbytes

    def act(self, x):
        """This rank's intermediate columns, bf16 [T, inter / world]."""
        if self.gate_up is not None and x.shape[0] <= self.rows:
            gu = self.gate_up(x[:, None, :].expand(-1, 2, -1))
            gate, up = gu[:, :self.n].float(), gu[:, self.n:].float()
        else:
            gate, up = self.gate(x).float(), self.up(x).float()
        return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)

    def __call__(self, x):
        """A dense layer's MLP, bf16 [T, hidden]. The output gathers as BF16: the gather only
        concatenates, so rounding before it is the same values at half the bytes."""
        full = self.comm.cat_last(self.act(x)).contiguous()
        return self.comm.cat_last(self.down(full, out_dtype=torch.float32).to(torch.bfloat16))


def load_dense(eng):
    from .tp import Lin, _q8_rows
    import exl3_ref as ref
    cfg, ck, d, r, w = eng.cfg, eng.ck, eng.dev, eng.rank, eng.world
    if ref.manifest_sha256(str(eng.model_dir)) != eng.x3pack.reader.header['source_sha256']:
        raise ValueError('EXL3 routed pack and dense checkpoint have different source manifests')
    source = Exl3Source(str(eng.model_dir))
    eng.draft_head = None
    eng.embed = ck.get_rows(P + 'embed_tokens.weight', r * eng.vl, (r + 1) * eng.vl)
    eng.embed = _q8_rows(eng.embed, d, eng.embed_q) if eng.embed_q else eng.embed.to(d, torch.bfloat16)
    eng.head = source.lin('lm_head', d, 1, r, w, 'n')
    eng.final_norm = ck.get(P + 'norm.weight').to(d, torch.bfloat16)
    eng.L, eng.fixed_exl3 = [], {}
    tensors = eng.embed if eng.embed_q else (eng.embed,)
    nbytes = sum(t.numel() * t.element_size() for t in tensors) + eng.head.nbytes
    hv = cfg.nope + cfg.v_dim
    for layer in range(eng.nL):
        p = f'{P}layers.{layer}.'
        a = p + 'self_attn.'
        # Published kv_b is FP16. GLM_KVB_FP8=1 quantizes it at load to FP8 with 128x128 block
        # amax scales (weights._fp8_quant; a rank's rows start on a 128 boundary): against
        # Quark's FP8 kv_b, layers 0/3/40/77/78 gave identical scales and 99.9% identical bytes,
        # relative error to FP16 0.02645 vs Quark's 0.02646 (2026-10-22). That is the
        # configuration whose quality docs/gotchas.md measured; decode reads half the bytes.
        kvb = ck.get_rows(a + 'kv_b_proj.weight', r * eng.hl * hv, (r + 1) * eng.hl * hv)
        if eng.kvb_fp8:
            from .kvb_fp8 import HeadFP8
            from .weights import _fp8_quant
            kvb = HeadFP8.from_rows(*_fp8_quant(kvb.to(d)), eng.hl)
            w_uk, w_uv = kvb.split(0, cfg.nope).to(d), kvb.split(cfg.nope, hv).to(d)
        else:
            kvb = kvb.to(d, torch.bfloat16).view(eng.hl, hv, cfg.kv_lora)
            w_uk, w_uv = kvb[:, :cfg.nope].contiguous(), kvb[:, cfg.nope:].contiguous()
        lw = dict(
            ln_in=ck.get(p + 'input_layernorm.weight').to(d, torch.bfloat16),
            ln_post=ck.get(p + 'post_attention_layernorm.weight').to(d, torch.bfloat16),
            q_a=source.lin(a + 'q_a_proj', d, 8),
            q_norm=ck.get(a + 'q_a_layernorm.weight').to(d, torch.bfloat16),
            q_b=source.lin(a + 'q_b_proj', d, 1, r, w, 'n'),
            kv_a=source.lin(a + 'kv_a_proj_with_mqa', d, 8, n_out=cfg.kv_lora + cfg.rope_dim),
            kv_norm=ck.get(a + 'kv_a_layernorm.weight').to(d, torch.bfloat16),
            w_uk=w_uk,
            w_uv=w_uv,
            o=source.lin(a + 'o_proj', d, 2, r, w, 'k'))
        if layer == eng.mtp_L:
            lw.update(enorm=ck.get(p + 'enorm.weight').to(d, torch.bfloat16),
                      hnorm=ck.get(p + 'hnorm.weight').to(d, torch.bfloat16),
                      sh_norm=ck.get(p + 'shared_head.norm.weight').to(d, torch.bfloat16),
                      eh=source.lin(p + 'eh_proj', d, 4, r, w, 'n'))
        if eng.full(layer):
            i = a + 'indexer.'
            lw.update(i_q=source.lin(i + 'wq_b', d, 1),
                      i_k=Lin(ck.get(i + 'wk.weight'), d),
                      i_kn_w=ck.get(i + 'k_norm.weight').to(d).float(),
                      i_kn_b=ck.get(i + 'k_norm.bias').to(d).float(),
                      i_w=ck.get(i + 'weights_proj.weight').to(d).float())
        if not eng.dense(layer):
            lw['router'] = ck.get(p + 'mlp.gate.weight').to(d).to(
                torch.bfloat16 if eng.router_bf16 else torch.float32)
            lw['bias'] = ck.get(p + 'mlp.gate.e_score_correction_bias').to(d).float()
        prefix = p + 'mlp.' + ('' if eng.dense(layer) else 'shared_experts.')
        eng.fixed_exl3[layer] = FixedMLP(source, prefix, eng)
        nbytes += eng.fixed_exl3[layer].nbytes
        nbytes += sum(v.numel() * v.element_size() if isinstance(v, torch.Tensor) else v.nbytes
                      for v in lw.values())
        eng.L.append(lw)
    eng.mtp_exl3 = None
    if eng.mtp_full:
        import exl3_moe as model
        import exl3_ref as ref
        from .exl3 import warm
        ref.set_dims(cfg.hidden, cfg.moe_inter)
        bits = eng.x3pack.bits(eng.mtp_L)
        eng.mtp_exl3 = model.Exl3Arena(cfg.experts, d, tp_rank=r, tp_world=w, bits=bits, codebook='mul1')
        for expert in range(cfg.experts):
            record, width = eng.x3pack.read(eng.mtp_L, expert)
            eng.mtp_exl3.load_slot(expert, record, width)
        warm(eng.mtp_exl3)
        nbytes += eng.mtp_exl3.total_bytes
    eng.dense_bytes = nbytes
    dim = cfg.rope_dim
    inv = 1. / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=d) / dim))
    freqs = torch.arange(eng.max_seq, device=d, dtype=torch.float)[:, None] * inv[None]
    emb = torch.cat((freqs, freqs), -1)
    eng.cos, eng.sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    source.drop_cache()
    ck.drop_cache()
    eng.log('[glm] dense, fixed MLP and MTP weights: native EXL3 checkpoint (no Quark weights)')


def place_fixed(eng):
    # Slot IDs before x3_off are reserved for fixed bookkeeping. Their actual
    # weights live in FixedMLP and the separate 4-bit MTP arena, not MXFP4 slots.
    eng.fixed = {layer: torch.zeros(1, dtype=torch.int32, device=eng.dev) for layer in range(eng.nL)}
    if eng.mtp_full:
        eng.lut[eng.mtp_L] = torch.arange(eng.cfg.experts, dtype=torch.int32, device=eng.dev)
        for expert in range(eng.cfg.experts):
            eng.owner[expert] = (eng.mtp_L, expert)
    eng.first_routed = eng.x3_off


def mlp(eng, layer, x, slots, weights, routed_slots=None, routed_weights=None):
    fixed = eng.fixed_exl3[layer]
    if eng.dense(layer):
        return fixed(x)
    from .exl3 import routed_down, routed_up
    if layer == eng.mtp_L:
        routed_slots, routed_weights = slots[:, :eng.cfg.topk], weights[:, :eng.cfg.topk]
        arena = eng.mtp_exl3
    else:
        arena = eng.x3
    # One all-gather each way, as the hybrid's gather_merge: the routed and shared intermediates
    # go together, each rank sums its hidden columns of both down projections and the sum gathers
    # once as BF16. Separate routed() and FixedMLP() calls made four gathers a layer (two of
    # fp32 outputs); per element this is the same arithmetic, since the gathers only concatenate.
    h = fixed.act(x)
    xd, ctx = routed_up(x, routed_slots, routed_weights, arena)
    xd_full, h_full = eng._gather_pair(xd, h)
    local = routed_down(xd_full.contiguous(), ctx, arena)
    local = local + fixed.down(h_full.contiguous(), out_dtype=torch.float32)
    return eng.comm.cat_last(local.to(torch.bfloat16))
