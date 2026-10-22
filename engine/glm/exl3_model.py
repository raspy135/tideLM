"""Dense and MTP weights from the same EXL3 checkpoint as the routed experts.

No Quark tensors are read on this path. Published trellises stay EXL3; raw
norms, routers, embeddings and absorbed MLA weights come from that checkpoint.
"""
import torch

from .exl3_attn import Exl3Source
from .weights import P


class FixedMLP:
    def __init__(self, source, prefix, eng):
        r, w, d = eng.rank, eng.world, eng.dev
        self.gate = source.lin(prefix + 'gate_proj', d, 4, r, w, 'n')
        self.up = source.lin(prefix + 'up_proj', d, 4, r, w, 'n')
        self.down = source.lin(prefix + 'down_proj', d, 2, r, w, 'n')
        self.comm = eng.comm

    @property
    def nbytes(self):
        return self.gate.nbytes + self.up.nbytes + self.down.nbytes

    def __call__(self, x):
        gate, up = self.gate(x).float(), self.up(x).float()
        h = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        full = self.comm.cat_last(h).contiguous()
        return self.comm.cat_last(self.down(full, out_dtype=torch.float32))


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
        # Published kv_b is FP16, not Quark's block-scaled FP8. Preserve its
        # values through the engine's BF16 MLA path; never invent FP8 scales.
        kvb = ck.get_rows(a + 'kv_b_proj.weight', r * eng.hl * hv, (r + 1) * eng.hl * hv)
        kvb = kvb.to(d, torch.bfloat16).view(eng.hl, hv, cfg.kv_lora)
        lw = dict(
            ln_in=ck.get(p + 'input_layernorm.weight').to(d, torch.bfloat16),
            ln_post=ck.get(p + 'post_attention_layernorm.weight').to(d, torch.bfloat16),
            q_a=source.lin(a + 'q_a_proj', d, 8),
            q_norm=ck.get(a + 'q_a_layernorm.weight').to(d, torch.bfloat16),
            q_b=source.lin(a + 'q_b_proj', d, 1, r, w, 'n'),
            kv_a=source.lin(a + 'kv_a_proj_with_mqa', d, 8, n_out=cfg.kv_lora + cfg.rope_dim),
            kv_norm=ck.get(a + 'kv_a_layernorm.weight').to(d, torch.bfloat16),
            w_uk=kvb[:, :cfg.nope].contiguous(),
            w_uv=kvb[:, cfg.nope:].contiguous(),
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
    fixed = eng.fixed_exl3[layer](x)
    if eng.dense(layer):
        return fixed.to(torch.bfloat16)
    from .exl3 import routed
    if layer == eng.mtp_L:
        routed_slots, routed_weights = slots[:, :eng.cfg.topk], weights[:, :eng.cfg.topk]
        arena = eng.mtp_exl3
    else:
        arena = eng.x3
    output = routed(x, routed_slots, routed_weights, arena, eng.comm)
    return (output + fixed).to(torch.bfloat16)
