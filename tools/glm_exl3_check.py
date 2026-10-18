"""Correctness of the EXL3 routed path in GlmTP: engine/glm/exl3.routed (the CUDA kernels and the
two Comm all-gathers) against a plain reference from the same arena (Exl3Arena.dequant_slot:
fp32 SiLU(x W1) * (x W3), gather, W2, gather, weighted sum), at decode and prefill shapes.

    GLM_ENV="GLM_EXPERT_FORMAT=exl3 GLM_EXL3_PACK=/packs/glm-exl3-experts-r{rank}of2.bin GLM_SLOTS=2048" \\
        tools/muse_pair_run.sh glm_exl3_check.py
"""
import sys

sys.path.insert(0, "/app")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.glm import exl3  # noqa: E402
from engine.glm.tp import GlmTP  # noqa: E402


@torch.no_grad()
def reference(eng, x, slots, w):
    A, out = eng.x3, torch.zeros((x.shape[0], eng.cfg.hidden), dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(slots.shape[1]):
            w1, w2, w3 = A.dequant_slot(int(slots[t, k]))
            h = F.silu(x[t:t + 1].float() @ w1) * (x[t:t + 1].float() @ w3)    # [1, inter / world]
            hf = eng.comm.cat_last(h.contiguous())                              # [1, inter]
            o = eng.comm.cat_last((hf @ w2).contiguous())                       # [1, hidden]
            out[t] += float(w[t, k]) * o[0]
    return out


def main():
    eng = GlmTP("/models")
    assert eng.x3 is not None, "set GLM_EXPERT_FORMAT=exl3"
    g = torch.Generator(device="cpu").manual_seed(0)
    L = 40
    res = eng.lut[L][eng.lut[L] >= 0].cpu()
    for T in (1, 3, 256):
        x = (torch.randn((T, eng.cfg.hidden), generator=g) * 0.5).to(torch.bfloat16).to(eng.dev)
        pick = res[torch.randint(0, res.numel(), (T, eng.cfg.topk), generator=g)].to(eng.dev) - eng.x3_off
        w = torch.rand((T, eng.cfg.topk), generator=g).to(eng.dev)
        y = exl3.routed(x, pick, w, eng.x3, eng.comm)
        rows = min(T, 4)
        ref = reference(eng, x[:rows], pick[:rows], w[:rows])
        err = float((y[:rows] - ref).norm() / ref.norm())
        if eng.rank == 0:
            print(f"[check] T={T}: rel err {err:.2e}, |y| {float(y[:rows].norm()):.3f}, finite {bool(torch.isfinite(y).all())}",
                  flush=True)


if __name__ == "__main__":
    main()
