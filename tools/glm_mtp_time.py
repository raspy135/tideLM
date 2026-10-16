"""Time an eager MTP draft step vs a graphed target step on the synthetic checkpoint."""
import sys, time
sys.path.insert(0, "/app")
import torch  # noqa: E402
from engine.glm.tp import GlmTP  # noqa: E402
eng = GlmTP("/app/results/glm-synth-v2", max_seq=2048, chunk=256, slots=64)
eng.reset(); eng.mtp_fill = True
eng.forward(list(range(3, 603)))
h = eng.last_hidden[-1:]
for _ in range(3):
    eng.mtp_forward(torch.tensor([5], device=eng.dev), h, 599)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20):
    eng.mtp_forward(torch.tensor([5], device=eng.dev), h, 599)
torch.cuda.synchronize()
if eng.rank == 0:
    print(f"[time] eager MTP step {(time.perf_counter() - t0) / 20 * 1000:.2f} ms", flush=True)
