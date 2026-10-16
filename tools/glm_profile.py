"""Profile one GLM decode step on the pair (eager, kernels visible). Rank 0 prints the table."""
import sys
sys.path.insert(0, "/app")
import torch  # noqa: E402
from engine.glm.tp import GlmTP  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "/app/results/glm-synth-v1"
eng = GlmTP(model, max_seq=int(sys.argv[2]) if len(sys.argv) > 2 else 2048, chunk=256, slots=48)
eng.use_graphs = False
eng.reset()
eng.forward(list(range(3, 603)))
for t in range(4):
    eng.forward([7 + t])
torch.cuda.synchronize()
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for t in range(10):
        eng.forward([11 + t])
    torch.cuda.synchronize()
if eng.rank == 0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=28, max_name_column_width=70))
