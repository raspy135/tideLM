"""Profile one Muse decode step on the pair (eager, so kernels are visible). Rank 0 prints the
top CUDA kernels by total time. Synthetic checkpoint by default."""
import os
import sys

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.muse.tp import MuseTP  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "/app/results/muse-synth-v2"
eng = MuseTP(model, max_seq=2048, chunk=256)
eng.use_graphs = False
eng.reset()
eng.forward(list(range(3, 603)))
for t in range(5):
    eng.forward([7 + t])
torch.cuda.synchronize()
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for t in range(10):
        eng.forward([11 + t])
    torch.cuda.synchronize()
if eng.rank == 0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=60))
