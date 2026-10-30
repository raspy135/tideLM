// Measure how often another GPU client takes the GPU away from CUDA work.
//
// One thread spins on %globaltimer and logs every interval in which it did not run. On an idle
// GB10 a gap means another context (Xorg, gnome-shell, a browser or Electron GPU process, another
// CUDA process) held the GPU. The decode probe's in-graph events see the same stalls as a random
// cheap kernel (rope, rmsnorm, residual) taking ~1 ms on one node, followed by the peer waiting the
// same amount in the next collective. Run it with the engine idle on each node and compare
// (a desktop session on one node stalls the pair).
//
//   nvcc -O2 -arch=native -o /tmp/gpu_gaps tools/gpu_timeslice_gaps.cu
//   /tmp/gpu_gaps [seconds=5] [threshold_ns=50000]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void spin(unsigned long long dur_ns, unsigned long long thresh_ns,
                     unsigned long long* starts, unsigned long long* lens,
                     int cap, int* count) {
  unsigned long long t0 = now(), prev = t0;
  int n = 0;
  while (true) {
    unsigned long long t = now();
    if (t - prev > thresh_ns && n < cap) { starts[n] = prev - t0; lens[n] = t - prev; n++; }
    prev = t;
    if (t - t0 > dur_ns) break;
  }
  *count = n;
}

int main(int argc, char** argv) {
  double seconds = argc > 1 ? atof(argv[1]) : 5.0;
  unsigned long long thresh = argc > 2 ? atoll(argv[2]) : 50000;  // 50 us
  const int cap = 1 << 16;
  unsigned long long *s, *l; int* c;
  cudaMalloc(&s, cap * 8); cudaMalloc(&l, cap * 8); cudaMalloc(&c, 4);
  spin<<<1, 1>>>((unsigned long long)(seconds * 1e9), thresh, s, l, cap, c);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
  int n; cudaMemcpy(&n, c, 4, cudaMemcpyDeviceToHost);
  static unsigned long long hs[1 << 16], hl[1 << 16];
  cudaMemcpy(hs, s, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hl, l, n * 8, cudaMemcpyDeviceToHost);
  double tot = 0, mx = 0; int over500 = 0;
  for (int i = 0; i < n; i++) { tot += hl[i]; if (hl[i] > mx) mx = hl[i]; if (hl[i] > 500000) over500++; }
  printf("window %.1fs  gaps>%.0fus: %d (%.1f/s)  >500us: %d  max %.3f ms  stolen %.2f%%\n",
         seconds, thresh / 1e3, n, n / seconds, over500, mx / 1e6, 100 * tot / (seconds * 1e9));
  // gap length histogram (ms)
  int h[8] = {0}; double edges[8] = {0.1, 0.25, 0.5, 1, 1.5, 2, 4, 1e9};
  for (int i = 0; i < n; i++) { double ms = hl[i] / 1e6; for (int b = 0; b < 8; b++) if (ms < edges[b]) { h[b]++; break; } }
  printf("len<0.1ms:%d <0.25:%d <0.5:%d <1:%d <1.5:%d <2:%d <4:%d >=4:%d\n", h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
  printf("first gaps (start_ms len_ms):");
  for (int i = 0; i < n && i < 24; i++) printf(" %.1f/%.2f", hs[i] / 1e6, hl[i] / 1e6);
  printf("\n");
  return 0;
}
