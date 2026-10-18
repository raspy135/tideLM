"""Boot the real HTTP server on both ranks from a source snapshot, benchmark it, shut it down.

The end-to-end check for DSV41_MAX_CONCURRENCY: scheduler thread, worker loop, HTTP clients.
Run through the disposable two-node runner so neither node's checkout is touched:

    GATE_IMAGE=<image> GATE_LOG_DIR=results/serve-bench-<date> GATE_SOURCE_ROOT=<snapshot> \\
      GATE_ENV="DSV41_MAX_CONCURRENCY=4" GATE_TIMEOUT=2400s \\
      bash tools/run_two_node_gate.sh serve_bench_tp.py --clients 4 --tokens 128

`--prompt-sets "2,8;2,3"` runs the client once per set on the same server (two Python requests,
then Python + a story). Both ranks print MemAvailable every 2 s ("MEM rank=...").

Rank 1 serves headless and exits when rank 0 shuts down. Rank 0 starts the server, waits for
/health, runs tools/bench_concurrency_http.py, prints /health's concurrency block, sends SIGTERM.
Everything else is the profile's own .env: dynamic experts, probe prefill, dynamic depth,
decode-time adaptation, the prefix cache.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from urllib.request import urlopen


def sample_memory(rank, stop):
    """Host MemAvailable every 2 s on this rank ("MEM rank=R t=... avail_gib=..."), the number the
    memory watchdog acts on; the low-water mark is the run's memory cost."""
    start = time.time()
    while not stop.wait(2.0):
        with open('/proc/meminfo') as f:
            avail = next(int(line.split()[1]) for line in f if line.startswith('MemAvailable'))
        print(f'MEM rank={rank} t={time.time() - start:.0f} avail_gib={avail / 1048576:.2f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8011)
    ap.add_argument('--max-seq', type=int, default=65536)
    ap.add_argument('--arena-gb', type=float, default=90.2)
    ap.add_argument('--prune-keep', type=float, default=.61)
    ap.add_argument('--clients', type=int, default=4)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--prompt-sets', default='',
                    help='semicolon-separated prompt sets for bench_concurrency_http.py --prompts, run '
                         'one after another on the same server, e.g. "2,8;2,3"')
    args = ap.parse_args()
    rank = int(os.environ.get('RANK', '0'))
    threading.Thread(target=sample_memory, args=(rank, threading.Event()), daemon=True).start()
    cmd = [sys.executable, '-m', 'server.app', '--model-dir', os.environ['MODEL_DIR'],
           '--host', '127.0.0.1', '--port', str(args.port), '--engine', 'v41',
           '--max-seq', str(args.max_seq), '--arena-gb', str(args.arena_gb),
           '--trace-stats', '/app/results/trace-union/stats/coverage.json',
           '--engine-kwargs', json.dumps({'prune_keep': args.prune_keep, 'expert_format': 'fp4',
                                          'transient_slots': 8, 'keep_free_gb': 6})]
    if rank != 0:
        # The worker's process-group teardown can wait on a head that has already exited, so
        # stop it shortly after it logs the shutdown instead of waiting for the gate timeout.
        worker = subprocess.Popen(cmd + ['--headless'], cwd='/app', stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in worker.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if 'EP2 worker: shutdown from rank 0' in line:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('serve_bench: worker still in teardown 30 s after shutdown; stopping it',
                          flush=True)
                    worker.kill()
                sys.exit(0)
        sys.exit(worker.wait())
    server = subprocess.Popen(cmd, cwd='/app')
    url = f'http://127.0.0.1:{args.port}'
    try:
        deadline = time.time() + 900
        while True:
            if server.poll() is not None:
                raise SystemExit(f'server exited during startup: {server.returncode}')
            try:
                with urlopen(url + '/health', timeout=5) as response:
                    if json.load(response).get('status') == 'ok':
                        break
            except OSError:
                pass
            if time.time() > deadline:
                raise SystemExit('server did not become healthy')
            time.sleep(5)
        bench = [sys.executable, '/app/tools/bench_concurrency_http.py', '--url', url,
                 '--clients', str(args.clients), '--tokens', str(args.tokens), '--rounds', str(args.rounds)]
        rc = 0
        for prompts in (args.prompt_sets.split(';') if args.prompt_sets else ['']):
            rc = rc or subprocess.call(bench + (['--prompts', prompts] if prompts else []), cwd='/app')
        with urlopen(url + '/health', timeout=30) as response:
            health = json.load(response)
        print('SERVE_BENCH_HEALTH ' + json.dumps({'status': health.get('status'),
                                                  'concurrency': health.get('concurrency')}), flush=True)
        if rc or health.get('status') != 'ok':
            raise SystemExit(f'benchmark rc={rc}, health={health.get("status")}')
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == '__main__':
    main()
