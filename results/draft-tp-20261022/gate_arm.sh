#!/usr/bin/env bash
# gate_arm.sh <arm-name> <source-root> [extra DSV41_ k=v ...]
set -euo pipefail
name=$1; src=$2; shift 2
cd /home/ryan/git/tidelm
GATE_ENV="DSV41_PROBE_PREFILL=0 DSV41_DYNAMIC_EXPERTS=0 DSV41_BENCH_TRANSIENT_SLOTS=136 DSV41_RESIDENT_EXPERTS= $*" \
GATE_SOURCE_ROOT=$src GATE_IMAGE=sha256:b97a181d0c9d176211c42d81da8562d28357b9416273264347ca07594a07dda3 \
GATE_LOG_DIR=results/draft-tp-20261022/$name GATE_PROFILE=/home/ryan/git/tidelm/.env.ds41-mxfp4 GATE_TIMEOUT=1500s \
bash tools/run_two_node_gate.sh bench_decode_kernels_tp.py --out /app/results/draft-tp-20261022/$name
