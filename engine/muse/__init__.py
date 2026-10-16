"""Muse Spark Open (816B-A42B) on two DGX Sparks: text model, TP2 engine, expert residency.

Kept apart from the DeepSeek V4.1 engine (engine/v41_engine.py and friends). Shared pieces are
imported, not modified: the native FP4 expert kernels (tools/fp4_moe_cuda.py), the FP8 dense path
(tools/fp8_linear.py) and the decode all-gather (engine/comm.py). Plan and numbers:
docs/muse-spark-plan.md.
"""
