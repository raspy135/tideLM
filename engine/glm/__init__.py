"""GLM-5.3 (754B, glm_moe_dsa) on two DGX Sparks: reference model, TP2 engine, expert residency.

Built on the Muse Spark work (engine/muse/): the same split (TP2, MXFP4 expert arena on the
native/Triton FP4 kernels, substitution or streamed misses, graphed decode, prefix reuse) with
GLM's DeepSeek-V3.2-style layers: MLA attention, the DSA indexer (21 of 78 layers; the others
reuse its top-k), DeepSeek MoE with a shared expert, and one MTP layer. See docs/glm53-plan.md.
"""
