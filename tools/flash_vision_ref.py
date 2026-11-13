"""flash_vision_ref.py: GLM-5.3-Flash's vision tower in the Rust engine against transformers.

transformers' Glm5NextImageProcessor turns each image into pixel_values and a grid; those values,
rounded to bf16 as the model rounds them, go to `flash_vision_check` (the Rust tower) and to
transformers' Glm5NextVisionModel (bf16, CUDA, the checkpoint's model.visual.* weights); the two
outputs' rows are compared. With --rust-patches the Rust preprocessing's patches are compared with
the processor's too (tidelm-image's glm5next path, through `flash_vision_check --image`).

    python -I tools/flash_vision_ref.py --dense DIR --bin rust/target/release/flash_vision_check IMAGE [IMAGE ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


def bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dense", required=True)
    ap.add_argument("--bin", required=True, help="flash_vision_check")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args(argv)
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoImageProcessor
    from transformers.models.glm5_next import configuration_glm5_next as C
    from transformers.models.glm5_next import modeling_glm5_next as M

    proc = AutoImageProcessor.from_pretrained(a.dense)
    imgs = [Image.open(p) for p in a.images]
    bt = proc(images=imgs, return_tensors="pt")
    pv, grid = bt["pixel_values"], bt["image_grid_thw"]
    grids = [(int(g[1]), int(g[2])) for g in grid]
    print(f"{len(imgs)} image(s): sizes {[im.size for im in imgs]}, grids {grids}, {pv.shape[0]} patches x {pv.shape[1]}")
    tmp = tempfile.mkdtemp()
    pfile, ofile = os.path.join(tmp, "p.bin"), os.path.join(tmp, "o.bin")
    bf16_bits(pv).tofile(pfile)
    r = subprocess.run([a.bin, "--model-dir", a.dense, "--patches", pfile, "--grids", ";".join(f"{h},{w}" for h, w in grids),
                        "--out", ofile, "--max", str(max(4096, ((pv.shape[0] + 1023) // 1024) * 1024))], capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    if r.returncode:
        return r.returncode
    eng = torch.from_numpy(np.fromfile(ofile, dtype=np.uint16).view(np.int16).astype(np.int32) << 16).view(torch.float32)
    eng = eng.reshape(-1, 4096)
    # transformers' tower, bf16 on CUDA
    vc = json.load(open(os.path.join(a.dense, "config.json")))["vision_config"]
    cfg = C.Glm5NextVisionConfig(**vc)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = M.Glm5NextVisionModel(cfg)
    model = model.to_empty(device="cuda").to(torch.bfloat16)
    idx = json.load(open(os.path.join(a.dense, "model.safetensors.index.json")))["weight_map"]
    sd, files = {}, {}
    for name, shard in idx.items():
        if name.startswith("model.visual."):
            if shard not in files:
                files[shard] = safe_open(os.path.join(a.dense, shard), framework="pt", device="cuda")
            sd[name[len("model.visual."):]] = files[shard].get_tensor(name)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("missing:", missing[:8], "unexpected:", unexpected[:8])
    # buffers the module builds itself (rotary inv_freq) were left empty by to_empty: rebuild them
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn") or type(m).__name__.endswith("RotaryEmbedding"):
            dim = m.inv_freq.shape[0] * 2
            m.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cuda") / dim))
    model.eval()
    acts = {}
    model.patch_embed.register_forward_hook(lambda m, i, o: acts.__setitem__("embed", o))
    for bi, blk in enumerate(model.blocks):
        blk.register_forward_hook(lambda m, i, o, bi=bi: acts.__setitem__(f"b{bi}", o[0] if isinstance(o, tuple) else o))
    model.post_layernorm.register_forward_hook(lambda m, i, o: acts.__setitem__("post", o))
    model.downsample.register_forward_hook(lambda m, i, o: acts.__setitem__("down", o))
    with torch.no_grad():
        ref = model(pv.to("cuda", torch.bfloat16), grid_thw=grid.to("cuda")).pooler_output.float().cpu()
    dp = os.environ.get("TIDELM_VISION_DUMP")
    if dp:
        for k in ["embed"] + [f"b{i}" for i in range(24)] + ["post", "down"]:
            f = f"{dp}.{k}.bin"
            if not os.path.exists(f) or k not in acts:
                continue
            e = torch.from_numpy(np.fromfile(f, dtype=np.uint16).view(np.int16).astype(np.int32) << 16).view(torch.float32)
            rr = acts[k].float().reshape(-1).cpu()
            if rr.numel() != e.numel():
                print(f"  {k}: sizes {e.numel()} vs {rr.numel()}")
                continue
            e = e.reshape(rr.numel() // (4096 if k == "down" else 1024), -1); rr = rr.reshape(e.shape)
            cosk = F.cosine_similarity(e, rr, dim=-1)
            print(f"  {k}: rel err {((e - rr).norm() / rr.norm()).item():.3e}, row cos min {cosk.min().item():.5f}")
    if ref.shape != eng.shape:
        print("shapes differ:", ref.shape, eng.shape)
        return 1
    rel = ((eng - ref).norm() / ref.norm()).item()
    cos = F.cosine_similarity(eng, ref, dim=-1)
    print(f"tower: {ref.shape[0]} rows, rel err {rel:.3e}, row cosine min {cos.min().item():.5f} mean {cos.mean().item():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
