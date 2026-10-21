"""engine/dream_overlay.py: the overlay must be served exactly, or refused before any expert is read."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import dream_overlay as D
from engine.experts import NAMES, ShardFile

SHAPES = {"w1.weight": ("I8", [4, 8]), "w1.scale": ("F8_E8M0", [4, 1]),
          "w2.weight": ("I8", [8, 2]), "w2.scale": ("F8_E8M0", [8, 1]),
          "w3.weight": ("I8", [4, 8]), "w3.scale": ("F8_E8M0", [4, 1])}


def tensors(seed):
    out = {}
    for i, n in enumerate(NAMES):
        dtype, shape = SHAPES[n]
        out[n] = (dtype, shape, bytes((seed * 31 + i * 7 + k) % 256 for k in range(shape[0] * shape[1])))
    return out


class DreamOverlayTest(unittest.TestCase):
    def setUp(self):
        """A one-layer, two-expert 'checkpoint' in one shard, and an overlay root with serve/current."""
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.md = root / "model"
        self.md.mkdir()
        shard = self.md / "model-00001-of-00001.safetensors"
        hdr, blobs, off = {}, [], 0
        for e in (0, 1):
            for n, (dtype, shape, raw) in tensors(e).items():
                hdr[D.prefix(0, e) + n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
                off += len(raw)
                blobs.append(raw)
        h = json.dumps(hdr).encode()
        h += b" " * (-len(h) % 8)
        shard.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))
        self.wm = {name: shard.name for name in hdr}
        (self.md / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.wm}))
        self.serve = root / "dream" / "serve"
        self.gen = self.serve / "gen-a"
        self.gen.mkdir(parents=True)
        os.symlink("gen-a", self.serve / "current")
        env = mock.patch.dict(os.environ, {D.ENV: str(self.serve / "current")})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)
        D._digest_cache.clear()

    def manifest(self, gen, experts, **over):
        m = {"version": D.VERSION, "parent": None, "experts": experts,
             "base_index_sha256": D.sha256_file(str(self.md / "model.safetensors.index.json"))}
        m.update(over)
        (gen / D.MANIFEST).write_text(json.dumps(m))

    def add(self, gen, layer, expert, t):
        f = D.file_name(layer, expert)
        return {"layer": layer, "expert": expert, "file": f,
                "sha256": D.write_expert(str(gen / f), layer, expert, t),
                "base_sha256": D.base_sha256(str(self.md), self.wm, layer, expert)}

    def load(self):
        return D.load(str(self.md), self.wm, n_layers=1)

    def test_identity_roundtrip_reads_the_checkpoint_bytes(self):
        t = D.read_expert(str(self.md), self.wm, 0, 1)
        self.assertEqual(t, tensors(1))
        self.manifest(self.gen, [self.add(self.gen, 0, 1, t)])
        ov = self.load()
        self.assertEqual(set(ov), {D.prefix(0, 1) + n for n in NAMES})
        # ShardFile -- what ExpertStore reads through -- finds the same bytes in the overlay file
        sf = ShardFile(ov[D.prefix(0, 1) + "w1.weight"])
        raw = (self.gen / "L0.E0001.safetensors").read_bytes()
        for n in NAMES:
            a, b = sf.spans[D.prefix(0, 1) + n]
            self.assertEqual(raw[a:b], tensors(1)[n][2])

    def test_changed_expert_is_what_gets_served_and_only_it(self):
        t = tensors(1)
        t["w2.weight"] = ("I8", [8, 2], bytes(16))
        self.manifest(self.gen, [self.add(self.gen, 0, 1, t)])
        ov = self.load()
        sf = ShardFile(ov[D.prefix(0, 1) + "w2.weight"])
        a, b = sf.spans[D.prefix(0, 1) + "w2.weight"]
        self.assertEqual((self.gen / "L0.E0001.safetensors").read_bytes()[a:b], bytes(16))
        self.assertNotIn(D.prefix(0, 0) + "w2.weight", ov)

    def test_refuses_a_file_that_does_not_match_its_manifest(self):
        e = self.add(self.gen, 0, 1, tensors(1))
        self.manifest(self.gen, [e])
        with open(self.gen / e["file"], "r+b") as f:
            f.seek(-1, 2)
            f.write(b"\x00")
        with self.assertRaisesRegex(ValueError, "partial copy"):
            self.load()

    def test_refuses_another_checkpoint(self):
        self.manifest(self.gen, [self.add(self.gen, 0, 1, tensors(1))], base_index_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "made for checkpoint"):
            self.load()

    def test_refuses_wrong_shape(self):
        t = tensors(1)
        t["w1.weight"] = ("I8", [8, 4], t["w1.weight"][2])
        self.manifest(self.gen, [self.add(self.gen, 0, 1, t)])
        with self.assertRaisesRegex(ValueError, "checkpoint's is"):
            self.load()

    def test_refuses_duplicates_and_non_backbone_experts(self):
        e = self.add(self.gen, 0, 1, tensors(1))
        self.manifest(self.gen, [e, e])
        with self.assertRaisesRegex(ValueError, "listed twice"):
            self.load()
        self.manifest(self.gen, [dict(e, layer=1)])
        with self.assertRaisesRegex(ValueError, "not a routed expert"):
            self.load()

    def test_digest_follows_the_generation_not_the_path(self):
        self.manifest(self.gen, [self.add(self.gen, 0, 1, tensors(1))])
        d1 = D.digest()
        gen_b = self.serve / "gen-b"
        gen_b.mkdir()
        self.manifest(gen_b, [self.add(gen_b, 0, 0, tensors(0))], parent="gen-a")
        (self.serve / "current").unlink()
        os.symlink("gen-b", self.serve / "current")
        self.assertTrue(D.path().endswith("gen-b"))
        self.assertEqual(len(d1), 16)
        self.assertNotEqual(D.digest(), d1)

    def test_off_is_empty(self):
        with mock.patch.dict(os.environ, {D.ENV: ""}):
            self.assertFalse(D.enabled())
            self.assertEqual(D.digest(), "")
            self.assertEqual(D.load("/nonexistent", {}, 40), {})


if __name__ == "__main__":
    unittest.main()


class DreamBaseHashTest(DreamOverlayTest):
    """An overlay trained from other weights is refused even when the index is identical."""

    def test_refuses_an_overlay_from_other_base_weights(self):
        e = self.add(self.gen, 0, 1, tensors(9))
        e["base_sha256"] = "f" * 64
        self.manifest(self.gen, [e])
        with self.assertRaisesRegex(ValueError, "trained from different weights"):
            self.load()

    def test_refuses_a_generation_without_base_hashes(self):
        e = self.add(self.gen, 0, 1, tensors(9))
        del e["base_sha256"]
        self.manifest(self.gen, [e])
        with self.assertRaisesRegex(ValueError, "predates base hashes"):
            self.load()


class DreamFP8TierTest(DreamOverlayTest):
    """load_fp8: the FP8 copy is decoded exactly (e4m3 x 2^k is exact in BF16), and verified."""

    def fp8_entry(self, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        t8, ref = {}, {}
        for m, (n, k) in {"w1": (2304, 5120), "w2": (5120, 2304), "w3": (2304, 5120)}.items():
            w = torch.randn(n, k, generator=g) * 0.02
            x = w.reshape(n, -1, 32)
            e = torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126) / 448.0))
            codes = (x / torch.exp2(e)).to(torch.float8_e4m3fn).reshape(n, k)
            sc = (e.squeeze(-1) + 127).to(torch.uint8)
            t8[f"{m}.weight"] = ("F8_E4M3", [n, k], codes.view(torch.uint8).numpy().tobytes())
            t8[f"{m}.scale"] = ("F8_E8M0", list(sc.shape), sc.numpy().tobytes())
            ref[m] = (codes.float() * torch.exp2(sc.float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16)
        entry = self.add(self.gen, 0, 1, D.read_expert(str(self.md), self.wm, 0, 1))
        f8 = D.file_name(0, 1).replace(".safetensors", ".fp8.safetensors")
        entry.update(fp8_file=f8, fp8_sha256=D.write_expert(str(self.gen / f8), 0, 1, t8), serve="fp8")
        return entry, ref

    def test_fp8_tier_decodes_exactly(self):
        import torch
        entry, ref = self.fp8_entry()
        self.manifest(self.gen, [entry, self.add(self.gen, 0, 0, D.read_expert(str(self.md), self.wm, 0, 0))])
        self.assertEqual([e["expert"] for e in D.fp8_entries()], [1])
        side = D.load_fp8("cpu")
        self.assertEqual(set(side), {0})
        self.assertEqual(set(side[0]), {1})
        for got, m in zip(side[0][1], ("w1", "w2", "w3")):
            self.assertEqual(got.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(got, ref[m]), m)
        self.assertIn("1 served FP8", D.describe())

    def test_fp8_tier_refuses_a_mismatched_copy(self):
        entry, _ = self.fp8_entry()
        entry["fp8_sha256"] = "0" * 64
        self.manifest(self.gen, [entry])
        with self.assertRaisesRegex(ValueError, "partial copy"):
            D.load_fp8("cpu")

    def test_tp_tier_slices_are_the_arena_layout(self):
        """load_fp8_tp: each rank holds its w1/w3 rows and its w2 output rows, as ExpertArena does,
        and the map sends only the tier's experts to a slot."""
        import torch
        entry, ref = self.fp8_entry()
        self.manifest(self.gen, [entry])
        self.assertEqual(D.fp8_tp_bytes(2), 2 * 1152 * (5120 + 160) + 2560 * (2304 + 72))
        full = D.load_fp8("cpu")[0][1]
        for rank in range(2):
            t = D.load_fp8_tp("cpu", rank, 2)[0]
            self.assertEqual(t.ids, [1])
            self.assertEqual(t.map.tolist().count(-1), 383)
            self.assertEqual(int(t.map[1]), 0)
            self.assertEqual(t.nbytes, D.fp8_tp_bytes(2))
            dq = lambda w, s: (w.view(torch.float8_e4m3fn).float()
                               * torch.exp2(s.float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16)
            rows = slice(rank * 1152, (rank + 1) * 1152)
            self.assertTrue(torch.equal(dq(t.w13[0, :1152], t.s13[0, :1152]), full[0][rows]))
            self.assertTrue(torch.equal(dq(t.w13[0, 1152:], t.s13[0, 1152:]), full[2][rows]))
            self.assertTrue(torch.equal(dq(t.w2[0], t.s2[0]), full[1][rank * 2560:(rank + 1) * 2560]))

    def test_pins_cover_every_overlay_expert(self):
        entry, _ = self.fp8_entry()
        self.manifest(self.gen, [entry, self.add(self.gen, 0, 0, D.read_expert(str(self.md), self.wm, 0, 0))])
        self.assertEqual(D.pins(), {0: [0, 1]})

    def test_pin_keep_swaps_in_pinned_experts_and_keeps_the_size(self):
        counts = {0: [5.0, 1.0, 9.0, 0.5, 3.0], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}
        keep = {0: [0, 2, 4], 1: [2, 3, 4]}
        out = D.pin_keep(keep, counts, {0: [3, 1], 1: [3]})
        self.assertEqual(sorted(out[0]), [1, 2, 3])   # 3 replaces 4 (demand 3.0), then 1 replaces 0 (5.0)
        self.assertEqual(out[1], [2, 3, 4])
        self.assertEqual(len(out[0]), 3)
        with self.assertRaisesRegex(ValueError, "fewer than its pinned"):
            D.pin_keep({0: [0]}, counts, {0: [1, 3]})

    def test_no_fp8_tier_when_everything_serves_fp4(self):
        self.manifest(self.gen, [self.add(self.gen, 0, 1, D.read_expert(str(self.md), self.wm, 0, 1))])
        self.assertEqual(D.fp8_entries(), [])
        self.assertEqual(D.load_fp8("cpu"), {})
