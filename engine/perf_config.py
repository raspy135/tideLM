"""Opt-in decode experiments; identical settings and sources are required on both ranks."""
from dataclasses import dataclass, asdict
import hashlib
import os
from pathlib import Path


@dataclass(frozen=True)
class DecodeExperiments:
    index_candidates: bool = False
    draft_router: bool = False
    hc_post_pre: bool = False
    attention_direct: bool = False
    engram_h2d: bool = False

    @classmethod
    def from_env(cls):
        values = {}
        for field, key in FLAGS.items():
            value = os.environ.get(key, "0")
            if value not in ("0", "1"):
                raise ValueError(f"{key} must be 0 or 1")
            values[field] = value == "1"
        # Batched decode composes its own HC/router boundaries. Until those
        # paths are qualified, do not silently mix experimental solo rounds
        # with baseline batched rounds under the same advertised configuration.
        if any(values.values()) and os.environ.get("DSV41_MAX_CONCURRENCY", "1") != "1":
            raise ValueError("decode performance experiments require DSV41_MAX_CONCURRENCY=1")
        return cls(**values)

    def boot_fields(self):
        fields = {f"perf_{key}": value for key, value in asdict(self).items()}
        # Disk prefixes and the config guard must not reuse results from a different
        # candidate-score implementation, even if both ranks select the same flag.
        root = Path(__file__).parent
        fields["perf_source"] = hashlib.sha256(b"".join(
            (root / name).read_bytes() for name in
            ("perf_config.py", "index_candidates.py", "index_candidates_cuda.py", "hc_post_pre.py",
             "attention_packed.py")
        )).hexdigest()
        return fields


FLAGS = {
    "index_candidates": "DSV41_INDEX_CANDIDATES",
    "draft_router": "DSV41_DRAFT_ROUTER_FUSED",
    "hc_post_pre": "DSV41_HC_POST_PRE",
    "attention_direct": "DSV41_ATTN_DIRECT",
    "engram_h2d": "DSV41_ENGRAM_ASYNC_H2D",
}
