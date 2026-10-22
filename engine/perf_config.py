"""Decode performance settings; both ranks must use identical settings and sources."""
from dataclasses import dataclass, asdict
import hashlib
import os
from pathlib import Path


@dataclass(frozen=True)
class DecodeExperiments:
    # The constructor describes an explicit baseline for component tests and
    # paired benchmarks. Serving defaults are selected by from_env below.
    index_candidates: bool = False
    draft_router: bool = False
    hc_post_pre: bool = False
    attention_direct: bool = False
    engram_h2d: bool = False

    @classmethod
    def from_env(cls):
        values = {}
        solo = os.environ.get("DSV41_MAX_CONCURRENCY", "1") == "1"
        for field, key in FLAGS.items():
            # Candidate scoring saved 2.39% at 24.6K tokens with identical
            # greedy output. Batched decode has not been qualified for it.
            default = "1" if field == "index_candidates" and solo else "0"
            value = os.environ.get(key, default)
            if value not in ("0", "1"):
                raise ValueError(f"{key} must be 0 or 1")
            values[field] = value == "1"
        # Batched decode composes its own HC/router boundaries. Until those
        # paths are qualified, do not silently mix experimental solo rounds
        # with baseline batched rounds under the same advertised configuration.
        if any(values.values()) and not solo:
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
