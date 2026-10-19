"""The routing-statistics DB at boot, and the bundled seed a fresh install starts from.

Without a DB, the resident set falls back to the router-bias placeholder (the same number of
experts in every layer, ranked by the router's load-balancing bias). The seed is a DB grown
elsewhere -- aggregate router mass per (layer, expert) and a token count, no text -- so a new
install places by measured demand from its first boot. It is read only when GLM_STATS_DB does
not exist yet; the engine's first save writes it to that path, and from then on the user's own
traffic accumulates on top of it. Rank 0 alone reads either file; the placement it ranks is
broadcast, so the two ranks cannot disagree about it.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "profiles" / "glm53-experts-v1.pt"


def _read(raw: bytes, shape):
    db = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    mass, tokens = db["mass"], int(db["tokens"])
    if tuple(mass.shape) != tuple(shape):
        return None, 0, f"shape {tuple(mass.shape)}, not this model's {tuple(shape)}"
    mass = mass.double()
    if not torch.isfinite(mass).all() or (mass < 0).any():
        return None, 0, "non-finite or negative mass"
    return mass, tokens, None


def load_stats(db_path: str, shape, seed: str = "1"):
    """(mass float64 [layers, experts], tokens, note) for GLM_STATS_DB=db_path.

    The local DB when the file exists -- a file of another shape starts empty, as before the
    seed existed. Otherwise GLM_STATS_SEED: "1" the bundled seed, "0" none, anything else a path.
    ``note`` is a line for the boot log, or None when there is nothing to say."""
    empty = torch.zeros(tuple(shape), dtype=torch.float64)
    path = Path(db_path)
    if path.exists():
        mass, tokens, why = _read(path.read_bytes(), shape)
        if mass is None:
            return empty, 0, f"stats DB {path} ignored ({why}): starting empty"
        return mass, tokens, None
    if seed == "0":
        return empty, 0, None
    src = DEFAULT_PATH if seed == "1" else Path(seed)
    if not src.exists():
        return empty, 0, f"no stats DB at {path} and no seed at {src}: starting empty"
    raw = src.read_bytes()
    mass, tokens, why = _read(raw, shape)
    if mass is None or tokens <= 0:
        return empty, 0, f"seed {src.name} ignored ({why or 'no tokens'}): starting empty"
    return mass, tokens, (f"no stats DB at {path} yet: starting from the seed {src.name} "
                          f"({hashlib.sha256(raw).hexdigest()[:12]}, {tokens} tokens of routing)")
