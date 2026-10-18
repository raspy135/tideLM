"""Adaptive expert loading: two knobs, everything else derived.

Adaptation used to take ~13 DSV41_PRUNE_* settings, several of which changed meaning with
DSV41_PRUNE_UNIT (routing slots vs requests). They only ever expressed two decisions:

  DSV41_ADAPT_SENSITIVITY  how far one request can move the expert ranking.
      off | low | medium | high | max  = demand half-life of never / 40 / 20 / 10 / 5 requests,
      or a number s in (0, 0.5): the newest request's share of the observed demand, so the
      half-life is ln 0.5 / ln(1 - s) requests. `medium` is the 2026-09 serving profile
      (half-life 20). Higher follows a changing workload faster and swaps more experts per
      request; too high chases each prompt and raises the miss rate (docs/gotchas.md,
      "The demand half-life has to be read against your traffic").
  DSV41_ADAPT_PRIOR  how many requests' worth of weight the shipped routing trace keeps
      against observed demand (default 8). Lower lets this server's own traffic dominate sooner.

With either knob set, the engine derives the rest -- request units, swaps on (off at
sensitivity `off`, or with an explicit legacy DSV41_PRUNE_SWAP=0, which scripts use to freeze
placement; likewise DSV41_PRUNE_SWAP_PREFILL=0), prefill-boundary swaps on, decode-time passes every 600 output tokens, a miss gate that
scales inversely with
sensitivity (2% at medium, 1% at high), a 512-swap cap, a 0.005 gain floor -- and ignores the
old per-setting variables, naming them in the startup log. With neither set, every legacy
DSV41_PRUNE_* variable is read exactly as before, with the same defaults.

DSV41_PRUNE_METRIC=frequency (default) ranks by selection counts; score uses the sum of raw
positive router scores for the unmasked top-k picks. In request units each request contributes
one normalized score distribution per layer. This is router preference, not measured expert
output contribution. The shipped frequency trace remains the cold-start prior in either mode.

Resolved once at import from the environment, which the launcher forwards identically to both
ranks; the EP2 boot guard compares the resolved values (`boot_fields`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

LEVELS = {"off": None, "low": 40.0, "medium": 20.0, "high": 10.0, "max": 5.0}   # half-life, requests
DEFAULT_PRIOR = 8.0
DECODE_TOKENS = 600
REQUEST_DB = "results/prune_demand_req.npz"
# Legacy settings the two knobs replace; reported (not read) when a knob is set.
REPLACED = ("DSV41_PRUNE_UNIT", "DSV41_PRUNE_PRIOR", "DSV41_PRUNE_HALFLIFE", "DSV41_PRUNE_SWAP",
            "DSV41_PRUNE_SWAP_PREFILL", "DSV41_PRUNE_SWAP_PREFILL_MIN",
            "DSV41_PRUNE_SWAP_PREFILL_MIN_MISS", "DSV41_PRUNE_SWAP_MAX", "DSV41_PRUNE_SWAP_MIN_GAIN",
            "DSV41_PRUNE_SWAP_MIN_GROWTH", "DSV41_PRUNE_SWAP_MIN_GROWTH_REQ", "DSV41_PRUNE_MISS")


@dataclass(frozen=True)
class AdaptConfig:
    source: str                  # "knobs" or "legacy"
    metric: str                  # "frequency" or pre-mask router "score"
    record: bool                 # record routing demand (DSV41_PRUNE_MISS)
    use_db: bool                 # blend the saved demand DB into the boot ranking (DSV41_PRUNE_ADAPT)
    request_unit: bool           # one vote per request (else per routing slot)
    db_path: str
    prior: float                 # trace weight: requests (request unit) or slots per layer
    halflife: float              # demand half-life, same unit as prior
    swap: bool                   # end-of-request swaps
    swap_prefill: bool           # prefill -> decode boundary swaps
    swap_prefill_min: int        # newly prefilled tokens needed for a boundary pass
    swap_prefill_min_miss: float  # ...and this prompt's own miss rate
    swap_max: int
    swap_min_gain: float         # x the layer's mean score
    min_growth_req: int          # requests between plans (request unit)
    min_growth_slots: float      # new slots between plans (slot unit)
    decode_tokens: int = 0       # decode-time pass every this many output tokens (0 = never)
    sensitivity: float | None = None   # newest request's share of observed demand (knobs only)
    level: str | None = None
    ignored: tuple = field(default_factory=tuple)

    def describe(self) -> str:
        if self.source == "legacy":
            return (f"expert adaptation from legacy DSV41_PRUNE_* settings "
                    f"({'request' if self.request_unit else 'slot'} units, prior {self.prior:g}, "
                    f"half-life {self.halflife:g}, swaps {'on' if self.swap else 'off'})")
        if not self.swap:
            s = "off (ranking frozen; demand still recorded)"
        else:
            s = (f"{self.level or 'custom'} ({self.sensitivity:.3f} of observed demand per request, "
                 f"half-life {self.halflife:.1f} requests)")
        msg = (f"expert adaptation: sensitivity {s}, prior {self.prior:g} requests; derived: "
               f"prefill swaps {'on' if self.swap_prefill else 'off'} at >= {self.swap_prefill_min} "
               f"new tokens and >= {self.swap_prefill_min_miss:.1%} misses, <= {self.swap_max} swaps "
               f"per pass, gain floor {self.swap_min_gain}, decode passes "
               + (f"every {self.decode_tokens} tokens at the same miss gate" if self.decode_tokens else "off"))
        if self.ignored:
            msg += f"; ignoring {', '.join(self.ignored)} (replaced by DSV41_ADAPT_*)"
        return msg

    def boot_fields(self) -> dict:
        """What both ranks must agree on: anything that decides whether a collective happens
        (the prefill-boundary pass) or what the ranking/mask is built from."""
        return {"adapt_source": self.source, "adapt_request_unit": self.request_unit,
                "prune_metric": self.metric, "prune_score_history_version": 2,
                "prune_record": self.record,
                "adapt_prior": self.prior, "adapt_halflife": round(self.halflife, 6),
                "adapt_swap": self.swap, "adapt_decode_tokens": self.decode_tokens,
                "prune_swap_prefill": "1" if self.swap_prefill else "0",
                "prune_swap_prefill_min": str(self.swap_prefill_min)}


def _sensitivity(raw: str) -> tuple[str | None, float | None]:
    """-> (level name or None, half-life in requests or None for off)."""
    v = raw.strip().lower()
    if v in LEVELS:
        return v, LEVELS[v]
    try:
        s = float(v)
    except ValueError:
        raise ValueError(f"DSV41_ADAPT_SENSITIVITY={raw!r}: use off, low, medium, high, max "
                         f"or a number in [0, 0.5)") from None
    if not 0 <= s < 0.5:
        raise ValueError(f"DSV41_ADAPT_SENSITIVITY={raw!r}: a number must be in [0, 0.5); 0.034 "
                         f"is medium (half-life 20), 0.5 would be a half-life of one request")
    if s == 0:
        return "off", None
    return None, math.log(0.5) / math.log(1 - s)


def _decode_tokens(raw: str | None) -> int:
    """DSV41_ADAPT_DECODE_TOKENS: optional override of the decode-pass interval; 0 turns decode
    passes off (the rollback if they ever misbehave), leaving the rest of adaptation alone."""
    if raw is None:
        return DECODE_TOKENS
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"DSV41_ADAPT_DECODE_TOKENS={raw!r}: output tokens between passes, 0 = off") from None
    if n < 0:
        raise ValueError(f"DSV41_ADAPT_DECODE_TOKENS={raw!r} must be >= 0")
    return n


def resolve(env=None) -> AdaptConfig:
    env = os.environ if env is None else env
    get = env.get
    if get("DSV41_ADAPT_URGENT", "0") not in ("", "0"):
        # Removed 2026-10-18 (docs/urgent-expert-loading.md): a 64-swap pass when ~30 tokens
        # missed > 10%, never measured as a quality gain and ~0.2 s a pass. DSV41_HOT_ASYNC reads
        # the experts a step misses most, at no measured cost (docs/hot-reads.md).
        raise ValueError("DSV41_ADAPT_URGENT was removed; DSV41_HOT_READS=1 DSV41_HOT_ASYNC=1 replaces it "
                         "(docs/hot-reads.md). Set it to 0 or delete the line.")
    metric = get("DSV41_PRUNE_METRIC", "frequency").strip().lower()
    if metric not in ("frequency", "score"):
        raise ValueError(f"DSV41_PRUNE_METRIC={metric!r}: use frequency or score")
    def db_path(request):
        default = REQUEST_DB if request else "results/prune_demand.npz"
        if metric == "score":
            default = default.removesuffix(".npz") + "_score.npz"
        return get("DSV41_PRUNE_DB", default)
    sens_raw, prior_raw = get("DSV41_ADAPT_SENSITIVITY"), get("DSV41_ADAPT_PRIOR")
    if sens_raw is None and prior_raw is None:
        request = get("DSV41_PRUNE_UNIT", "slot") == "request"
        return AdaptConfig(
            source="legacy",
            metric=metric,
            record=metric == "score" or get("DSV41_PRUNE_MISS", "0") == "1",
            use_db=get("DSV41_PRUNE_ADAPT", "1") == "1",
            request_unit=request,
            db_path=db_path(request),
            prior=float(get("DSV41_PRUNE_PRIOR", "2e7")),
            halflife=float(get("DSV41_PRUNE_HALFLIFE", "2e7")),
            swap=get("DSV41_PRUNE_SWAP", "0") == "1",
            swap_prefill=get("DSV41_PRUNE_SWAP_PREFILL", "0") == "1",
            swap_prefill_min=int(get("DSV41_PRUNE_SWAP_PREFILL_MIN", "1024")),
            swap_prefill_min_miss=float(get("DSV41_PRUNE_SWAP_PREFILL_MIN_MISS", "0.10")),
            swap_max=int(get("DSV41_PRUNE_SWAP_MAX", "64")),
            swap_min_gain=float(get("DSV41_PRUNE_SWAP_MIN_GAIN", "0.05")),
            min_growth_req=max(1, int(get("DSV41_PRUNE_SWAP_MIN_GROWTH_REQ", "1"))),
            min_growth_slots=float(get("DSV41_PRUNE_SWAP_MIN_GROWTH", "5e5")),
            decode_tokens=0,
        )
    level, halflife = _sensitivity(sens_raw if sens_raw is not None else "medium")
    try:
        prior = float(prior_raw) if prior_raw is not None else DEFAULT_PRIOR
    except ValueError:
        raise ValueError(f"DSV41_ADAPT_PRIOR={prior_raw!r}: a number of requests, e.g. 8") from None
    if prior < 0:
        raise ValueError(f"DSV41_ADAPT_PRIOR={prior_raw!r} must be >= 0")
    on = halflife is not None
    s = 1 - 0.5 ** (1 / halflife) if on else 0.0
    # The legacy OFF switches still work with the knobs: benchmarks and tests freeze placement
    # in-process with DSV41_PRUNE_SWAP=0 / DSV41_PRUNE_SWAP_PREFILL=0, and switching adaptation
    # off is always the safe direction. Any other legacy value is ignored and reported.
    freeze = get("DSV41_PRUNE_SWAP") == "0"
    no_prefill = freeze or get("DSV41_PRUNE_SWAP_PREFILL") == "0"
    honored = {k for k, off in (("DSV41_PRUNE_SWAP", freeze),
                                ("DSV41_PRUNE_SWAP_PREFILL", get("DSV41_PRUNE_SWAP_PREFILL") == "0")) if off}
    decode_tokens = _decode_tokens(get("DSV41_ADAPT_DECODE_TOKENS")) if (on and not freeze) else 0
    return AdaptConfig(
        source="knobs",
        metric=metric,
        record=True,                 # demand is the input to everything below, and the miss report
        use_db=get("DSV41_PRUNE_ADAPT", "1") == "1",
        request_unit=True,
        db_path=db_path(True),
        prior=prior,
        halflife=halflife if on else float("inf"),
        swap=on and not freeze,
        swap_prefill=on and not no_prefill,
        swap_prefill_min=32,
        # 2% at medium: a prompt must be served this badly before the boundary pass re-fits for
        # it. More sensitive -> act on smaller misses; clamped so neither end becomes absurd.
        swap_prefill_min_miss=min(0.2, max(0.005, 0.02 * (1 - 0.5 ** (1 / 20)) / s)) if on else 0.2,
        swap_max=512,
        swap_min_gain=0.005,
        min_growth_req=1,
        min_growth_slots=0.0,
        # ~30 s of decode at ~20 tok/s for a ~0.2 s pass (~1% overhead), gated on the decode
        # stretch's own miss rate like the prefill pass (swap_prefill_min_miss).
        decode_tokens=decode_tokens,
        sensitivity=s if on else 0.0,
        level=level,
        ignored=tuple(k for k in REPLACED if k in env and k not in honored),
    )


CFG = resolve()
