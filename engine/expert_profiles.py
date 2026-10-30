"""Residency ranking settings and the mask digest.

The fixed topic profiles that lived here (DSV41_EXPERT_TOPICS, DSV41_PRUNE_SOURCE=saliency,
DSV41_PRUNE_RANK=maxmin; the maxmin ranker ported from 0xBakeer/deepseek-v41-flash-spark) were
removed on 2026-10-17: measured 2026-10-06, they did not qualify as a default (9/14 against 12/14
on separate questions, 27% raw misses against 14%; docs/expert-profiles.md,
RESULTS.md). Residency ranks the shipped coding/general traces, adapted by the demand DB.
"""
import hashlib

import numpy as np

RANKS = ('sum', 'max')
VERSION = 2


def mask_digest(masks):
    if masks is None:
        return 'all-experts'
    h = hashlib.sha256()
    for layer in sorted(masks):
        h.update(int(layer).to_bytes(4, 'little'))
        m = masks[layer]
        if hasattr(m, 'detach'):
            m = m.detach().cpu().numpy()
        h.update(np.asarray(m, dtype=np.bool_).tobytes())
    return h.hexdigest()


class ExpertProfile:
    """How the coding and general traces combine into one residency ranking (DSV41_PRUNE_RANK)."""

    def __init__(self, env, n_layers=40, n_experts=384):
        source = (env.get('DSV41_PRUNE_SOURCE') or 'counts').strip()
        self.rank = (env.get('DSV41_PRUNE_RANK') or 'sum').strip()
        topics = (env.get('DSV41_EXPERT_TOPICS') or '').strip()
        if source != 'counts' or self.rank not in RANKS or topics:
            raise ValueError('expert selection is DSV41_PRUNE_RANK=sum|max on the routing counts; the fixed '
                             'topic profiles (saliency, maxmin, DSV41_EXPERT_TOPICS) were removed 2026-10-17 '
                             '-- see docs/expert-profiles.md')
        self.n_layers, self.n_experts = n_layers, n_experts
        self.topics = ('coding', 'general')

    def boot_fields(self):
        return {'expert_profile_version': VERSION, 'prune_source': 'counts', 'prune_rank': self.rank,
                'expert_topics': self.topics}

    def report(self):
        return {**self.boot_fields(), 'expert_topics': list(self.topics)}
