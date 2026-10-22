"""Host-known bounds and graph keys for opt-in decode indexer bucketing."""


def index_bucket(end: int, capacity: int) -> int:
    if capacity < 1 or not 0 <= end <= capacity:
        raise ValueError('index bound must fit the allocated context')
    # Split caches must retain at least 2048 candidates per rank, including
    # masked padding at short contexts. Unsplit MTP keeps the same top-k too.
    target = max(end, min(4096, capacity))
    return min(capacity, 1 << (target - 1).bit_length())


def graph_key(kind: str, rows: int, bound=None, *, logits=True):
    if kind == 'decode':
        return rows if bound is None else (kind, rows, bound)
    key = (kind, rows, logits) if kind == 'mtp' else (kind, rows)
    return key if bound is None else (*key, bound)


def bucket_keys(keys, bound):
    """Only opt-in keys have a trailing bucket; baseline keys stay unchanged."""
    return [key for key in keys if isinstance(key, tuple)
            and key[0] in ('decode', 'split', 'hot', 'mtp')
            and len(key) == (4 if key[0] == 'mtp' else 3) and key[-1] == bound]
