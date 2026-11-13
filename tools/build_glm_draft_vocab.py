"""Write the GLM tokenizer's draft-vocabulary ranking (GLM_DRAFT_VOCAB, rust/crates/tidelm-glm/src/flash.rs).

    python3 tools/build_glm_draft_vocab.py --tokenizer <model dir>/tokenizer.json \
        --source chat:3:corpus/chat/perfectblend.parquet:max=80000000 \
        --source zh:2:corpus/wiki/zh.parquet:max=40000000 \
        --source zh:2:corpus/chat/alpaca_gpt4_zh.jsonl:keys=instruction_zh+input_zh+output_zh \
        --source agent:2:corpus/chat/glm53_public_seqs.jsonl:src=claude-code+openhands+mini-swe-agent \
        --source code:3:rust/ --out engine/glm_draft_vocab_ranking_v1.txt

The MTP draft passes of GLM-5.3-Flash on one box score only the ranking's first N ids (the draft
head); a token outside them cannot be drafted, which costs that round's remaining drafts and never a
token of output. So the list wants every language's common tokens, not one corpus's: each group's
counts are normalized to frequencies before the weighted sum (tools/build_draft_vocab.py, the DS
tokenizer's, does the same), and the tokenizer's added tokens (the template's and the tool calls')
come first whatever their counts.

A source is GROUP:WEIGHT:PATH[:opt=value,...]:
  max=CHARS     read at most this many characters of it
  keys=a+b      a record's fields to read (default: every string field but urls, ids and roles)
  src=a+b       for records with a `src` field, only these
A group of weight 0 is not ranked, only reported: what the list costs a language it leaves out.
PATH is a .parquet / .jsonl / .json of records, or a directory or file of source text. A record with
an `ids` list is taken as already tokenized (this tokenizer's ids). Every `--holdout`-th text of a
source is kept out of the counts; the header reports each group's held-out tokens unlisted at
`--totals` listed ids (the ranking's first ids, then the lowest unlisted: draft_vocab::select).

Needs `tokenizers` and, for parquet, `pyarrow`.
"""
import argparse
import collections
import glob
import json
import os

EXTS = ('.md', '.txt', '.py', '.cpp', '.cu', '.h', '.rs', '.go', '.js', '.ts', '.sh', '.toml', '.html')
SKIP = {'url', 'id', 'index', 'prompt_id', 'from', 'role', 'source', 'key', 'src'}


def strings(o, keys):
    """The text of a record: its string fields (those in `keys`, or all but SKIP), joined."""
    out, stack = [], [(None, o)]
    while stack:
        k, v = stack.pop()
        if isinstance(v, str):
            if (keys and k in keys) or (not keys and k not in SKIP):
                out.append(v)
        elif isinstance(v, dict):
            stack.extend(reversed(list(v.items())))
        elif isinstance(v, list):
            stack.extend((k, x) for x in reversed(v))
    return '\n'.join(out)


def records(path):
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    elif path.endswith('.jsonl'):
        for line in open(path, encoding='utf-8', errors='ignore'):
            try:
                yield json.loads(line)
            except ValueError:
                continue
    elif path.endswith('.json'):
        d = json.load(open(path, encoding='utf-8'))
        yield from (d if isinstance(d, list) else [d])
    elif os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, '**', '*'), recursive=True)):
            if f.endswith(EXTS) and os.path.isfile(f) and '/target/' not in f:
                yield open(f, encoding='utf-8', errors='ignore').read()
    else:
        yield open(path, encoding='utf-8', errors='ignore').read()


def read(path, opts):
    """A source's items: strings, or id lists for pre-tokenized records."""
    keys = set(opts['keys'].split('+')) if 'keys' in opts else None
    srcs = set(opts['src'].split('+')) if 'src' in opts else None
    cap, used = int(opts.get('max', 0)), 0
    for r in records(path):
        if isinstance(r, dict):
            if srcs and r.get('src') not in srcs:
                continue
            item = r['ids'] if isinstance(r.get('ids'), list) else strings(r, keys)
        else:
            item = r
        if not item:
            continue
        yield item
        used += len(item) * (4 if isinstance(item, list) else 1)
        if cap and used >= cap:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--vocab', type=int, default=154880, help="the model's vocabulary (its head's rows)")
    ap.add_argument('--source', action='append', required=True, metavar='GROUP:WEIGHT:PATH[:opts]')
    ap.add_argument('--holdout', type=int, default=10)
    ap.add_argument('--totals', default='24576,32768,40960,49152,65536,81920')
    ap.add_argument('--top', type=int, default=98304, help='ids written')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dump', help="also write every group's counts (train, held-out) to this JSON, for studies")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    def count(items, into):
        texts = [x for x in items if isinstance(x, str)]
        for x in items:
            if isinstance(x, list):
                into.update(x)
        for i in range(0, len(texts), 256):
            for e in tok.encode_batch(texts[i:i + 256], add_special_tokens=False):
                into.update(e.ids)

    train = collections.defaultdict(collections.Counter)
    held = collections.defaultdict(collections.Counter)
    weights, names = {}, collections.defaultdict(list)
    for spec in args.source:
        group, weight, rest = spec.split(':', 2)
        path, _, o = rest.partition(':')
        opts = dict(kv.split('=', 1) for kv in o.split(',') if kv)
        if weights.setdefault(group, float(weight)) != float(weight):
            raise SystemExit(f'group {group} given two weights')
        items = list(read(path, opts))
        count([x for i, x in enumerate(items) if i % args.holdout], train[group])
        count(items[::args.holdout], held[group])
        names[group].append(os.path.basename(path.rstrip('/')) + (f' ({o})' if o else ''))
        print(f'{group:6s} {path}: {len(items)} texts, group at {sum(train[group].values())} tokens', flush=True)

    if args.dump:
        json.dump({'weights': weights, 'train': {g: dict(c) for g, c in train.items()},
                   'held': {g: dict(c) for g, c in held.items()}}, open(args.dump, 'w'))
    score = collections.defaultdict(float)
    for group, c in train.items():
        total = sum(c.values())
        for i, n in c.items():
            if i < args.vocab and weights[group] > 0:
                score[i] += weights[group] * n / total
    special = sorted(i for i in tok.get_added_tokens_decoder() if i < args.vocab)
    first = set(special)
    order = (special + [i for i in sorted(score, key=lambda i: (-score[i], i)) if i not in first])[:args.top]

    totals = [int(t) for t in args.totals.split(',')]
    lines = [f'# GLM tokenizer draft vocabulary ranking (tools/build_glm_draft_vocab.py): {len(order)} ids, the',
             f"# tokenizer's {len(special)} added tokens, then most frequent first. Per-group frequencies, weighted:"]
    for g in train:
        lines.append(f'#   {g} x{weights[g]:g}: {sum(train[g].values())} tokens, {len(train[g])} distinct ids - ' + ', '.join(names[g]))
    lines.append(f'# Held-out tokens unlisted (every {args.holdout}th text of each source), by listed ids:')
    lines.append('#   group   ' + ' '.join(f'{t:>7d}' for t in totals))
    for g, c in held.items():
        n, row = sum(c.values()), []
        for t in totals:
            listed = set(order[:t])
            fill = (i for i in range(args.vocab) if i not in listed)
            while len(listed) < t:
                listed.add(next(fill))
            row.append(100 * sum(k for i, k in c.items() if i not in listed) / max(1, n))
        lines.append(f'#   {g:7s} ' + ' '.join(f'{m:6.2f}%' for m in row) + f'   ({n} tokens)')
    print('\n'.join(lines), flush=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        for i in range(0, len(order), 16):
            f.write(' '.join(map(str, order[i:i + 16])) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
