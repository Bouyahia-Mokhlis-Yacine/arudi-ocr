"""Text -> arudi conversion: batched greedy decoding, loop guard, MBR ensembling."""
import re

import torch

try:  # rapidfuzz is faster, but optional (e.g. offline Kaggle notebooks don't ship it)
    from rapidfuzz.distance import Levenshtein
    _lev = Levenshtein.distance
except ImportError:
    def _lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

from .models import PAD, BOS, EOS, UNK

LOOP = re.compile(r"(.{1,4}?)\1{3,}")


def looks_broken(pred, src):
    r = len(pred) / max(1, len(src))
    return LOOP.search(pred) is not None or r > 1.75 or r < 0.9 or not pred


@torch.no_grad()
def greedy(model, itos, texts, batch_size=64, device="cpu"):
    stoi = {c: i for i, c in enumerate(itos)}
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        seqs = [[stoi.get(c, UNK) for c in s][: model.maxlen - 2] or [UNK] for s in chunk]
        X = torch.full((len(seqs), max(map(len, seqs))), PAD, device=device)
        for j, s in enumerate(seqs):
            X[j, :len(s)] = torch.tensor(s)
        cap = min(model.maxlen - 1, int(1.75 * X.size(1)) + 6)  # length cap: arudi is ~1.3x the source
        mem = model.encode(X)
        Y = torch.full((len(seqs), 1), BOS, device=device)
        done = torch.zeros(len(seqs), dtype=torch.bool, device=device)
        for _ in range(cap):
            nxt = model.decode(mem, X, Y)[:, -1].argmax(-1)
            nxt[done] = PAD
            Y = torch.cat([Y, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        out += ["".join(itos[t] for t in row[1:] if t > 3) for row in Y.tolist()]
    return out


def mbr(candidates):
    """Pick the candidate with the smallest total Levenshtein distance to all others."""
    return min(candidates, key=lambda c: (sum(_lev(c, h) for h in candidates), candidates.index(c)))


def convert(converters, texts, device="cpu"):
    """converters: list of (model, itos). One model -> greedy; several -> MBR over their outputs."""
    outs = [greedy(m, itos, texts, device=device) for m, itos in converters]
    preds = [mbr(list(c)) if len(c) > 2 else c[0] for c in zip(*outs)]
    # repair rare decoding loops by falling back to any non-broken member output
    return [p if not looks_broken(p, s) else next((c for c in cands if not looks_broken(c, s)), LOOP.sub(r"\1", p))
            for p, s, cands in zip(preds, texts, zip(*outs))]
