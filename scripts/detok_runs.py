#!/usr/bin/env python3
"""Decode the generated_ids logged by kimi_k3_dist_generate back to text.

Stage 4 compares a drafting run against a non-drafting one. Because the batched verify
path reduces over a different collective payload shape than sequential decode, the two are
not expected to be bit-identical (Stage 3 receipt), so the comparison that matters is
whether both read as valid model output — which needs the actual text, not just ids.

Usage: detok_runs.py RUN_DIR TAG [TAG ...]
"""
import os
import re
import sys

from transformers import AutoTokenizer

TOKENIZER_DIR = os.environ.get("K3_TOKENIZER_DIR", os.path.expanduser("~/models/k3-tokenizer"))
ID_RE = re.compile(r"prompt=(\d+) generated_ids=([\d,]+)")


def runs_for(run_dir, tag):
    path = os.path.join(run_dir, "draft_%s_r0.stderr" % tag)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = ID_RE.search(line)
            if m:
                out.append((int(m.group(1)), [int(x) for x in m.group(2).split(",") if x]))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    run_dir, tags = sys.argv[1], sys.argv[2:]
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    per_tag = {t: dict(runs_for(run_dir, t)) for t in tags}

    prompts = sorted({p for d in per_tag.values() for p in d})
    for p in prompts:
        print("=" * 78)
        print("PROMPT %d" % p)
        for t in tags:
            ids = per_tag[t].get(p)
            if ids is None:
                print("  [%s] (no output)" % t)
                continue
            print("  [%s] %d tokens" % (t, len(ids)))
            print("    %r" % tok.decode(ids))
        # Token-level agreement against the first tag, reported and never asserted: the
        # engine's own batched/sequential gap makes exact equality the wrong bar.
        base = per_tag[tags[0]].get(p)
        for t in tags[1:]:
            other = per_tag[t].get(p)
            if not base or not other:
                continue
            n = min(len(base), len(other))
            same = sum(1 for i in range(n) if base[i] == other[i])
            pref = 0
            while pref < n and base[pref] == other[pref]:
                pref += 1
            print("    agreement %s vs %s: %d/%d positions, common prefix %d"
                  % (tags[0], t, same, n, pref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
