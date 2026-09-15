#!/usr/bin/env bash
# Confirm the stage-2 dataset really contains both instruction strings with a
# sane balance. A SmolVLA fine-tune on a one-sided dataset will happily train
# and then ignore the instruction at inference -- catch that here, not in the demo.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$HOME/miniforge3/envs/hack_lerobot/bin/python" - <<PY
import sys, json, collections
from pathlib import Path
sys.path.insert(0, "$REPO/src")
root = Path("$REPO/datasets/stage2_sort")
if not root.exists():
    sys.exit("no stage2_sort dataset yet")

import pandas as pd
# --resume writes NEW chunk files, so every check must glob all of them. Reading
# only file-000 reports half the dataset and looks like the second class is
# missing entirely.
eps = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/**/*.parquet"))])
counts = collections.Counter(
    (list(r["tasks"])[0] if len(r["tasks"]) else "??") for _, r in eps.iterrows()
)
print(f"{len(eps)} episodes:")
for _, r in eps.sort_values("episode_index").iterrows():
    task = list(r["tasks"])[0] if len(r["tasks"]) else "??"
    print(f"  ep {r['episode_index']:>2}  {r['length']:>4} frames  {task!r}")

print("\nbalance:")
for k, v in sorted(counts.items()):
    print(f"  {v:>3}  {k!r}")

if len(counts) < 2:
    print("\n!! only one instruction present -- record the other class before training")
else:
    lo, hi = min(counts.values()), max(counts.values())
    print("\nBalance looks fine." if hi <= 2 * lo else f"\n!! unbalanced ({lo} vs {hi})")
PY
