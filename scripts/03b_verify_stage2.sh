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

tasks_file = root / "meta" / "tasks.jsonl"
eps_file   = root / "meta" / "episodes.jsonl"
if tasks_file.exists():
    tasks = [json.loads(l) for l in tasks_file.read_text().splitlines() if l.strip()]
    print("task strings in dataset:")
    for t in tasks:
        print("  ", t)
if eps_file.exists():
    eps = [json.loads(l) for l in eps_file.read_text().splitlines() if l.strip()]
    counts = collections.Counter(
        tuple(e.get("tasks", [])) if isinstance(e.get("tasks"), list) else (e.get("tasks"),)
        for e in eps
    )
    print(f"\n{len(eps)} episodes total:")
    for k, v in counts.items():
        print(f"  {v:>3}  {k}")
    if len(counts) < 2:
        print("\n!! only one instruction present — record the other class before training")
    else:
        lo, hi = min(counts.values()), max(counts.values())
        if hi > 2 * lo:
            print(f"\n!! unbalanced ({lo} vs {hi}) — record more of the smaller class")
        else:
            print("\nBalance looks fine.")
PY
