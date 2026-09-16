#!/usr/bin/env python
"""Export the trained SmolVLA to OpenVINO IR, bypassing `physicalai export`.

`physicalai export` loads via Lightning's load_from_checkpoint, so it only takes
a .ckpt from `physicalai fit`. Our policy was trained with lerobot-train, which
writes a LeRobot-format pretrained_model/ directory.

But the export logic itself lives on the Policy mixin as `policy.export(path,
backend="openvino")`, which works on an instance. So the checkpoint format only
has to be loadable, not Lightning-shaped:

  1. filter the trained config.json to the fields SmolVLAConfig declares
     (same incompatibility as scripts/20_prepare_vla_base.py -- see its docstring)
  2. construct SmolVLA from that directory, which loads the trained weights
  3. call policy.export(..., backend="openvino")

Usage:
  python scripts/21_export_policy_ov.py
  python scripts/21_export_policy_ov.py --backend onnx
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="openvino", choices=["openvino", "onnx", "torch"])
    ap.add_argument("--checkpoint", default=None, help="pretrained_model dir (default: newest)")
    args = ap.parse_args()

    from percept2act.config import Scenario

    scn = Scenario.load()
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        root = scn.abs_path("policies.stage2_sort.export_dir")
        found = sorted(root.glob("checkpoints/*/pretrained_model"))
        if not found:
            print(f"no checkpoint under {root}")
            return 1
        ckpt = found[-1]

    out = REPO / "experiments" / "smolvla_sort_ov"
    staged = REPO / "experiments" / "_smolvla_trained_pai"

    print(f"checkpoint : {ckpt}")
    print(f"backend    : {args.backend}")
    print(f"output     : {out}\n")

    from physicalai.policies.smolvla.config import SmolVLAConfig

    # Same filter as 20_prepare_vla_base.py: physicalai's SmolVLAConfig does not
    # declare LeRobot's input_features/output_features/normalization_mapping, and
    # from_dict(strict=False) does NOT drop them -- it passes them to jsonargparse
    # anyway, which rejects the whole config.
    hf_cfg = json.loads((ckpt / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    kept = {k: v for k, v in hf_cfg.items() if k in fields}

    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    (staged / "config.json").write_text(json.dumps(kept, indent=2))
    for f in ckpt.iterdir():
        if f.name != "config.json" and f.is_file():
            shutil.copy2(f, staged / f.name)
    print(f"staged {len(kept)} config fields -> {staged}")

    from physicalai.policies import SmolVLA

    policy = SmolVLA(pretrained_name_or_path=str(staged), num_cameras=0, image_key_reorder_map={})
    policy.eval()
    print(f"loaded trained policy: {type(policy).__name__}\n")

    out.mkdir(parents=True, exist_ok=True)
    policy.export(out, backend=args.backend)

    print(f"\nartifacts in {out}:")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
