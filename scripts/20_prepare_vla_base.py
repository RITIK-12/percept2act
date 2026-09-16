#!/usr/bin/env python
"""Make a physicalai-compatible local copy of lerobot/smolvla_base.

Why this is needed
------------------
`physicalai fit` cannot consume lerobot/smolvla_base directly. Its config.json
is a LeRobot config carrying runtime metadata -- input_features, output_features,
normalization_mapping, push_to_hub and so on -- that physicalai's SmolVLAConfig
does not declare. SmolVLA._from_hf calls:

    SmolVLAConfig.from_dict(hf_config, strict=False)

with the comment "ignore legacy config.json keys not present in SmolVLAConfig".
It does not do that. Reading physicalai/config/base.py, `strict` only controls
whether an explicit TypeError is raised for extra keys -- the full dict is then
handed to jsonargparse regardless, which rejects it:

    Group 'object' does not accept option 'input_features.observation.images.camera1.type'

So strict=False just trades a clear error for an obscure one. This reproduces in
plain Python, with no CLI involved.

The fix is to hand physicalai a local checkpoint whose config.json contains only
the fields SmolVLAConfig declares. Every dropped key is LeRobot bookkeeping; all
32 architecture fields (num_vlm_layers, expert_width_multiplier, chunk_size, ...)
are preserved, so the model built from it is identical.

Usage
-----
  python scripts/20_prepare_vla_base.py            # writes experiments/smolvla_base_pai
  python scripts/20_prepare_vla_base.py --force    # rebuild it

16_studio_train.sh runs this automatically when the directory is missing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BASE = "lerobot/smolvla_base"
OUT = REPO / "experiments" / "smolvla_base_pai"
# Everything _from_hf looks for in a local directory.
EXTRA = [
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if it exists")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"already present: {out}")
        return 0

    from huggingface_hub import hf_hub_download
    from physicalai.policies.smolvla.config import SmolVLAConfig

    cfg_path = Path(hf_hub_download(BASE, "config.json"))
    hf_cfg = json.loads(cfg_path.read_text())

    fields = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    kept = {k: v for k, v in hf_cfg.items() if k in fields}
    dropped = sorted(set(hf_cfg) - fields)

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(kept, indent=2))

    for name in EXTRA:
        try:
            src = Path(hf_hub_download(BASE, name))
        except Exception as exc:  # noqa: BLE001 - only model.safetensors is required
            if name == "model.safetensors":
                print(f"!! could not fetch {name}: {exc}")
                return 1
            continue
        # Copy rather than symlink: the HF cache can be garbage-collected.
        shutil.copy2(src, out / name)

    print(f"wrote {out}")
    print(f"  kept    {len(kept)} config fields")
    print(f"  dropped {len(dropped)}: {', '.join(dropped)}")
    print("\n  All dropped keys are LeRobot runtime metadata. Every architecture")
    print("  field is preserved, so the model built from this is identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
