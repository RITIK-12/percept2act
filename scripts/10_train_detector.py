#!/usr/bin/env python
"""Fit PatchCore on the captured normals and export it to OpenVINO IR.

PatchCore builds a memory bank of patch features from normal images and scores a
new image by distance to its nearest neighbours in that bank. There is no
backprop, so this fits in minutes on a few dozen images -- which is exactly what
a Challenge Day defect reveal allows for.

PatchCore rather than PaDiM because the defect is physical damage (chips,
cracks, missing studs): localized structural differences, which nearest-neighbour
patch matching localizes well.

Usage
-----
  python scripts/10_train_detector.py                  # fit + export FP16
  python scripts/10_train_detector.py --int8           # additionally export INT8
  python scripts/10_train_detector.py --skip-fit       # re-export an existing ckpt

Then benchmark device placement:  python scripts/11_benchmark.py
Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from percept2act.config import Scenario  # noqa: E402


def count_images(d: Path) -> int:
    return len(list(d.glob("**/*.png"))) + len(list(d.glob("**/*.jpg"))) if d.exists() else 0


def brick_fraction(img) -> float:
    """Fraction of the crop that looks like a saturated brick rather than bare mat."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    return float(((s > 70) & (v > 60)).mean())


def quarantine_empty(d: Path, min_fraction: float = 0.02) -> int:
    """Move brick-less crops aside.

    Auto-capture fires on a timer, so the first frames often land before the
    brick is in place. Training on those teaches PatchCore that an empty mat is
    normal, which drags the decision boundary toward "anything with a brick in
    it is unusual".
    """
    import cv2

    if not d.exists():
        return 0
    rejects = d.parent / f"{d.name}_rejected"
    moved = 0
    for p in sorted(list(d.glob("**/*.png")) + list(d.glob("**/*.jpg"))):
        img = cv2.imread(str(p))
        if img is None or brick_fraction(img) < min_fraction:
            rejects.mkdir(parents=True, exist_ok=True)
            p.rename(rejects / p.name)
            moved += 1
    if moved:
        print(f"  quarantined {moved} brick-less crop(s) -> {rejects}")
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--int8", action="store_true", help="also export an INT8 model")
    ap.add_argument("--skip-fit", action="store_true", help="export only, reuse last checkpoint")
    ap.add_argument("--epochs", type=int, default=1, help="PatchCore needs exactly 1")
    args = ap.parse_args()

    scn = Scenario.load()
    normals = REPO / str(scn.require("detector.normals_dir"))
    defects = REPO / str(scn.require("detector.defects_dir"))
    export_root = REPO / str(scn.require("detector.export_dir"))
    h, w = scn.require("detector.input_size")

    print("screening captures for brick-less frames...")
    quarantine_empty(normals)
    quarantine_empty(defects)

    n_normal, n_abnormal = count_images(normals), count_images(defects)
    print(f"normals  : {n_normal:>4}  {normals}")
    print(f"defects  : {n_abnormal:>4}  {defects}   (threshold tuning only)")
    if n_normal < 20:
        print("\n! Need at least ~20 normals (50 is better).")
        print("  python scripts/09_capture_bricks.py --class good --auto 60")
        return 1

    # Imported late: anomalib pulls in torch and timm, which takes a few seconds.
    from anomalib.data import Folder
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    datamodule = Folder(
        name="brick",
        normal_dir=str(normals),
        abnormal_dir=str(defects) if n_abnormal else None,
        # With no labelled defects, hold back 20% of normals so Anomalib can
        # still compute a threshold instead of defaulting to a guess.
        normal_split_ratio=0.2 if not n_abnormal else 0.0,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
    )

    model = Patchcore(
        backbone=scn.require("detector.backbone"),
        layers=tuple(scn.require("detector.layers")),
        coreset_sampling_ratio=scn.require("detector.coreset_sampling_ratio"),
        num_neighbors=scn.require("detector.num_neighbors"),
    )

    engine = Engine(max_epochs=args.epochs, default_root_dir=str(export_root))

    if not args.skip_fit:
        print("\n=== fitting PatchCore (memory bank, no backprop) ===")
        engine.fit(model=model, datamodule=datamodule)
        print("\n=== validating ===")
        try:
            results = engine.test(model=model, datamodule=datamodule)
            print(json.dumps(results, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            print(f"(test skipped: {exc})")

    exports: dict[str, str] = {}

    print("\n=== exporting OpenVINO IR (FP16) ===")
    fp16_dir = export_root / "openvino_fp16"
    path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=str(fp16_dir),
        input_size=(h, w),
    )
    print(f"  -> {path}")
    exports["fp16"] = str(path)

    if args.int8:
        print("\n=== exporting OpenVINO IR (INT8 post-training quantization) ===")
        try:
            int8_dir = export_root / "openvino_int8"
            path8 = engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=str(int8_dir),
                input_size=(h, w),
                compression_type=CompressionType.INT8_PTQ,
                datamodule=datamodule,
            )
            print(f"  -> {path8}")
            exports["int8"] = str(path8)
        except Exception as exc:  # noqa: BLE001
            print(f"  INT8 export failed ({exc}); keeping FP16. This is fine --")
            print("  quantization is step 5 on the fallback ladder, not a requirement.")

    # Record where the IR landed so the detector service does not have to guess.
    manifest = export_root / "exports.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(exports, indent=2))

    # Surface the threshold Anomalib settled on: it seeds detector.threshold.
    threshold = None
    for attr in ("image_threshold", "pixel_threshold"):
        t = getattr(model, attr, None)
        value = getattr(t, "value", None) if t is not None else None
        if value is not None:
            threshold = float(value)
            print(f"\n{attr} = {threshold:.4f}")
            break

    print("\n" + "=" * 68)
    print(f"exports written: {json.dumps(exports, indent=2)}")
    print(f"manifest       : {manifest}")
    if threshold is not None:
        print(
            f"\nPut this in config/scenario.yaml:\n"
            f"  detector:\n    threshold: {threshold:.4f}"
        )
    print(
        "\nThen sanity-check separation between good and damaged bricks:\n"
        "  python scripts/11_benchmark.py --detector-only\n"
        "If the two score distributions overlap, widen detector.uncertainty_band\n"
        "rather than pretending the detector is confident."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
