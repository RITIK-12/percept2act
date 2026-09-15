#!/usr/bin/env python
"""Measure the detector on NPU vs GPU vs CPU, and check score separation.

Produces the two things the demo needs:

  1. the stage/device/latency table, measured rather than asserted -- 20 of the
     100 points are for appropriate NPU/GPU/CPU placement, and the rubric asks
     you to justify each choice with observed results;
  2. proof the detector actually separates good from damaged bricks, which is
     what decides the threshold and the uncertainty band.

Usage
-----
  python scripts/09_benchmark.py                    # latency on all devices
  python scripts/09_benchmark.py --detector-only    # score separation only
  python scripts/09_benchmark.py --iters 100

Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from percept2act.config import Scenario  # noqa: E402
from percept2act.detector import Detector  # noqa: E402


def load_set(d: Path) -> list[np.ndarray]:
    imgs = []
    for p in sorted(list(d.glob("*.png")) + list(d.glob("*.jpg"))):
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    return imgs


def bench_devices(scn: Scenario, iters: int) -> None:
    import openvino as ov

    available = ov.Core().available_devices
    h, w = scn.require("detector.input_size")
    dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    print(f"benchmarking {iters} inferences per device; available: {available}\n")
    header = f"{'device':<8} {'mean':>10} {'p50':>10} {'p95':>10} {'fps':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for device in ("NPU", "GPU", "CPU"):
        if device not in available:
            print(f"{device:<8} {'unavailable':>10}")
            continue
        try:
            det = Detector(scn, device=device)
            det.warmup(5)  # exclude graph compilation from the numbers
            samples = [det.score(dummy).latency_ms for _ in range(iters)]
        except Exception as exc:  # noqa: BLE001
            print(f"{device:<8} failed: {str(exc)[:60]}")
            continue

        ordered = sorted(samples)
        mean = statistics.fmean(samples)
        row = {
            "device": device,
            "mean": mean,
            "p50": statistics.median(samples),
            "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)],
            "fps": 1000.0 / mean,
        }
        rows.append(row)
        print(
            f"{device:<8} {row['mean']:>8.2f}ms {row['p50']:>8.2f}ms "
            f"{row['p95']:>8.2f}ms {row['fps']:>8.1f}"
        )

    if rows:
        best = min(rows, key=lambda r: r["mean"])
        configured = scn.device_for("detector")
        print(f"\nfastest: {best['device']} at {best['mean']:.2f}ms")
        print(f"configured: openvino.devices.detector = {configured}")
        if best["device"] != configured:
            print(
                f"\nNote: {best['device']} measured faster than the configured "
                f"{configured}.\nThat does not automatically mean switch -- the NPU is "
                "chosen partly to\nkeep the CPU free for the 30 Hz control loop. Say that "
                "out loud rather\nthan just quoting the fastest number."
            )


def check_separation(scn: Scenario) -> None:
    normals = load_set(REPO / str(scn.require("detector.normals_dir")))
    defects = load_set(REPO / str(scn.require("detector.defects_dir")))
    print(f"\nscoring {len(normals)} normals and {len(defects)} defects\n")
    if not normals:
        print("no normals captured yet -- run scripts/06_capture_normals.py")
        return

    det = Detector(scn)
    det.warmup()
    good = [det.score(i).score for i in normals]
    bad = [det.score(i).score for i in defects] if defects else []

    def describe(name: str, xs: list[float]) -> None:
        if not xs:
            return
        print(
            f"  {name:<10} n={len(xs):<4} min={min(xs):.4f} "
            f"mean={statistics.fmean(xs):.4f} max={max(xs):.4f}"
        )

    describe("good", good)
    describe("defective", bad)

    print(f"\n  current threshold : {det.threshold:.4f}")
    print(f"  uncertainty band  : +/-{det.band:.4f}")

    if not bad:
        print(
            "\nNo defective samples, so the threshold is unvalidated. Capture a few:\n"
            "  python scripts/06_capture_normals.py --class defective --auto 15"
        )
        return

    gap = min(bad) - max(good)
    midpoint = (max(good) + min(bad)) / 2
    if gap > 0:
        print(f"\n  clean separation, gap {gap:.4f}")
        print(f"  suggested threshold: {midpoint:.4f}")
        print(f"  suggested band     : {max(gap / 3, 0.01):.4f}")
    else:
        overlap = -gap
        print(f"\n  !! distributions OVERLAP by {overlap:.4f}")
        print(f"  suggested threshold: {midpoint:.4f}")
        print(f"  suggested band     : {max(overlap, 0.02):.4f}")
        print(
            "\n  Widen the band so overlapping scores route to `unsure` instead of\n"
            "  being guessed. Scoring an honest 'unsure' path beats a confident\n"
            "  wrong answer under the reliability criterion.\n"
            "  Then: more normals, tighter crop, or steadier lighting."
        )

    n_unsure = sum(1 for s in good + bad if abs(s - det.threshold) <= det.band)
    print(f"\n  at the current settings, {n_unsure}/{len(good)+len(bad)} samples read as unsure")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--detector-only", action="store_true", help="skip device sweep")
    args = ap.parse_args()

    scn = Scenario.load()
    if not args.detector_only:
        bench_devices(scn, args.iters)
    check_separation(scn)

    print(
        "\nRecord the table above in docs/DEMO.md. The rubric asks for architecture,\n"
        "workload placement, optimization choices, and observed results -- this is\n"
        "the observed-results half."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
