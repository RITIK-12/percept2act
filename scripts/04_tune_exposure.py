#!/usr/bin/env python
"""Auto-tune the wrist camera's exposure, gain and brightness, then save it.

The Sonix module defaults to blowing out the frame and loses its settings on
every replug, so this sweeps the controls, scores each setting, picks the best
and writes it into config/scenario.yaml. Preflight re-applies it from there.

Scoring balances three things a policy input needs:
  * mid-range brightness -- detail in both the dark mat and the bright bricks
  * few saturated pixels -- blown highlights are unrecoverable information loss
  * high sharpness      -- longer exposures blur a moving wrist camera

Usage
-----
  python scripts/04_tune_exposure.py                 # tune the wrist cam
  python scripts/04_tune_exposure.py --camera front  # any UVC camera role
  python scripts/04_tune_exposure.py --no-save       # report only

Point the camera at the actual workspace first -- it tunes for what it sees.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from percept2act.config import Scenario  # noqa: E402

TARGET_MEAN = 118.0  # mid-grey, leaves headroom both ways
EXPOSURES = [20, 30, 40, 60, 80, 110, 150, 200, 300]
BRIGHTNESS = [0, 16, 32]


def v4l2_set(dev: str, ctrl: str, value: int) -> None:
    subprocess.run(
        ["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
        capture_output=True,
        check=False,
    )


def grab(cap: cv2.VideoCapture, n: int = 12) -> np.ndarray | None:
    best = None
    for _ in range(n):
        ok, frame = cap.read()
        if ok and frame is not None:
            best = frame
    return best


def score_frame(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(frame.mean())
    blown = float((frame > 250).mean())
    crushed = float((frame < 5).mean())
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Laplacian variance rises with overall brightness whether or not the lens
    # is focused, so raw sharpness would just pick the longest exposure. Divide
    # it out and use sharpness only to break ties between similar exposures.
    contrast = sharp / max(mean, 1.0)

    # Saturated pixels are unrecoverable, so they are penalised hard and with a
    # cliff above 5%. Crushed blacks matter too -- the mat is dark matte and the
    # brick sits on it.
    quality = 100.0
    quality -= abs(mean - TARGET_MEAN) * 0.6
    quality -= blown * 100 * 4.0
    quality -= max(0.0, blown * 100 - 5.0) ** 2  # cliff past 5% saturated
    quality -= crushed * 100 * 2.0
    quality += min(contrast, 12.0)  # tie-break only, capped
    return {
        "mean": mean,
        "blown": blown * 100,
        "crushed": crushed * 100,
        "sharp": sharp,
        "quality": quality,
    }


def patch_yaml(role: str, exposure: int, brightness: int) -> None:
    """Insert or update exposure/brightness under `cameras.<role>:`."""
    path = REPO / "config" / "scenario.yaml"
    lines = path.read_text().splitlines(keepends=True)
    out, in_role, done_exp, done_bri = [], False, False, False

    for line in lines:
        stripped = line.strip()
        if in_role and stripped and not line.startswith("    "):
            # leaving the role block: append anything we did not overwrite
            if not done_exp:
                out.append(f"    exposure: {exposure}            # [TUNED]\n")
            if not done_bri:
                out.append(f"    brightness: {brightness}            # [TUNED]\n")
            in_role = True if False else False
        if stripped == f"{role}:" and line.startswith("  ") and not line.startswith("   "):
            in_role, done_exp, done_bri = True, False, False
            out.append(line)
            continue
        if in_role and stripped.startswith("exposure:"):
            out.append(f"    exposure: {exposure}            # [TUNED]\n")
            done_exp = True
            continue
        if in_role and stripped.startswith("brightness:"):
            out.append(f"    brightness: {brightness}            # [TUNED]\n")
            done_bri = True
            continue
        out.append(line)

    if in_role:  # role block ran to end of file
        if not done_exp:
            out.append(f"    exposure: {exposure}            # [TUNED]\n")
        if not done_bri:
            out.append(f"    brightness: {brightness}            # [TUNED]\n")

    path.write_text("".join(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="wrist")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    scn = Scenario.load()
    kind = scn.require(f"cameras.{args.camera}.kind")
    if kind != "opencv":
        print(f"{args.camera} is a {kind} camera; this tunes v4l2/UVC cameras only.")
        print("RealSense auto-exposure is generally fine as shipped.")
        return 1

    dev = str(Path(str(scn.require(f"cameras.{args.camera}.index_or_path"))).resolve())
    print(f"tuning {args.camera} ({dev})\n")

    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print(f"cannot open {dev}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, scn.require(f"cameras.{args.camera}.width"))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, scn.require(f"cameras.{args.camera}.height"))

    v4l2_set(dev, "auto_exposure", 1)  # manual
    v4l2_set(dev, "gain", 0)

    header = f"{'exp':>5} {'bright':>7} {'mean':>7} {'blown%':>7} {'dark%':>7} {'sharp':>9} {'score':>9}"
    print(header)
    print("-" * len(header))

    results = []
    try:
        for brightness in BRIGHTNESS:
            v4l2_set(dev, "brightness", brightness)
            for exposure in EXPOSURES:
                v4l2_set(dev, "exposure_time_absolute", exposure)
                frame = grab(cap)
                if frame is None:
                    continue
                s = score_frame(frame)
                results.append((s["quality"], exposure, brightness, s))
                print(
                    f"{exposure:>5} {brightness:>7} {s['mean']:>7.1f} {s['blown']:>7.2f} "
                    f"{s['crushed']:>7.2f} {s['sharp']:>9.1f} {s['quality']:>9.1f}"
                )
    finally:
        cap.release()

    if not results:
        print("\nno frames captured")
        return 1

    results.sort(reverse=True, key=lambda r: r[0])
    _, exposure, brightness, s = results[0]

    print(f"\nbest: exposure={exposure} brightness={brightness}")
    print(f"  mean {s['mean']:.1f}  saturated {s['blown']:.2f}%  sharpness {s['sharp']:.1f}")

    v4l2_set(dev, "brightness", brightness)
    v4l2_set(dev, "exposure_time_absolute", exposure)
    print("  applied to the live camera")

    if args.no_save:
        print("\n(--no-save: config not written)")
        return 0

    patch_yaml(args.camera, exposure, brightness)
    print(f"\nwrote cameras.{args.camera}.exposure/brightness to config/scenario.yaml")
    print("Preflight re-applies these after any replug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
