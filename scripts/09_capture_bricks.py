#!/usr/bin/env python
"""Capture the brick image dataset Anomalib trains on.

PatchCore fits on NORMAL samples only -- no defective examples are needed for
training. Capture ~50 good bricks here, varying colour, size, position and
rotation, and PatchCore learns what "undamaged" looks like. Anything it cannot
reconstruct from that memory bank scores as anomalous.

Capture a handful of DEFECTIVE bricks too (`--class defective`). They are not
used for fitting -- only for choosing the threshold, which is the difference
between a detector that works and one that guesses.

Usage
-----
  # live window, SPACE to capture a good brick, reposition, repeat
  python scripts/09_capture_bricks.py --class good

  # 60 automatic captures, one per second -- slide the brick around while it runs
  python scripts/09_capture_bricks.py --class good --auto 60 --interval 1.0

  # a few damaged ones for threshold tuning
  python scripts/09_capture_bricks.py --class defective --auto 15

Keys:  SPACE = capture   c = re-show the crop box   q = done
Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402

from percept2act.cameras import open_stream  # noqa: E402
from percept2act.config import Scenario  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True, choices=["good", "defective"])
    ap.add_argument("--auto", type=int, default=0, help="capture N frames automatically")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto captures")
    ap.add_argument("--headless", action="store_true", help="no window (implies --auto)")
    args = ap.parse_args()

    scn = Scenario.load()
    role = scn.require("cameras.detector_input")
    crop = scn.require("inspection_station.crop")
    x0, y0, x1, y1 = crop

    key = "detector.normals_dir" if args.cls == "good" else "detector.defects_dir"
    dest = REPO / str(scn.require(key))
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.glob("*.png")))

    print(f"capturing '{args.cls}' crops from camera role '{role}'")
    print(f"  crop   : {crop}")
    print(f"  dest   : {dest}  ({existing} already there)")
    if args.cls == "good":
        print("  Vary colour, size, position and rotation. Aim for ~50 total.")
    else:
        print("  A handful is enough -- these only tune the threshold.")
    print()

    stream = open_stream(scn, role)
    saved = 0
    target = args.auto
    headless = args.headless or (args.auto and args.headless)
    last = 0.0
    win = f"percept2act - capture {args.cls}"

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                print(f"! crop {crop} is empty for a {frame.shape[1]}x{frame.shape[0]} frame")
                return 2

            take = False
            if target:
                now = time.time()
                if now - last >= args.interval:
                    take, last = True, now
            if not headless:
                view = frame.copy()
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
                label = f"{args.cls}: {existing + saved} saved" + (
                    f" / target {existing + target}" if target else "   SPACE=capture  q=done"
                )
                for col, th in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                    cv2.putText(view, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)
                cv2.imshow(win, view)
                cv2.imshow("crop (what Anomalib sees)", patch)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    take = True

            if take:
                out = dest / f"{args.cls}_{int(time.time()*1000)}.png"
                cv2.imwrite(str(out), patch)
                saved += 1
                print(f"  [{existing + saved}] {out.name}")
                if target and saved >= target:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        if not headless:
            cv2.destroyAllWindows()

    total = existing + saved
    print(f"\n{saved} new, {total} total in {dest}")
    if args.cls == "good" and total < 30:
        print("! fewer than 30 normals -- PatchCore will be unreliable. Capture more.")
    elif args.cls == "good":
        print("Next:  python scripts/10_train_detector.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
