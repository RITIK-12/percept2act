#!/usr/bin/env python
"""Live focus + aim assistant.

Opens a window per camera with a sharpness readout so the wrist lens can be
focused and the boom camera aimed without guessing. Sharpness is the variance of
the Laplacian: twist the lens until the number PEAKS, then stop.

Usage
-----
  # focus the wrist lens -- watch SHARPNESS, turn the barrel until it maxes out
  python scripts/01b_camera_focus.py --camera wrist

  # aim the boom straight down at the mat
  python scripts/01b_camera_focus.py --camera overhead

  # frame the inspection station, with the configured crop drawn on
  python scripts/01b_camera_focus.py --camera inspect --show-crop

  # no display available? write a frame + score to disk every second instead
  python scripts/01b_camera_focus.py --camera wrist --headless

Keys:  q = quit   s = save a snapshot
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
import numpy as np  # noqa: E402

from percept2act.config import Scenario  # noqa: E402

OUT = REPO / "experiments" / "camera_preview"


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper. Peak it by twisting the lens."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class Stream:
    """Uniform frame source over UVC and RealSense."""

    def __init__(self, scn: Scenario, role: str):
        self.role = role
        self.kind = scn.require(f"cameras.{role}.kind")
        self.w = scn.require(f"cameras.{role}.width")
        self.h = scn.require(f"cameras.{role}.height")
        self.fps = scn.require(f"cameras.{role}.fps")
        self._cap = None
        self._pipe = None

        if self.kind == "opencv":
            path = str(scn.require(f"cameras.{role}.index_or_path"))
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise SystemExit(f"cannot open {path}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        else:
            import pyrealsense2 as rs

            serial = str(scn.require(f"cameras.{role}.serial"))
            self._pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
            self._pipe.start(cfg)

    def read(self) -> np.ndarray | None:
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        frames = self._pipe.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass


def annotate(frame: np.ndarray, role: str, sharp: float, best: float, crop=None) -> np.ndarray:
    out = frame.copy()
    if crop:
        x0, y0, x1, y1 = crop
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, "inspect crop", (x0, max(18, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    blown = (out > 250).mean() * 100
    bar = int(min(sharp / max(best, 1e-6), 1.0) * 200)
    lines = [
        f"{role}  SHARPNESS {sharp:8.1f}   (best {best:8.1f})",
        f"brightness {frame.mean():5.1f}   saturated {blown:4.1f}%",
        "twist lens until SHARPNESS peaks   q=quit  s=save",
    ]
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (10, 88), (10 + bar, 100), (0, 255, 255), -1)
    cv2.rectangle(out, (10, 88), (210, 100), (255, 255, 255), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", required=True, choices=["wrist", "overhead", "inspect"])
    ap.add_argument("--show-crop", action="store_true", help="draw inspection_station.crop")
    ap.add_argument("--headless", action="store_true", help="no window; write to disk instead")
    ap.add_argument("--seconds", type=float, default=0, help="auto-quit after N seconds")
    args = ap.parse_args()

    scn = Scenario.load()
    crop = scn.get("inspection_station.crop") if args.show_crop else None
    stream = Stream(scn, args.camera)
    OUT.mkdir(parents=True, exist_ok=True)
    live = OUT / f"live_{args.camera}.png"

    best = 0.0
    t0 = time.time()
    last_print = 0.0
    win = f"percept2act · {args.camera}"
    print(f"streaming {args.camera}; sharpness peaks when focus is correct")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue
            sharp = sharpness(frame)
            best = max(best, sharp)
            view = annotate(frame, args.camera, sharp, best, crop)

            if args.headless:
                now = time.time()
                if now - last_print > 1.0:
                    last_print = now
                    cv2.imwrite(str(live), view)
                    print(f"  sharpness {sharp:9.1f}  best {best:9.1f}  -> {live}")
            else:
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    dest = OUT / f"{args.camera}_focus_{int(time.time())}.png"
                    cv2.imwrite(str(dest), frame)
                    print(f"  saved {dest}")

            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.imwrite(str(live), view if "view" in dir() else frame)
        stream.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\nbest sharpness seen: {best:.1f}")
    print(f"last frame: {live}")
    print(
        "\nRule of thumb: a focused 640x480 wrist view of the mat scores >150.\n"
        "Under ~60 is visibly soft and will hurt policy learning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
