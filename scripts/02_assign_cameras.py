#!/usr/bin/env python
"""Assign the two RealSense cameras to their roles, and tune the inspect crop.

The physical question ("which D415 is on the boom and which is on the tripod?")
cannot be answered from serial numbers alone, so this script grabs one frame per
camera, writes them to disk for you to look at, then writes your decision back
into config/scenario.yaml.

Usage
-----
  # 1. capture previews from every camera it can find
  python scripts/02_assign_cameras.py --preview

  # 2. look at experiments/camera_preview/*.png, then commit the assignment
  python scripts/02_assign_cameras.py --assign overhead=<serial> inspect=<serial>

  # 3. once bricks are staged at the station, tune the inspection crop
  python scripts/02_assign_cameras.py --tune-crop 180,120,460,360

Run in the `hack_lerobot` env (it has pyrealsense2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from percept2act.config import Scenario  # noqa: E402

OUT = REPO / "experiments" / "camera_preview"


def realsense_serials() -> list[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]


def grab_realsense(serial: str, width: int, height: int, fps: int) -> np.ndarray | None:
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        pipe.start(cfg)
        # Discard the first frames; the D415 auto-exposure needs a moment.
        for _ in range(15):
            frames = pipe.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {serial}: {exc}")
        return None
    finally:
        try:
            pipe.stop()
        except Exception:  # noqa: BLE001
            pass


def grab_uvc(path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ! could not open {path}")
        return None
    try:
        for _ in range(10):
            ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def cmd_preview(scn: Scenario) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    serials = realsense_serials()
    print(f"RealSense devices found: {serials or 'NONE'}")
    for serial in serials:
        frame = grab_realsense(serial, 640, 480, 30)
        if frame is None:
            continue
        dest = OUT / f"realsense_{serial}.png"
        cv2.imwrite(str(dest), frame)
        print(f"  wrote {dest}")

    wrist_path = scn.get("cameras.wrist.index_or_path")
    if wrist_path:
        frame = grab_uvc(str(wrist_path))
        if frame is not None:
            dest = OUT / "wrist_uvc.png"
            cv2.imwrite(str(dest), frame)
            print(f"  wrote {dest}")

    print(
        "\nNow open the previews. The BOOM camera looks straight down at the mat\n"
        "and sees the plates and the brick pile. The TRIPOD camera looks forward\n"
        "across the mat at roughly brick height.\n\n"
        "Then run:\n"
        "  python scripts/02_assign_cameras.py --assign overhead=<serial> inspect=<serial>"
    )
    return 0


def _patch_yaml(edits: dict[str, str]) -> None:
    """Rewrite scenario.yaml in place, preserving comments.

    A line-oriented patch rather than a yaml round-trip, precisely so the
    explanatory comments in scenario.yaml survive. Each key is written under its
    parent `cameras.<role>:` block.
    """
    cfg_path = REPO / "config" / "scenario.yaml"
    lines = cfg_path.read_text().splitlines(keepends=True)
    role: str | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and not line.startswith(" " * 3):
            role = None
        if stripped.rstrip(":") in edits and line.startswith("  ") and stripped.endswith(":"):
            role = stripped.rstrip(":")
            continue
        if role and stripped.startswith("serial:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f'{indent}serial: "{edits[role]}"           # [ASSIGNED]\n'
            role = None
    cfg_path.write_text("".join(lines))


def cmd_assign(pairs: list[str]) -> int:
    # Roles come from the config rather than a hardcoded list, so renaming a
    # camera role in scenario.yaml does not silently break this command.
    scn = Scenario.load()
    valid = {
        role
        for role, entry in (scn.get("cameras") or {}).items()
        if isinstance(entry, dict) and entry.get("kind") == "realsense"
    }

    edits: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"! expected role=serial, got {pair!r}")
            return 2
        role, serial = pair.split("=", 1)
        if role not in valid:
            print(f"! role must be one of {sorted(valid)}, got {role!r}")
            return 2
        edits[role] = serial.strip()

    available = realsense_serials()
    for role, serial in edits.items():
        if available and serial not in available:
            print(f"! serial {serial} for {role} is not connected. Present: {available}")
            return 2

    _patch_yaml(edits)
    print(f"Updated config/scenario.yaml: {edits}")

    # Re-read from disk to prove the line-oriented patch actually landed, rather
    # than reporting success from the in-memory dict we just built.
    for role in edits:
        print(f"  cameras.{role}.serial -> {Scenario.load().require(f'cameras.{role}.serial')}")
    return 0


def cmd_tune_crop(scn: Scenario, crop: str) -> int:
    x0, y0, x1, y1 = (int(v) for v in crop.split(","))
    serial = scn.require("cameras.inspect.serial")
    frame = grab_realsense(serial, 640, 480, 30)
    if frame is None:
        print("! could not grab a frame from the inspect camera")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    marked = frame.copy()
    cv2.rectangle(marked, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(str(OUT / "inspect_crop_overlay.png"), marked)
    cv2.imwrite(str(OUT / "inspect_crop.png"), frame[y0:y1, x0:x1])
    print(
        f"wrote {OUT/'inspect_crop_overlay.png'} and {OUT/'inspect_crop.png'}\n"
        "The crop should tightly frame a presented brick with a little margin.\n"
        f"When it looks right, set inspection_station.crop: [{x0}, {y0}, {x1}, {y1}]"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true", help="grab one frame per camera")
    ap.add_argument("--assign", nargs="+", metavar="ROLE=SERIAL", help="commit an assignment")
    ap.add_argument("--tune-crop", metavar="X0,Y0,X1,Y1", help="preview the inspect crop")
    args = ap.parse_args()

    if args.assign:
        return cmd_assign(args.assign)

    scn = Scenario.load()
    if args.tune_crop:
        return cmd_tune_crop(scn, args.tune_crop)
    if args.preview:
        return cmd_preview(scn)

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
