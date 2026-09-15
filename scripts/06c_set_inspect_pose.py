#!/usr/bin/env python
"""Teleoperate the follower to a good inspect pose, then save it.

Drive the follower with the leader while watching the wrist camera live, with
the Anomalib crop drawn on. When the brick sits nicely inside the green box,
press ENTER: the follower's current joint positions are written to
`inspection_station.pose` in config/scenario.yaml.

That pose is what the loop returns to before every inspection, so the detector
always sees a brick from the same angle and distance. Getting it right here is
worth more than any amount of threshold tuning later.

You can also nudge the crop from inside this tool, so the pose and the crop are
chosen together against the same live view.

Keys
----
  ENTER  save the current pose (and crop) and exit
  w/a/s/d   move the crop box up / left / down / right
  +  /  -   grow / shrink the crop box
  q      quit without saving

Usage
-----
  python scripts/06c_set_inspect_pose.py

Put ONE brick where it will sit during the demo before you start.
Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from percept2act.cameras import open_stream  # noqa: E402
from percept2act.config import Scenario  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def save(pose: list[float], crop: list[int]) -> None:
    """Rewrite inspection_station.pose and .crop, preserving the comments."""
    path = REPO / "config" / "scenario.yaml"
    lines = path.read_text().splitlines(keepends=True)
    pose_s = "[" + ", ".join(f"{v:.2f}" for v in pose) + "]"
    crop_s = "[" + ", ".join(str(int(v)) for v in crop) + "]"
    out, in_block = [], False
    for line in lines:
        stripped = line.strip()
        if stripped == "inspection_station:":
            in_block = True
            out.append(line)
            continue
        if in_block and stripped and not line.startswith(" "):
            in_block = False
        if in_block and stripped.startswith("pose:"):
            out.append(f"  pose: {pose_s}   # [TELEOP-SET]\n")
            continue
        if in_block and stripped.startswith("crop:"):
            out.append(f"  crop: {crop_s}      # [TELEOP-SET] [x0, y0, x1, y1]\n")
            continue
        out.append(line)
    path.write_text("".join(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-save", action="store_true", help="preview only")
    args = ap.parse_args()

    scn = Scenario.load()
    crop = list(scn.require("inspection_station.crop"))

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=str(scn.require("arms.follower.port")),
            id=str(scn.require("arms.follower.id")),
            cameras={},  # we open the wrist camera ourselves below
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(
            port=str(scn.require("arms.leader.port")),
            id=str(scn.require("arms.leader.id")),
        )
    )

    print("connecting...")
    robot.connect()
    leader.connect()
    stream = open_stream(scn, scn.require("cameras.detector_input"))
    win = "percept2act - set inspect pose  [ENTER=save  wasd=move crop  +/-=size  q=quit]"

    print(
        "\nDrive the follower with the leader.\n"
        "Get ONE brick centred in the green box, filling a good part of it.\n"
        "Then press ENTER in the camera window.\n"
    )

    saved = False
    try:
        while True:
            action = leader.get_action()
            robot.send_action(action)

            frame = stream.read()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            x0, y0, x1, y1 = (int(v) for v in crop)
            x0, y0 = max(0, min(x0, w - 10)), max(0, min(y0, h - 10))
            x1, y1 = max(x0 + 10, min(x1, w)), max(y0 + 10, min(y1, h))

            view = frame.copy()
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            patch = frame[y0:y1, x0:x1]
            lines = [
                f"crop [{x0},{y0},{x1},{y1}]  {x1-x0}x{y1-y0}",
                f"patch mean {patch.mean():.0f}   ENTER=save  q=quit",
            ]
            for i, text in enumerate(lines):
                yy = 24 + i * 24
                cv2.putText(view, text, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, text, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, view)
            if patch.size:
                cv2.imshow("crop (what Anomalib sees)", patch)

            key = cv2.waitKey(1) & 0xFF
            step = 10
            if key in (13, 10):  # Enter
                obs = robot.get_observation()
                names = list(robot.action_features)
                pose = [float(obs.get(n, 0.0)) for n in names]
                print("\ncaptured inspect pose:")
                for n, v in zip(names, pose):
                    print(f"  {n:<16} {v:8.2f}")
                print(f"  crop {[x0, y0, x1, y1]}")
                if not args.no_save:
                    save(pose, [x0, y0, x1, y1])
                    print("\nwritten to config/scenario.yaml")
                else:
                    print("\n(--no-save: nothing written)")
                saved = True
                break
            if key == ord("q"):
                print("\nquit without saving")
                break
            if key == ord("a"):
                crop = [x0 - step, y0, x1 - step, y1]
            elif key == ord("d"):
                crop = [x0 + step, y0, x1 + step, y1]
            elif key == ord("w"):
                crop = [x0, y0 - step, x1, y1 - step]
            elif key == ord("s"):
                crop = [x0, y0 + step, x1, y1 + step]
            elif key in (ord("+"), ord("=")):
                crop = [x0 - step, y0 - step, x1 + step, y1 + step]
            elif key in (ord("-"), ord("_")):
                crop = [x0 + step, y0 + step, x1 - step, y1 - step]
            else:
                crop = [x0, y0, x1, y1]

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stream.close()
        cv2.destroyAllWindows()
        robot.disconnect()
        leader.disconnect()

    if saved and not args.no_save:
        print(
            "\nNext: hold this pose and capture normals.\n"
            "  terminal 1:  python scripts/06b_hold_inspect_pose.py --from-config\n"
            "  terminal 2:  python scripts/06_capture_normals.py --class good --auto 50"
        )
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
