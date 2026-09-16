#!/usr/bin/env python
"""Capture a per-brick image set at the inspect pose, in one terminal.

Why this exists alongside 09_capture_bricks.py
----------------------------------------------
The first dataset was ~47 crops of a SINGLE brick, and the memory bank ended up
encoding that brick rather than "undamaged". A different good brick then scored
as defective. Worse, the reported numbers hid it: nothing was held out, so every
normal was scored against a bank containing its own patches and read 0.0000.

So this script keeps each brick in its OWN folder:

    datasets/v2/good/red-1/*.png
    datasets/v2/good/blue-2/*.png
    datasets/v2/defective/red-1/*.png
    datasets/v2/empty/mat/*.png

Fit on some folders, hold one out entirely, and score the held-out brick. That
score is the only real measure of whether the detector generalizes.

It also parks the arm itself, so you no longer need 08_hold_pose.py running in a
second terminal.

Usage
-----
  python scripts/18_capture_set.py --class good      --brick red-1
  python scripts/18_capture_set.py --class defective --brick red-1
  python scripts/18_capture_set.py --class empty                    # bare mat
  python scripts/18_capture_set.py --class good --brick red-1 --auto 10

Keys:  SPACE = capture   q = done

ROTATE AND SHIFT THE BRICK BETWEEN SHOTS. Ten captures of a brick sitting
perfectly still is one image with sensor noise, and buys no diversity at all.

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

from percept2act.cameras import open_stream  # noqa: E402
from percept2act.config import Scenario  # noqa: E402


from percept2act.cameras import brick_fraction  # noqa: E402


def park_arm(scn: Scenario, settle: float):
    """Ramp the follower to the inspect pose and leave torque holding it.

    Returns (robot, pose_dict) or (None, None) if no pose is configured. The
    ramp is deliberate: a step command to a distant target pulls full current at
    once, which is what tripped shoulder_lift's overload latch during recording.
    """
    pose_cfg = scn.get("inspection_station.pose")
    if pose_cfg is None:
        raise SystemExit(
            "inspection_station.pose is not set.\n"
            "  set it first:  python scripts/05_set_inspect_pose.py"
        )

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=str(scn.require("arms.follower.port")),
            id=str(scn.require("arms.follower.id")),
            cameras={},  # this script opens the camera itself; do not fight over it
        )
    )
    robot.connect()

    target = np.asarray(pose_cfg, dtype=float)
    names = list(robot.action_features)
    pose = {n: float(v) for n, v in zip(names, target)}

    obs = robot.get_observation()
    current = np.array([float(obs.get(n, pose[n])) for n in names], dtype=float)
    steps = max(1, int(settle * 30))
    print(f"ramping to the inspect pose over {settle:.1f}s...")
    for i in range(1, steps + 1):
        blend = current + (target - current) * (i / steps)
        robot.send_action({n: float(v) for n, v in zip(names, blend)})
        time.sleep(1.0 / 30)
    print("arm parked and holding.\n")
    return robot, pose


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True,
                    choices=["good", "defective", "empty"])
    ap.add_argument("--brick", default=None,
                    help="label for THIS brick, e.g. red-1. Required except for --class empty.")
    ap.add_argument("--count", type=int, default=10, help="target captures (default 10)")
    ap.add_argument("--root", default="datasets/v2", help="dataset root (default datasets/v2)")
    ap.add_argument("--auto", action="store_true",
                    help="capture automatically on an interval instead of on SPACE")
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between auto captures")
    ap.add_argument("--no-arm", action="store_true", help="do not move the arm")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to reach the pose")
    args = ap.parse_args()

    if args.cls == "empty":
        args.brick = args.brick or "mat"
    elif not args.brick:
        ap.error("--brick is required (label each brick separately, e.g. --brick red-1)")

    scn = Scenario.load()
    role = scn.require("cameras.detector_input")
    x0, y0, x1, y1 = scn.require("inspection_station.crop")

    dest = REPO / args.root / args.cls / args.brick
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.glob("*.png")))

    print(f"class  : {args.cls}")
    print(f"brick  : {args.brick}")
    print(f"dest   : {dest}  ({existing} already there)")
    print(f"target : {args.count}")
    if args.cls == "empty":
        print("\n  Bare mat, no brick. These validate the occupancy gate")
        print("  (watch.brick_fraction) -- they are NOT a third Anomalib class.")
    else:
        print("\n  ROTATE AND SHIFT THE BRICK BETWEEN SHOTS.")
    print()

    robot = pose = None
    stream = None
    saved = 0
    last_capture = 0.0
    last_hold = 0.0
    win = f"percept2act - capture {args.cls}/{args.brick}"

    try:
        if not args.no_arm:
            robot, pose = park_arm(scn, args.settle)

        stream = open_stream(scn, role)

        while saved < args.count:
            frame = stream.read()
            if frame is None:
                continue
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                print(f"! crop [{x0},{y0},{x1},{y1}] is empty for a "
                      f"{frame.shape[1]}x{frame.shape[0]} frame")
                return 2

            # Nudge the servos back onto the pose periodically. Torque holds them
            # anyway; this catches slow drift over a long capture session.
            now = time.time()
            if robot is not None and now - last_hold > 0.5:
                robot.send_action(pose)
                last_hold = now

            frac = brick_fraction(patch)
            take = False
            if args.auto and now - last_capture >= args.interval:
                take, last_capture = True, now

            view = frame.copy()
            occupied = frac >= float(scn.get("watch.brick_fraction", 0.025))
            colour = (0, 255, 0) if occupied else (0, 165, 255)
            cv2.rectangle(view, (x0, y0), (x1, y1), colour, 2)
            label = f"{args.cls}/{args.brick}  {saved}/{args.count}   brick={frac:.3f}"
            hint = "auto" if args.auto else "SPACE=capture  q=done"
            for col, th in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                cv2.putText(view, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)
                cv2.putText(view, hint, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, th, cv2.LINE_AA)
            cv2.imshow(win, view)
            cv2.imshow("crop (what Anomalib sees)", patch)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                take = True

            if take:
                out = dest / f"{args.brick}_{int(time.time()*1000)}.png"
                cv2.imwrite(str(out), patch)
                saved += 1
                flag = "" if occupied or args.cls == "empty" else "   <- crop looks EMPTY"
                print(f"  [{saved}/{args.count}] {out.name}  brick={frac:.3f}{flag}")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if stream is not None:
            stream.close()
        cv2.destroyAllWindows()
        if robot is not None:
            print("releasing arm")
            robot.disconnect()

    total = existing + saved
    print(f"\n{saved} new, {total} total in {dest}")
    print("\nNext:")
    print(f"  another brick :  python scripts/18_capture_set.py --class {args.cls} --brick <label>")
    print(f"  see the set   :  find {args.root} -name '*.png' | cut -d/ -f2-4 | sort | uniq -c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
