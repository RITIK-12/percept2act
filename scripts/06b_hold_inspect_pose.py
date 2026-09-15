#!/usr/bin/env python
"""Park the follower at its inspect pose and hold it there.

The wrist camera moves with the arm, so the detector only produces comparable
scores if every inspection happens from the same pose. This drives the follower
to the first frame of a recorded episode -- which IS the pose the camera was in
when that episode started -- and holds it with torque on until you stop it.

Use it in a second terminal while capturing normals, so the camera does not
drift between images.

Usage
-----
  python scripts/06b_hold_inspect_pose.py             # pose from episode 0
  python scripts/06b_hold_inspect_pose.py --episode 2
  python scripts/06b_hold_inspect_pose.py --show      # print the joint targets

Ctrl-C releases the arm. Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from percept2act.config import Scenario  # noqa: E402


def first_frame_action(scn: Scenario, episode: int) -> np.ndarray:
    import pandas as pd

    root = REPO / str(scn.require("replay.dataset_root"))
    eps = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/**/*.parquet"))]
    )
    row = eps[eps["episode_index"] == episode]
    if row.empty:
        raise SystemExit(f"episode {episode} not found; have {sorted(eps['episode_index'])}")
    start = int(row.iloc[0]["dataset_from_index"])

    data = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("data/**/*.parquet"))])
    data = data.sort_values("index").reset_index(drop=True)
    return np.asarray(data.iloc[start]["action"]).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument(
        "--from-config",
        action="store_true",
        help="use inspection_station.pose (set with 06c) instead of an episode's first frame",
    )
    ap.add_argument("--show", action="store_true", help="print the pose and exit")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to reach the pose")
    args = ap.parse_args()

    scn = Scenario.load()
    if args.from_config:
        pose_cfg = scn.get("inspection_station.pose")
        if pose_cfg is None:
            raise SystemExit(
                "inspection_station.pose is not set.\n"
                "  Set it by teleoperating to a good pose:\n"
                "    python scripts/06c_set_inspect_pose.py"
            )
        target = np.asarray(pose_cfg, dtype=float)
    else:
        target = first_frame_action(scn, args.episode)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=str(scn.require("arms.follower.port")),
            id=str(scn.require("arms.follower.id")),
            cameras={},  # the capture script opens the camera; do not fight over it
        )
    )

    names = None
    robot.connect()
    try:
        names = list(robot.action_features)
        pose = {n: float(v) for n, v in zip(names, target)}
        src = "config" if args.from_config else f"episode {args.episode} frame 0"
        print(f"inspect pose ({src}):")
        for n, v in pose.items():
            print(f"  {n:<16} {v:8.2f}")
        if args.show:
            return 0

        # Ramp from the current position rather than snapping: a step command to
        # a distant pose is exactly what trips shoulder_lift's overload latch.
        obs = robot.get_observation()
        current = np.array(
            [float(obs.get(n, pose[n])) for n in names], dtype=float
        )
        steps = max(1, int(args.settle * 30))
        print(f"\nramping over {args.settle:.1f}s...")
        for i in range(1, steps + 1):
            blend = current + (target - current) * (i / steps)
            robot.send_action({n: float(v) for n, v in zip(names, blend)})
            time.sleep(1.0 / 30)

        print("\nHOLDING. The wrist camera is now at the inspect pose.")
        print("Capture in another terminal:")
        print("  python scripts/06_capture_normals.py --class good --auto 50 --interval 1.0")
        print("\nCtrl-C here to release the arm.")
        while True:
            robot.send_action(pose)
            time.sleep(1.0 / 30)

    except KeyboardInterrupt:
        print("\nreleasing")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
