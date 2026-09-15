#!/usr/bin/env python
"""Run the closed loop. This is the deliverable.

The pipeline is identical regardless of what moves the arm; only the last hop
swaps. Build it up in this order so a failure is always localized:

  1. no arm at all -- proves PERCEIVE, DETECT, REASON and the latency logging
       python scripts/10_run_loop.py --executor stub --bricks 5

  2. arm live, replaying recorded demos -- proves ACT, no training needed
       python scripts/10_run_loop.py --executor replay --bricks 1
       python scripts/10_run_loop.py --executor replay --bricks 10 --no-pause

  3. once SmolVLA has trained, swap one flag. Nothing else changes.
       python scripts/10_run_loop.py --executor policy --bricks 10

Run in the `hack_lerobot` env. Keep a hand near the follower's power for the
first run with the arm live.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from percept2act.config import Scenario  # noqa: E402
from percept2act.executor import build_executor  # noqa: E402
from percept2act.orchestrator import Orchestrator  # noqa: E402


def build_robot(scn: Scenario):
    """Connect the follower with the configured policy-input cameras."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from percept2act.lerobot_args import camera_config

    cameras = {}
    for role, entry in camera_config(scn, list(scn.require("cameras.policy_inputs"))).items():
        kind = entry.pop("type")
        cameras[role] = (
            OpenCVCameraConfig(**entry) if kind == "opencv" else RealSenseCameraConfig(**entry)
        )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=str(scn.require("arms.follower.port")),
            id=str(scn.require("arms.follower.id")),
            cameras=cameras,
        )
    )
    robot.connect()
    return robot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--executor",
        default="stub",
        choices=["stub", "replay", "policy"],
        help="what carries out the placement (default: stub, which moves nothing)",
    )
    ap.add_argument("--bricks", type=int, default=1)
    ap.add_argument("--device", help="override the detector device (NPU/GPU/CPU)")
    ap.add_argument("--no-pause", action="store_true", help="do not wait between bricks")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    scn = Scenario.load()
    robot = None

    if args.executor != "stub":
        print("connecting to the follower...")
        robot = build_robot(scn)
    else:
        print("STUB executor: detection and reasoning only, the arm will not move.")

    detector = None
    if args.device:
        from percept2act.detector import Detector

        detector = Detector(scn, device=args.device)

    try:
        executor = build_executor(args.executor, scn, robot=robot)
        print(f"executor: {executor.name} on {executor.device}")

        with Orchestrator(
            scn,
            robot=robot,
            executor=executor,
            detector=detector,
            pause_between=not args.no_pause,
        ) as orch:
            print(f"detector: {orch.detector}\n")
            report = orch.run(bricks=args.bricks)

            print("\n" + "=" * 70)
            print("RUN REPORT")
            print("=" * 70)
            print(report.summary())
            print("\n" + "=" * 70)
            print("LATENCY BY STAGE AND DEVICE  (say these numbers out loud)")
            print("=" * 70)
            print(orch.latency.table())
            executor.close()
    finally:
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
