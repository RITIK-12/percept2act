#!/usr/bin/env python
"""Run the closed loop. This is the deliverable.

Build up in three stages so a failure is always localized:

  1. detector only, no robot -- confirms DETECT and REASON
       python scripts/10_run_loop.py --dry-run --bricks 5

  2. one brick, robot live -- confirms ACT
       python scripts/10_run_loop.py --bricks 1

  3. full run
       python scripts/10_run_loop.py --bricks 14

Run in the `hack_lerobot` env. Keep a hand near the follower's power for the
first robot-live run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from percept2act.config import Scenario  # noqa: E402
from percept2act.orchestrator import Orchestrator  # noqa: E402


def build_robot(scn: Scenario):
    """Connect the follower with the configured policy-input cameras."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from percept2act.lerobot_args import camera_config

    cam_cfg = camera_config(scn, list(scn.require("cameras.policy_inputs")))

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras = {}
    for role, entry in cam_cfg.items():
        kind = entry.pop("type")
        if kind == "opencv":
            cameras[role] = OpenCVCameraConfig(**entry)
        else:
            cameras[role] = RealSenseCameraConfig(**entry)

    cfg = SO101FollowerConfig(
        port=str(scn.require("arms.follower.port")),
        id=str(scn.require("arms.follower.id")),
        cameras=cameras,
    )
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def build_sorter(scn: Scenario):
    from percept2act.policy_runner import PolicyRunner

    ckpt = scn.abs_path("policies.stage2_sort.export_dir")
    # lerobot-train writes checkpoints under <output_dir>/checkpoints/<step>/pretrained_model
    candidates = sorted(ckpt.glob("checkpoints/*/pretrained_model"))
    if not candidates:
        candidates = sorted(ckpt.glob("**/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(
            f"no trained policy under {ckpt}\n"
            "  train it first:  bash scripts/05_train.sh smolvla"
        )
    latest = candidates[-1]
    print(f"policy checkpoint: {latest}")
    return PolicyRunner(
        checkpoint=str(latest),
        policy_type=str(scn.require("policies.stage2_sort.policy")),
        device="xpu",
        fps=int(scn.require("policies.control_fps")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bricks", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="detector only, robot untouched")
    ap.add_argument("--device", help="override the detector device (NPU/GPU/CPU)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    scn = Scenario.load()
    robot = sorter = None

    if not args.dry_run:
        print("connecting to the follower...")
        robot = build_robot(scn)
        sorter = build_sorter(scn)
        print(f"  {sorter}")
    else:
        print("DRY RUN: detector and reasoning only, the arm will not move.")

    detector = None
    if args.device:
        from percept2act.detector import Detector

        detector = Detector(scn, device=args.device)

    try:
        with Orchestrator(scn, robot=robot, sorter=sorter, detector=detector) as orch:
            print(f"  {orch.detector}\n")
            report = orch.run(bricks=args.bricks)

            print("\n" + "=" * 70)
            print("RUN REPORT")
            print("=" * 70)
            print(report.summary())
            print("\n" + "=" * 70)
            print("LATENCY BY STAGE AND DEVICE  (say these numbers out loud)")
            print("=" * 70)
            print(orch.latency.table())
    finally:
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
