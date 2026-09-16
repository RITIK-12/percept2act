"""Build LeRobot CLI arguments from config/scenario.yaml.

The record / train / replay scripts all need the same camera and arm arguments.
Generating them from one place means a camera reassignment is a config edit, not
a hunt through shell scripts.

Note the RealSense discriminator is `intelrealsense`, not `realsense` --
verified against lerobot 0.6.1 `CameraConfig.register_subclass`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from percept2act.config import Scenario

_KIND_TO_TYPE = {"opencv": "opencv", "realsense": "intelrealsense"}


def camera_config(scn: Scenario, roles: list[str]) -> dict[str, dict]:
    """Build the dict passed to --robot.cameras for the given roles."""
    out: dict[str, dict] = {}
    for role in roles:
        kind = scn.require(f"cameras.{role}.kind")
        entry: dict[str, object] = {
            "type": _KIND_TO_TYPE[kind],
            "width": scn.require(f"cameras.{role}.width"),
            "height": scn.require(f"cameras.{role}.height"),
            "fps": scn.require(f"cameras.{role}.fps"),
        }
        if kind == "opencv":
            entry["index_or_path"] = str(scn.require(f"cameras.{role}.index_or_path"))
        else:
            entry["serial_number_or_name"] = str(scn.require(f"cameras.{role}.serial"))
            entry["use_depth"] = bool(scn.get(f"cameras.{role}.use_depth", False))
        out[role] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit LeRobot CLI args from scenario.yaml")
    ap.add_argument(
        "what",
        choices=["cameras", "follower-port", "follower-id", "leader-port", "leader-id",
                 "follower-type", "leader-type", "instruction"],
    )
    ap.add_argument("--roles", help="comma-separated camera roles (default: policy_inputs)")
    ap.add_argument("--key", help="instruction key for `instruction`")
    args = ap.parse_args()

    scn = Scenario.load()

    if args.what == "cameras":
        roles = (
            args.roles.split(",")
            if args.roles
            else list(scn.require("cameras.policy_inputs"))
        )
        print(json.dumps(camera_config(scn, roles)))
    elif args.what == "instruction":
        print(scn.instruction(args.key))
    else:
        arm, field = args.what.split("-")
        print(scn.require(f"arms.{arm}.{'port' if field == 'port' else field}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def camera_objects(scn: Scenario, roles: list[str]) -> dict:
    """Real camera config objects, for the Python API rather than the CLI.

    camera_config() above builds plain dicts because lerobot-record takes them
    as a JSON string on the command line. Constructing a robot in-process needs
    the actual dataclasses instead.
    """
    from pathlib import Path

    out: dict = {}
    for role in roles:
        kind = scn.require(f"cameras.{role}.kind")
        common = {
            "width": scn.require(f"cameras.{role}.width"),
            "height": scn.require(f"cameras.{role}.height"),
            "fps": scn.require(f"cameras.{role}.fps"),
        }
        if kind == "opencv":
            from lerobot.cameras.opencv import OpenCVCameraConfig

            # index_or_path is `int | Path`, never a str.
            out[role] = OpenCVCameraConfig(
                index_or_path=Path(str(scn.require(f"cameras.{role}.index_or_path"))),
                **common,
            )
        else:
            from lerobot.cameras.realsense import RealSenseCameraConfig

            out[role] = RealSenseCameraConfig(
                serial_number_or_name=str(scn.require(f"cameras.{role}.serial")),
                **common,
            )
    return out
