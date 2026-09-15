#!/usr/bin/env python
"""Scan an arm's motor bus and report which IDs answer.

Run this when lerobot reports "Missing motor IDs". The SO-101's Feetech servos
are daisy-chained, so which IDs are missing tells you what kind of fault it is:

  * a CONTIGUOUS TAIL missing (e.g. 4,5,6) -> a cable came loose between the
    last good motor and the first missing one;
  * a SINGLE motor missing mid-chain -> that servo has latched its overload or
    over-temperature protection, or has failed. The chain still passes signal
    through it, which is why the ones after it still answer.

A latched servo is cleared by power-cycling the arm: unplug the POWER barrel
jack (not the USB), wait 5 seconds, plug it back in.

Usage
-----
  python scripts/00b_scan_motors.py                # follower
  python scripts/00b_scan_motors.py --arm leader

Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from percept2act.config import Scenario  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="follower", choices=["follower", "leader"])
    args = ap.parse_args()

    scn = Scenario.load()
    port = str(scn.require(f"arms.{args.arm}.port"))
    print(f"scanning {args.arm} on {port}\n")

    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    motors = {
        name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
        for i, name in enumerate(JOINTS, start=1)
    }
    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        # handshake=False is the point of this script: the normal connect()
        # asserts every expected motor is present, which is the very failure we
        # are here to diagnose. Open the port, then ping the bus directly.
        bus.connect(handshake=False)
        found = bus.broadcast_ping() or {}
    except Exception as exc:  # noqa: BLE001
        print(f"bus scan failed: {exc}")
        print("\nIs the arm powered? Is another process holding the port?")
        print("  pgrep -fa 'lerobot|record'")
        return 1
    finally:
        try:
            bus.disconnect()
        except Exception:  # noqa: BLE001
            pass

    print(f"{'id':>3}  {'joint':<15} {'status'}")
    print("-" * 36)
    missing = []
    for i, name in enumerate(JOINTS, start=1):
        if i in found:
            print(f"{i:>3}  {name:<15} OK  (model {found[i]})")
        else:
            print(f"{i:>3}  {name:<15} *** NOT RESPONDING ***")
            missing.append(i)

    if not missing:
        print(f"\nAll {len(JOINTS)} motors answering. Ready to record.")
        return 0

    print(f"\n{len(missing)} motor(s) missing: {missing}")
    tail = list(range(min(missing), len(JOINTS) + 1))
    if missing == tail:
        print(
            "\nThat is a contiguous tail, so it is almost certainly a CABLE.\n"
            f"Reseat the connector between motor {min(missing) - 1} and motor {min(missing)}, "
            "and check the next one along."
        )
    else:
        print(
            "\nSingle motor(s) missing mid-chain, so the wiring is passing signal.\n"
            "That servo has latched its overload / over-temperature protection.\n\n"
            "  1. Unplug the arm's POWER barrel jack (not the USB)\n"
            "  2. Wait 5 seconds\n"
            "  3. Plug it back in, then re-run this scan\n\n"
            "If it stays missing after a power cycle, reseat the cables either\n"
            "side of it. If it is warm to the touch, let it cool first."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
