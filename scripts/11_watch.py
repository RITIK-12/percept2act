#!/usr/bin/env python
"""Live autonomous sorting. Put a brick down; the arm sorts it. No leader.

This is the demo. The camera watches the inspection spot; when a brick appears
and the scene settles, the loop scores it, picks the matching instruction and
places it in the corresponding plate, then goes back to watching.

    watch -> brick appears -> settle -> PatchCore on NPU -> verdict
          -> instruction -> arm places it -> watch again

Usage
-----
  python scripts/11_watch.py                    # replay executor, live window
  python scripts/11_watch.py --executor stub    # rehearse, arm never moves
  python scripts/11_watch.py --no-show          # headless
  python scripts/11_watch.py --bricks 5         # stop after 5

Ctrl-C to stop. Run in the `hack_lerobot` env.

Nothing here touches the leader arm -- put it down.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402

from percept2act.config import Scenario  # noqa: E402
from percept2act.executor import build_executor  # noqa: E402
from percept2act.orchestrator import Orchestrator  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))

WIN = "percept2act - live sorting"


def build_robot(scn: Scenario):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=str(scn.require("arms.follower.port")),
            id=str(scn.require("arms.follower.id")),
            cameras={},  # the orchestrator owns the wrist camera
        )
    )
    robot.connect()
    return robot


def overlay(frame, patch, frac, run, need, crop, status, verdict=None):
    x0, y0, x1, y1 = crop
    view = frame.copy()
    colour = (0, 220, 0) if run >= need else (0, 200, 255)
    cv2.rectangle(view, (x0, y0), (x1, y1), colour, 2)

    lines = [status, f"brick coverage {frac*100:5.1f}%   settle {run}/{need}"]
    if verdict is not None:
        lines.append(
            f"score {verdict.score:.4f} -> {verdict.verdict.upper()}"
        )
    for i, text in enumerate(lines):
        yy = 26 + i * 26
        cv2.putText(view, text, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 1, cv2.LINE_AA)
    return view


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--executor", default="replay", choices=["stub", "replay", "policy"])
    ap.add_argument("--bricks", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--no-show", action="store_true", help="no live window")
    ap.add_argument("--settle", type=int, default=12, help="frames the scene must hold steady")
    ap.add_argument(
        "--delay", type=float, default=2.0,
        help="seconds to hold the verdict on screen before the arm moves",
    )
    ap.add_argument("--device", help="override detector device (NPU/GPU/CPU)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    scn = Scenario.load()
    show = not args.no_show

    robot = None
    if args.executor != "stub":
        print("connecting to the follower...")
        robot = build_robot(scn)
    else:
        print("STUB executor: the arm will not move.")

    detector = None
    if args.device:
        from percept2act.detector import Detector

        detector = Detector(scn, device=args.device)

    sorted_count = 0
    try:
        executor = build_executor(args.executor, scn, robot=robot)
        with Orchestrator(scn, robot=robot, executor=executor, detector=detector) as orch:
            print(f"detector: {orch.detector}")
            print(f"executor: {executor.name} on {executor.device}\n")

            crop = scn.require("inspection_station.crop")
            state = {"status": "", "verdict": None}

            def draw(frame, patch, frac, run, need):
                if not show:
                    return
                cv2.imshow(WIN, overlay(frame, patch, frac, run, need, crop,
                                        state["status"], state["verdict"]))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt

            orch.goto_inspect_pose()

            while True:
                state["status"] = "WAITING - place a brick"
                state["verdict"] = None
                print("\nwaiting for a brick...")
                orch.wait_for_brick(present=True, stable_frames=args.settle, on_frame=draw)

                state["status"] = "INSPECTING"
                verdict, reinspections = orch.resolve_verdict()
                state["verdict"] = verdict

                brick_class = verdict.verdict
                note = ""
                if brick_class == "unsure":
                    brick_class = scn.require("classes.unsure.fallback_class")
                    note = f" (unsure after {reinspections} re-inspection)"

                plate = scn.plate_for_class(brick_class)
                instruction = scn.instruction_for_class(brick_class)
                print(
                    f"  score {verdict.score:.4f} -> {verdict.verdict} -> {plate} plate{note}\n"
                    f"  {instruction!r}"
                )

                # Hold the verdict on screen before moving. Gives you time to get
                # your hand clear, and makes the decision readable to anyone
                # watching instead of the arm just lurching.
                if args.delay > 0:
                    deadline = time.time() + args.delay
                    while True:
                        left = deadline - time.time()
                        if left <= 0:
                            break
                        state["status"] = (
                            f"{verdict.verdict.upper()} -> {plate} plate  |  moving in {left:0.1f}s"
                        )
                        if show:
                            frame = orch.camera.read()
                            if frame is not None:
                                draw(frame, None, 0.0, 0, args.settle)
                        else:
                            time.sleep(0.05)

                state["status"] = f"PLACING in {plate} plate"
                if show:
                    frame = orch.camera.read()
                    if frame is not None:
                        draw(frame, None, 0.0, 0, args.settle)
                steps = orch.place(brick_class)
                sorted_count += 1
                print(f"  done in {steps} steps  ({sorted_count} sorted)")

                state["status"] = "returning to inspect pose"
                state["verdict"] = None
                if show:
                    frame = orch.camera.read()
                    if frame is not None:
                        draw(frame, None, 0.0, 0, args.settle)
                orch.goto_inspect_pose()

                if args.bricks and sorted_count >= args.bricks:
                    break

                # Do not re-trigger on a brick that is still sitting there.
                state["status"] = "waiting for the spot to clear"
                orch.wait_for_brick(
                    present=False, stable_frames=args.settle, timeout=20, on_frame=draw
                )

            print("\n" + "=" * 68)
            print(f"{sorted_count} bricks sorted")
            print("=" * 68)
            print(orch.latency.table())
            executor.close()

    except KeyboardInterrupt:
        print(f"\nstopped. {sorted_count} bricks sorted.")
    finally:
        if show:
            cv2.destroyAllWindows()
        if robot is not None:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
