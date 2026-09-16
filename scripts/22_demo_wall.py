#!/usr/bin/env python
"""The live demo, with all three cameras on screen.

Identical pipeline to 12_demo.py -- ONLY the wrist camera feeds the detector and
the policy. The two RealSense views are display-only, so a judge can see the
workspace and the arm while the verdict is being made.

They are read on a background thread and never touched by the control loop, so
adding them cannot slow the 30 Hz policy loop or change a single inference.

    PERCEIVE   wrist camera at a fixed inspect pose
    DETECT     Anomalib PatchCore on OpenVINO / NPU, median of N frames
    REASON     score -> verdict -> instruction string
    ACT        SmolVLA, conditioned on that string, places the brick

Usage:
  python scripts/22_demo_wall.py --executor policy
  python scripts/22_demo_wall.py --executor stub      # perception only
  python scripts/22_demo_wall.py --bricks 5

q or Ctrl-C to stop. Run in the `hack_lerobot` env.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from percept2act.cameras import open_stream  # noqa: E402
from percept2act.config import Scenario  # noqa: E402
from percept2act.executor import build_executor  # noqa: E402
from percept2act.orchestrator import Orchestrator  # noqa: E402

WIN = "percept2act - Perceive / Detect / Reason / Act"
MAIN_W, MAIN_H = 640, 480
SIDE_W, SIDE_H = 320, 240
HEADER = 64

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
GREEN = (80, 220, 80)
CORAL = (90, 110, 240)
AMBER = (0, 190, 255)


def label(img, text, org, scale=0.55, colour=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def panel(frame, w, h, title, live=True):
    """Fit a frame into a titled panel, or a placeholder if the camera is down."""
    if frame is None:
        out = np.full((h, w, 3), 30, np.uint8)
        label(out, "no signal", (w // 2 - 45, h // 2), 0.6, GREY)
    else:
        out = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    label(out, title, (8, 20), 0.5, WHITE if live else GREY)
    return out


class SideCameras:
    """Display-only RealSense feeds, polled off the control loop.

    A dropped or missing camera degrades to a 'no signal' panel rather than
    taking the demo down -- these are decoration, not part of the pipeline.
    """

    def __init__(self, scn: Scenario, roles: list[str], fps: float = 15.0):
        self.frames: dict[str, np.ndarray | None] = {r: None for r in roles}
        self.streams: dict[str, object] = {}
        for role in roles:
            try:
                self.streams[role] = open_stream(scn, role)
                print(f"  side camera '{role}': ok")
            except Exception as exc:  # noqa: BLE001
                print(f"  side camera '{role}': unavailable ({type(exc).__name__})")
        self._stop = threading.Event()
        self._period = 1.0 / fps
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            for role, stream in self.streams.items():
                try:
                    f = stream.read()
                    if f is not None:
                        self.frames[role] = f
                except Exception:  # noqa: BLE001, S110 - decoration must never kill the demo
                    pass
            time.sleep(self._period)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        for stream in self.streams.values():
            try:
                stream.close()
            except Exception:  # noqa: BLE001, S110
                pass


def compose(wrist, crop, state, sides, side_roles, stats):
    """Build the three-camera wall with the status header."""
    main = panel(wrist, MAIN_W, MAIN_H, "WRIST  - detector + policy input")
    v = state.get("verdict")
    if wrist is not None and crop:
        # ONE box: the region Anomalib scores. Grey until classified, then
        # tinted by verdict with the class named on the box itself.
        x0, y0, x1, y1 = crop
        sx, sy = MAIN_W / wrist.shape[1], MAIN_H / wrist.shape[0]
        a = (int(x0 * sx), int(y0 * sy))
        b = (int(x1 * sx), int(y1 * sy))
        colour = GREY if v is None else (CORAL if v.verdict == "defective" else GREEN)
        cv2.rectangle(main, a, b, colour, 2 if v is None else 3)
        if v is not None:
            cv2.rectangle(main, (a[0], a[1] - 24), (a[0] + 190, a[1]), colour, -1)
            cv2.putText(main, f"{v.verdict.upper()}  {v.score:.3f}", (a[0] + 6, a[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    side = np.vstack([
        panel(sides.frames.get(r), SIDE_W, SIDE_H, r.upper() + "  - view only",
              live=r in sides.streams)
        for r in side_roles
    ])
    if side.shape[0] < MAIN_H:
        side = np.vstack([side, np.full((MAIN_H - side.shape[0], SIDE_W, 3), 30, np.uint8)])

    body = np.hstack([main, side[:MAIN_H]])
    head = np.full((HEADER, body.shape[1], 3), 22, np.uint8)

    v = state.get("verdict")
    label(head, state.get("status", ""), (12, 25), 0.62,
          GREY if v is None else (CORAL if v.verdict == "defective" else GREEN))
    if v is not None:
        label(head, f"score {v.score:.4f}   thr {v.threshold:.4f}   {v.device}  {v.latency_ms:.0f}ms",
              (12, 50), 0.48, WHITE)
    else:
        label(head, stats, (12, 50), 0.48, GREY)

    return np.vstack([head, body])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--executor", default="policy", choices=["stub", "replay", "policy"])
    ap.add_argument("--bricks", type=int, default=0, help="stop after N (0 = until q)")
    ap.add_argument("--delay", type=float, default=2.0, help="hold the verdict before moving")
    ap.add_argument("--settle", type=int, default=12, help="steady frames before acting")
    ap.add_argument("--device", default=None, help="override the detector device")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    scn = Scenario.load()

    robot = None
    if args.executor != "stub":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        print("connecting to the follower...")
        robot = SO101Follower(SO101FollowerConfig(
            port=str(scn.require("arms.follower.port")),
            id=str(scn.require("arms.follower.id")),
            cameras={},  # the orchestrator owns the wrist camera
        ))
        robot.connect()

    detector = None
    if args.device:
        from percept2act.detector import Detector

        detector = Detector(scn, device=args.device)

    # Everything except the detector input; those are the display-only views.
    used = str(scn.require("cameras.detector_input"))
    side_roles = [r for r in ("overhead", "front") if r != used and scn.get(f"cameras.{r}")]
    print("opening side cameras (display only)...")
    sides = SideCameras(scn, side_roles)

    sorted_count = 0
    try:
        executor = build_executor(args.executor, scn, robot=robot)
        with Orchestrator(scn, robot=robot, executor=executor, detector=detector) as orch:
            crop = scn.require("inspection_station.crop")
            stats = (f"detector {orch.detector.device}  |  policy {executor.name}/"
                     f"{executor.device}  |  vote {scn.get('detector.vote_frames', 12)} frames")
            print(f"\n{stats}\n")
            state = {"status": "", "verdict": None}

            def draw(frame, patch=None, frac=0.0, run=0, need=0):
                cv2.imshow(WIN, compose(frame, crop, state, sides, side_roles, stats))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt

            def refresh():
                f = orch.camera.read()
                if f is not None:
                    draw(f)

            orch.goto_inspect_pose()

            while True:
                state["status"] = "WAITING - place a brick at the inspect spot"
                state["verdict"] = None
                orch.wait_for_brick(present=True, stable_frames=args.settle, on_frame=draw)

                state["status"] = "INSPECTING - Anomalib PatchCore"
                refresh()
                verdict, reinspections = orch.resolve_verdict()
                state["verdict"] = verdict

                brick_class = verdict.verdict
                if brick_class == "unsure":
                    brick_class = scn.require("classes.unsure.fallback_class")
                plate = scn.plate_for_class(brick_class)
                instruction = scn.instruction_for_class(brick_class)
                print(f"  score {verdict.score:.4f} -> {verdict.verdict} -> {plate}\n"
                      f"  {instruction!r}")

                deadline = time.time() + args.delay
                while (left := deadline - time.time()) > 0:
                    state["status"] = f'{instruction}   |   moving in {left:0.1f}s'
                    refresh()

                state["status"] = f"ACT - SmolVLA placing in the {plate} plate"
                refresh()

                # Redraw from inside the control loop. The policy loop blocks the
                # main thread for the whole placement, so without this the feed
                # freezes on the last pre-motion frame -- exactly when the arm is
                # the thing worth watching. obs already carries the wrist frame
                # the policy just used (RGB), so this costs no extra camera read.
                def on_step(step, obs):
                    f = None
                    if obs is not None and isinstance(obs.get(used), np.ndarray):
                        f = cv2.cvtColor(obs[used], cv2.COLOR_RGB2BGR)
                    elif step % 3 == 0:
                        f = orch.camera.read()   # replay gives no obs
                    if f is not None:
                        state["status"] = f"ACT - placing in the {plate} plate   step {step}"
                        draw(f)

                executor.step_hook = on_step
                try:
                    steps = orch.place(brick_class)
                finally:
                    executor.step_hook = None
                sorted_count += 1
                print(f"  done in {steps} steps  ({sorted_count} sorted)")

                state["status"] = "returning to the inspect pose"
                state["verdict"] = None
                refresh()
                orch.goto_inspect_pose()

                if args.bricks and sorted_count >= args.bricks:
                    break

                state["status"] = "clear the spot for the next brick"
                orch.wait_for_brick(present=False, stable_frames=args.settle, timeout=20,
                                    on_frame=draw)

            print(f"\n{sorted_count} bricks sorted")
            print(orch.latency.table())

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sides.close()
        cv2.destroyAllWindows()
        if robot is not None:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
