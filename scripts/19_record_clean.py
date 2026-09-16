#!/usr/bin/env python
"""Record teleop episodes that ALL start from the same pose.

Why this exists alongside 06_record.sh
--------------------------------------
`lerobot-record` keeps teleop live during its reset phase, so the follower ends
up wherever the leader left it and every episode starts somewhere different.
The first dataset drifted ~20 degrees between episodes and mixed single- and
multi-brick takes; the policy trained on it could not grip reliably.

Physical AI Studio's own recorder cannot fix this either -- its command is just
start_recording(task) / save_episode, with no home-pose parameter. So recording
happens here, and training still happens in Studio: physicalai's
LeRobotDataModule reads this dataset straight from disk, no import step.

BOTH arms are driven back to inspection_station.pose between episodes. The
leader is a teleoperator but it is the same servo hardware, so torque can be
enabled and it can be driven exactly like the follower.

That matters: parking only the follower and then engaging teleop with the leader
somewhere else snaps the follower across on the first frame, which is how the
original ~20 degree drift got into the dataset. With both parked they start
matched, and the leader's torque is released only once you have a hand on it.

Usage
-----
  python scripts/19_record_clean.py --class coral --episodes 20
  python scripts/19_record_clean.py --class blue  --episodes 20

A live wrist-camera window shows the leader alignment while you line up, then
the frame count while recording. SPACE or q in that window ends an episode
(ENTER in the terminal works too). Ctrl-C stops and keeps everything saved.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from percept2act.config import Scenario  # noqa: E402

WIN = "percept2act - record  (SPACE/q = end episode)"


def preview(obs: dict, role: str, crop, lines: list[tuple[str, tuple]], recording: bool) -> int:
    """Show the wrist feed with a status overlay. Returns the key pressed.

    The robot is configured with color_mode=RGB (that is what gets recorded), so
    the frame has to be flipped back to BGR for cv2 or the preview shows the
    brick in the wrong colour and you second-guess a perfectly good take.
    """
    frame = obs.get(role)
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return -1
    view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if crop:
        x0, y0, x1, y1 = crop
        cv2.rectangle(view, (x0, y0), (x1, y1), (90, 90, 90), 1)
    if recording:
        cv2.circle(view, (view.shape[1] - 24, 24), 9, (0, 0, 255), -1)

    for i, (text, colour) in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)

    cv2.imshow(WIN, view)
    return cv2.waitKey(1) & 0xFF


def _reader(flag: dict) -> None:
    """Background ENTER reader, so ending an episode does not block the loop."""
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    flag["stop"] = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True,
                    choices=["good", "defective", "coral", "blue"],
                    help="destination plate: coral=defective, blue=good")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=25.0, help="max episode length")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to ramp back")
    args = ap.parse_args()

    cls = {"coral": "defective", "blue": "good"}.get(args.cls, args.cls)

    scn = Scenario.load()
    task = scn.instruction_for_class(cls)
    plate = scn.plate_for_class(cls)
    # The same guard 06_record.sh carries: a truncated instruction is identical
    # for both plates, which silently destroys instruction conditioning.
    if " " not in task:
        raise SystemExit(f"instruction {task!r} has no space -- truncated somewhere")

    pose_cfg = scn.get("inspection_station.pose")
    if pose_cfg is None:
        raise SystemExit("inspection_station.pose is not set; run scripts/05_set_inspect_pose.py")
    target = np.asarray(pose_cfg, dtype=float)

    crop = scn.get("inspection_station.crop")
    role = str(scn.require("cameras.detector_input"))
    repo_id = str(scn.require("policies.stage2_sort.dataset_repo_id"))
    root = REPO / str(scn.require("replay.dataset_root"))
    fps = int(scn.require("policies.control_fps"))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    from percept2act.lerobot_args import camera_objects

    cams = camera_objects(scn, list(scn.require("cameras.policy_inputs")))
    robot = SO101Follower(SO101FollowerConfig(
        port=str(scn.require("arms.follower.port")),
        id=str(scn.require("arms.follower.id")),
        cameras=cams,
    ))
    leader = SO101Leader(SO101LeaderConfig(
        port=str(scn.require("arms.leader.port")),
        id=str(scn.require("arms.leader.id")),
    ))

    print(f"task    : {task!r}")
    print(f"plate   : {plate}")
    print(f"dataset : {repo_id}  ({root})")
    print(f"cameras : {list(cams)}")

    robot.connect()
    leader.connect()

    names = list(robot.action_features)
    pose = {n: float(v) for n, v in zip(names, target)}
    threads = 4 * max(1, len(cams))

    if root.exists():
        dataset = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=threads)
        print(f"resuming: {dataset.num_episodes} episode(s) already recorded")
    else:
        dataset = LeRobotDataset.create(
            repo_id, fps, root=root, robot_type=robot.name,
            features={
                **hw_to_dataset_features(robot.action_features, "action", use_video=True),
                **hw_to_dataset_features(robot.observation_features, "observation", use_video=True),
            },
            use_videos=True,
            image_writer_threads=threads,
        )
        print("created a new dataset")

    bare = [n.removesuffix(".pos") for n in names]

    def park() -> None:
        """Drive BOTH arms to the inspect pose.

        The leader is a teleoperator, but it is the same servo hardware, so
        torque can be enabled and it can be driven like the follower. Parking
        both means they start every episode already matched -- no lining up by
        hand, and no snap on the first teleop frame.

        Interpolated, never stepped: a step command to a distant target pulls
        full current at once and trips shoulder_lift's overload latch.
        """
        leader.enable_torque()
        obs = robot.get_observation()
        cur_f = np.array([float(obs.get(n, pose[n])) for n in names], dtype=float)
        lead = leader.get_action()
        cur_l = np.array([float(lead.get(n, pose[n])) for n in names], dtype=float)

        steps = max(1, int(args.settle * fps))
        for i in range(1, steps + 1):
            f = cur_f + (target - cur_f) * (i / steps)
            l = cur_l + (target - cur_l) * (i / steps)
            robot.send_action({n: float(v) for n, v in zip(names, f)})
            leader.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(bare, l)})
            time.sleep(1.0 / fps)

    recorded = 0
    try:
        for ep in range(1, args.episodes + 1):
            print(f"\n--- episode {ep}/{args.episodes} ---")
            print("  parking the follower...")
            park()

            # Both arms hold the pose while the brick is placed. Torque stays ON
            # here so the leader does not sag out of position while waiting.
            print("  place the brick, hold the leader handle, then SPACE to record")
            while True:
                obs = robot.get_observation()
                robot.send_action(pose)
                k = preview(obs, role, crop, [
                    (f"episode {ep}/{args.episodes}   ->  {plate} plate", (255, 255, 255)),
                    ("both arms parked at the inspect pose", (0, 255, 0)),
                    ("place the brick + hold the leader, then SPACE", (0, 255, 255)),
                ], recording=False)
                if k in (ord(" "), 13, 10):
                    break
                if k == ord("q"):
                    raise KeyboardInterrupt
                time.sleep(1.0 / fps)

            # Release the leader only now, with a hand already on it.
            leader.disable_torque()
            print("    RECORDING -- SPACE/q to end")

            flag = {"stop": False}
            threading.Thread(target=_reader, args=(flag,), daemon=True).start()

            t0, frames = time.time(), 0
            while not flag["stop"] and time.time() - t0 < args.seconds:
                loop_t = time.perf_counter()
                obs = robot.get_observation()
                act = leader.get_action()
                robot.send_action(act)
                dataset.add_frame({
                    **build_dataset_frame(dataset.features, obs, prefix="observation"),
                    **build_dataset_frame(dataset.features, act, prefix="action"),
                    "task": task,
                })
                frames += 1

                elapsed = time.time() - t0
                k = preview(obs, role, crop, [
                    (f"REC  episode {ep}/{args.episodes}   {plate} plate", (255, 255, 255)),
                    (f"{frames} frames   {elapsed:4.1f}s / {args.seconds:.0f}s", (255, 255, 255)),
                    (task, (0, 255, 255)),
                ], recording=True)
                if k in (ord(" "), ord("q")):
                    flag["stop"] = True

                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - loop_t)))

            dataset.save_episode()
            recorded += 1
            print(f"  saved {frames} frames ({frames / fps:.1f}s)")
            if not flag["stop"]:
                print("  hit the time limit (ended on --seconds, not a keypress)")

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        print("\nparking and disconnecting...")
        try:
            park()
        except Exception:  # noqa: BLE001 -- never block disconnect on a park failure
            pass
        cv2.destroyAllWindows()
        try:
            leader.disable_torque()   # never leave the handle stiff
        except Exception:  # noqa: BLE001
            pass
        robot.disconnect()
        leader.disconnect()

    other = "blue" if cls == "defective" else "coral"
    print(f"\n{recorded} recorded this run, {dataset.num_episodes} total in {root}")
    print("\nNext:")
    print(f"  the other class :  python scripts/19_record_clean.py --class {other} --episodes {args.episodes}")
    print("  check balance   :  bash scripts/07_verify_dataset.sh")
    print("  train in Studio :  bash scripts/16_studio_train.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
