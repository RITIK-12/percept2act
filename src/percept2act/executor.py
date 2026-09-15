"""Swappable ACT stage.

The whole pipeline -- cameras, Anomalib on NPU, verdict, instruction string,
latency logging -- is identical no matter what actually moves the arm. Only the
last hop differs, so it lives behind one interface with three backends:

    stub     logs the decision, never touches the arm.  Validates the loop.
    replay   plays back one teleop demo per destination. No training required.
    policy   the fine-tuned SmolVLA, in torch or as OpenVINO IR.

Build and demo with `replay`, then switch to `policy` with a single flag once
training finishes. Nothing upstream changes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from percept2act.config import Scenario

log = logging.getLogger(__name__)


class Executor(Protocol):
    """Carries out the placement chosen by the reasoning stage."""

    name: str
    device: str

    def execute(self, brick_class: str, instruction: str) -> int:
        """Place the brick. Returns the number of control steps taken."""
        ...

    def close(self) -> None: ...


class StubExecutor:
    """Logs the decision and moves nothing.

    Use this to prove PERCEIVE -> DETECT -> REASON works before letting the arm
    move, and to rehearse the loop away from the rig.
    """

    name = "stub"
    device = "CPU"

    def __init__(self, scn: Scenario):
        self.scn = scn

    def execute(self, brick_class: str, instruction: str) -> int:
        plate = self.scn.plate_for_class(brick_class)
        log.info("[stub] would place in %s plate via: %r", plate, instruction)
        time.sleep(0.2)  # keep the loop's pacing realistic
        return 0

    def close(self) -> None:
        pass


class ReplayExecutor:
    """Plays back a recorded teleop demo, one per destination plate.

    Blind playback, so it only works when every brick starts at the same marked
    spot -- which is also what makes the detector accurate, since each brick is
    then inspected at the same pose and scale.

    No training required, which is why this is the backend that gets the loop
    closed today.
    """

    name = "replay"

    def __init__(self, scn: Scenario, robot: Any, episodes: dict[str, int] | None = None):
        self.scn = scn
        self.robot = robot
        self.device = "CPU"
        self.fps = int(scn.require("policies.control_fps"))
        self.dataset = self._load_dataset(scn)
        self.episodes = episodes or dict(scn.get("replay.episodes") or {})
        if not self.episodes:
            raise ValueError(
                "no replay episodes configured. Add to config/scenario.yaml:\n"
                "  replay:\n    episodes:\n      good: 0\n      defective: 1"
            )
        self._cache: dict[int, list[dict]] = {}

    @staticmethod
    def _load_dataset(scn: Scenario):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = scn.abs_path("policies.stage2_sort.export_dir").parent.parent / "datasets" / "stage2_sort"
        root = Path(scn.get("replay.dataset_root") or root)
        repo_id = str(scn.require("policies.stage2_sort.dataset_repo_id"))
        if not root.exists():
            raise FileNotFoundError(
                f"no recorded dataset at {root}\n"
                "  record two demos first:\n"
                "    bash scripts/03_record_stage2.sh defective 1\n"
                "    bash scripts/03_record_stage2.sh good 1"
            )
        return LeRobotDataset(repo_id, root=root)

    def _frames(self, episode: int) -> list[dict]:
        """Extract the action sequence for one episode, once, then cache it."""
        if episode in self._cache:
            return self._cache[episode]

        idx = self.dataset.episode_data_index
        start, end = int(idx["from"][episode]), int(idx["to"][episode])
        frames = [self.dataset[i] for i in range(start, end)]
        self._cache[episode] = frames
        log.info("loaded replay episode %d: %d frames", episode, len(frames))
        return frames

    def execute(self, brick_class: str, instruction: str) -> int:
        episode = self.episodes.get(brick_class)
        if episode is None:
            raise KeyError(f"no replay episode configured for class {brick_class!r}")

        plate = self.scn.plate_for_class(brick_class)
        log.info("[replay] episode %d -> %s plate  (%r)", episode, plate, instruction)

        frames = self._frames(episode)
        names = list(self.robot.action_features)
        period = 1.0 / self.fps

        for step, frame in enumerate(frames, 1):
            t0 = time.perf_counter()
            values = np.asarray(frame["action"]).reshape(-1)
            self.robot.send_action({n: float(v) for n, v in zip(names, values)})
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        return len(frames)

    def close(self) -> None:
        pass


class PolicyExecutor:
    """The trained SmolVLA, conditioned on the instruction string.

    This is the backend that makes the instruction do real work: the same
    checkpoint routes a brick to either plate based only on the sentence it is
    given. Swap to it once training converges.
    """

    name = "policy"

    def __init__(self, scn: Scenario, robot: Any, runtime: str = "torch"):
        self.scn = scn
        self.robot = robot
        self.runtime = runtime
        self.device = scn.device_for("stage2_sort")
        self.max_steps = int(scn.require("policies.max_steps_per_stage"))
        self.runner = self._build()

    def _build(self):
        from percept2act.policy_runner import PolicyRunner

        ckpt = self._find_checkpoint()
        log.info("policy checkpoint: %s", ckpt)
        return PolicyRunner(
            checkpoint=str(ckpt),
            policy_type=str(self.scn.require("policies.stage2_sort.policy")),
            device="xpu" if self.device in ("GPU", "XPU") else "cpu",
            fps=int(self.scn.require("policies.control_fps")),
        )

    def _find_checkpoint(self) -> Path:
        root = self.scn.abs_path("policies.stage2_sort.export_dir")
        for pattern in ("checkpoints/*/pretrained_model", "**/pretrained_model"):
            found = sorted(root.glob(pattern))
            if found:
                return found[-1]
        raise FileNotFoundError(
            f"no trained policy under {root}\n"
            "  train it:  bash scripts/05_train.sh smolvla\n"
            "  or run the loop with --executor replay in the meantime"
        )

    def execute(self, brick_class: str, instruction: str) -> int:
        plate = self.scn.plate_for_class(brick_class)
        log.info("[policy] %r -> %s plate", instruction, plate)
        return self.runner.run_until_done(
            robot=self.robot,
            task=instruction,
            max_steps=self.max_steps,
        )

    def close(self) -> None:
        pass


def build_executor(
    kind: str,
    scn: Scenario,
    robot: Any | None = None,
) -> Executor:
    """Construct the requested backend, failing with an actionable message."""
    if kind == "stub":
        return StubExecutor(scn)
    if robot is None:
        raise ValueError(f"executor {kind!r} needs a connected robot")
    if kind == "replay":
        return ReplayExecutor(scn, robot)
    if kind == "policy":
        return PolicyExecutor(scn, robot)
    raise ValueError(f"unknown executor {kind!r}; expected stub, replay or policy")
