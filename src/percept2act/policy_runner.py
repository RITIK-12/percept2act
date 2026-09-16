"""Run a trained LeRobot policy on the follower arm.

This is the ACT stage. A policy consumes (robot state, camera images, language
task) and emits joint targets. The language task is the whole point here: the
same SmolVLA checkpoint routes a brick to either plate depending only on the
instruction string, which is how Anomalib's verdict reaches the robot.

Everything runs in the `hack_lerobot` env -- one process, because that env has
lerobot, anomalib and OpenVINO together.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


class PolicyRunner:
    """Wraps a LeRobot policy and drives the robot with it at a fixed rate."""

    def __init__(
        self,
        checkpoint: str,
        policy_type: str,
        device: str = "xpu",
        fps: int = 30,
        dataset_root: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.policy_type = policy_type
        self.fps = fps
        self.device = self._pick_device(device)
        self.policy = self._load()
        self.policy.eval()
        self.features, self.robot_type = self._load_feature_spec(dataset_root)
        self.preprocessor, self.postprocessor = self._make_processors()

    def _load_feature_spec(self, dataset_root: str | None):
        """Feature spec from the dataset the policy was trained on.

        build_dataset_frame orders the state vector by the dataset's own
        `names` list. Deriving that ordering from anywhere else risks feeding
        the policy joints in a different order than it was trained on, which
        does not raise -- it just drives the arm somewhere wrong.
        """
        import json

        if dataset_root is None:
            return None, None
        info = Path(dataset_root) / "meta" / "info.json"
        if not info.exists():
            log.warning("no dataset info at %s; falling back to observation order", info)
            return None, None
        data = json.loads(info.read_text())
        return data.get("features"), data.get("robot_type")

    def _make_processors(self):
        """The checkpoint's own pre/post-processor pipelines.

        These are not optional plumbing. The preprocessor runs
        tokenizer_processor, which produces `observation.language.tokens` --
        SmolVLA reads that key directly and raises without it -- and
        normalizer_processor. The postprocessor runs unnormalizer_processor, so
        skipping it returns actions in normalized space rather than joint
        degrees.
        """
        from lerobot.policies import make_pre_post_processors

        override = {"device_processor": {"device": self.device}}
        return make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides=override,
            postprocessor_overrides=override,
        )

    @staticmethod
    def _pick_device(requested: str) -> str:
        if requested == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested not in ("cpu",):
            log.warning("device %r unavailable, falling back to cpu", requested)
        return "cpu"

    def _load(self):
        """Load the checkpoint through LeRobot's policy registry."""
        from lerobot.policies.factory import get_policy_class

        cls = get_policy_class(self.policy_type)
        policy = cls.from_pretrained(self.checkpoint)
        return policy.to(self.device)

    # -- observation plumbing -------------------------------------------------

    def build_batch(
        self,
        observation: dict[str, Any],
        task: str,
        image_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Turn a LeRobot robot observation into a policy batch.

        The robot hands back joint positions as scalars and camera frames as HWC
        uint8 arrays. Policies want a batched float tensor of state and NCHW
        float images in [0, 1], plus the task string.
        """
        batch: dict[str, Any] = {}

        state_vals, frames = [], {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                frames[key] = value
            elif isinstance(value, (int, float, np.floating, np.integer)):
                state_vals.append(float(value))

        if state_vals:
            batch["observation.state"] = torch.tensor(
                state_vals, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        for key, frame in frames.items():
            if image_keys and key not in image_keys:
                continue
            arr = frame.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            # LeRobot names camera observations `observation.images.<role>`;
            # a raw robot observation may already use that key.
            name = key if key.startswith("observation.") else f"observation.images.{key}"
            batch[name] = tensor

        batch["task"] = [task]
        return batch

    @torch.no_grad()
    def step(self, observation: dict[str, Any], task: str) -> Any:
        """One control step: raw robot values + instruction -> joint targets.

        Goes through LeRobot's own predict_action so the checkpoint's processor
        pipeline runs exactly as it did in training.
        """
        from lerobot.common.control_utils import predict_action
        from lerobot.utils.feature_utils import build_dataset_frame

        if self.features is not None:
            frame = build_dataset_frame(self.features, observation, prefix="observation")
        else:
            frame = self.build_batch(observation, task)

        action = predict_action(
            frame,
            self.policy,
            torch.device(self.device),
            self.preprocessor,
            self.postprocessor,
            use_amp=False,
            task=task,
            robot_type=self.robot_type,
        )
        return np.asarray(action.squeeze(0).float().cpu().numpy()).reshape(-1)

    def reset(self) -> None:
        """Clear the action queue between episodes.

        ACT and SmolVLA both emit action chunks and buffer them internally.
        Carrying that buffer across bricks makes the arm start the next pick with
        the tail of the previous place.
        """
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def run_until_done(
        self,
        robot,
        task: str,
        max_steps: int,
        action_names: list[str] | None = None,
        on_step=None,
        frames=None,
    ) -> int:
        """Drive the robot with this policy for up to `max_steps` control cycles.

        Returns the number of steps actually executed. There is no learned
        termination signal, so the caller bounds the stage by step count and
        checks the physical outcome afterwards.

        `frames` is an optional callable returning {name: HWC uint8 RGB}. The
        robot is connected with cameras={} so the detector and the policy do not
        fight over the same /dev/video node, which means get_observation() hands
        back joint state and nothing else -- the policy would then raise "All
        image features are missing from the batch". The caller supplies the
        wrist frame from the stream it already owns.
        """
        self.reset()
        names = action_names or list(robot.action_features)
        period = 1.0 / self.fps
        steps = 0

        for steps in range(1, max_steps + 1):
            t0 = time.perf_counter()
            obs = robot.get_observation()
            if frames is not None:
                extra = frames()
                if not extra:
                    # A dropped frame would raise inside the policy. Skip the
                    # cycle and try again rather than killing the run.
                    log.warning("no camera frame this cycle, skipping step %d", steps)
                    time.sleep(period)
                    continue
                obs = {**obs, **extra}
            values = self.step(obs, task)
            action = {name: float(v) for name, v in zip(names, np.asarray(values).reshape(-1))}
            robot.send_action(action)

            if on_step is not None and on_step(steps, obs, action) is False:
                break

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        return steps

    def __repr__(self) -> str:
        return (
            f"PolicyRunner({self.policy_type} @ {self.device}, "
            f"ckpt={self.checkpoint}, fps={self.fps})"
        )
