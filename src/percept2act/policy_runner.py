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
    ):
        self.checkpoint = checkpoint
        self.policy_type = policy_type
        self.fps = fps
        self.device = self._pick_device(device)
        self.policy = self._load()
        self.policy.eval()

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
    def step(self, observation: dict[str, Any], task: str) -> dict[str, Any]:
        """One control step: observation + instruction -> action dict."""
        batch = self.build_batch(observation, task)
        action = self.policy.select_action(batch)
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).float().cpu().numpy()
        return action

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
    ) -> int:
        """Drive the robot with this policy for up to `max_steps` control cycles.

        Returns the number of steps actually executed. There is no learned
        termination signal, so the caller bounds the stage by step count and
        checks the physical outcome afterwards.
        """
        self.reset()
        names = action_names or list(robot.action_features)
        period = 1.0 / self.fps
        steps = 0

        for steps in range(1, max_steps + 1):
            t0 = time.perf_counter()
            obs = robot.get_observation()
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
