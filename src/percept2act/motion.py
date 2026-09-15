"""Safe point-to-point moves for the follower.

Two things everything else needs:

  * `ramp_to` -- interpolate to a target pose instead of stepping to it. A step
    command to a distant target makes the servos pull maximum current at once,
    which is what trips shoulder_lift's overload latch and drops it off the bus.
  * `median_start_pose` -- derive a sensible inspect pose from recorded demos
    rather than asking someone to type in six joint angles.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

RAMP_FPS = 30


def current_pose(robot: Any, names: list[str]) -> np.ndarray:
    """Read the follower's present joint positions, in action-feature order."""
    obs = robot.get_observation()
    out = []
    for n in names:
        # Observations expose joints as "<joint>.pos"; action features use the
        # same key, but fall back to a bare name just in case.
        v = obs.get(n, obs.get(n.replace(".pos", ""), None))
        out.append(float(v) if v is not None else 0.0)
    return np.asarray(out, dtype=float)


def ramp_to(
    robot: Any,
    target: np.ndarray,
    seconds: float = 2.0,
    names: list[str] | None = None,
) -> int:
    """Interpolate from the current pose to `target` over `seconds`.

    Returns the number of control steps sent.
    """
    names = names or list(robot.action_features)
    target = np.asarray(target, dtype=float).reshape(-1)
    start = current_pose(robot, names)

    steps = max(1, int(seconds * RAMP_FPS))
    period = 1.0 / RAMP_FPS
    for i in range(1, steps + 1):
        blend = start + (target - start) * (i / steps)
        t0 = time.perf_counter()
        robot.send_action({n: float(v) for n, v in zip(names, blend)})
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
    return steps


def hold(robot: Any, pose: np.ndarray, seconds: float, names: list[str] | None = None) -> None:
    """Keep commanding one pose, so the arm does not sag under gravity."""
    names = names or list(robot.action_features)
    cmd = {n: float(v) for n, v in zip(names, np.asarray(pose).reshape(-1))}
    end = time.time() + seconds
    while time.time() < end:
        robot.send_action(cmd)
        time.sleep(1.0 / RAMP_FPS)


def episode_frames(dataset_root: str | Path, episode: int) -> np.ndarray:
    """Action sequence for one episode, shape (T, n_joints)."""
    import pandas as pd

    root = Path(dataset_root)
    eps = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/**/*.parquet"))]
    )
    row = eps[eps["episode_index"] == episode]
    if row.empty:
        raise KeyError(
            f"episode {episode} not in {root}; have {sorted(eps['episode_index'].tolist())}"
        )
    start = int(row.iloc[0]["dataset_from_index"])
    end = int(row.iloc[0]["dataset_to_index"])

    data = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("data/**/*.parquet"))]
    ).sort_values("index").reset_index(drop=True)
    return np.stack(data.iloc[start:end]["action"].values)


def median_start_pose(dataset_root: str | Path) -> np.ndarray:
    """Median first-frame pose across all recorded episodes.

    Teleop start poses drift by ~20 degrees episode to episode, so no single
    recording defines "the" inspect pose. The median is close to all of them and
    therefore a short, safe ramp from wherever the arm happens to be.
    """
    import pandas as pd

    root = Path(dataset_root)
    eps = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/**/*.parquet"))]
    )
    data = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("data/**/*.parquet"))]
    ).sort_values("index").reset_index(drop=True)

    starts = [
        np.asarray(data.iloc[int(r["dataset_from_index"])]["action"]).reshape(-1)
        for _, r in eps.iterrows()
    ]
    return np.median(np.stack(starts), axis=0)
