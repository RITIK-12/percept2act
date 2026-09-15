"""Scenario config loading.

Nothing scenario-specific is hardcoded anywhere else in this package. If a value
needs to change because of something the organizers revealed, it changes in
config/scenario.yaml and nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "scenario.yaml"


class Scenario:
    """Dict-backed config with dotted-path access and TODO@RIG guards."""

    def __init__(self, data: dict[str, Any], path: Path):
        self._data = data
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Scenario":
        p = Path(path or os.environ.get("PERCEPT2ACT_CONFIG", DEFAULT_CONFIG))
        with open(p) as fh:
            return cls(yaml.safe_load(fh), p)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Fetch a value, failing loudly if it is missing or still a placeholder.

        Catching TODO@RIG here turns 'the arm moved to the wrong place and we
        spent 20 minutes guessing why' into an immediate, named error.
        """
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"{self.path}: missing required key '{dotted}'")
        if value is None:
            raise ValueError(f"{self.path}: '{dotted}' is null — fill it in")
        if isinstance(value, str) and value.startswith("TODO@"):
            raise ValueError(
                f"{self.path}: '{dotted}' is still {value!r}.\n"
                "  Run scripts/01_assign_cameras.py to fill in camera serials."
            )
        return value

    # -- convenience accessors -------------------------------------------------

    def instruction(self, key: str) -> str:
        return self.require(f"task.instructions.{key}")

    def instruction_for_class(self, class_name: str) -> str:
        """Map a detector verdict to the language instruction that drives SmolVLA."""
        key = self.require(f"classes.{class_name}.instruction")
        return self.instruction(key)

    def plate_for_class(self, class_name: str) -> str:
        return self.require(f"classes.{class_name}.plate")

    def device_for(self, stage: str) -> str:
        return self.require(f"openvino.devices.{stage}")

    def abs_path(self, dotted: str) -> Path:
        """Resolve a config path value relative to the repo root."""
        return (REPO_ROOT / str(self.require(dotted))).resolve()

    @property
    def detector_url(self) -> str:
        host = self.require("runtime.detector_host")
        port = self.require("runtime.detector_port")
        return f"http://{host}:{port}"

    def __repr__(self) -> str:
        return f"Scenario({self.path})"
