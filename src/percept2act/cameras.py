"""Uniform frame access over the UVC wrist cam and the RealSense cameras.

The rig mixes a Sonix UVC module with two RealSense D415s, which have completely
different capture APIs. Everything downstream just wants `.read() -> BGR array`,
so that difference is confined to this module.

Two gotchas encoded here, both found on the rig:
  * the Sonix module loses its v4l2 exposure settings on replug and defaults to
    blowing out the whole frame, so exposure is re-applied on open;
  * the D415 needs a dozen throwaway frames before auto-exposure settles.
"""

from __future__ import annotations

import subprocess
from typing import Protocol

import cv2
import numpy as np

from percept2act.config import Scenario

# Found by sweeping the Sonix module against the mat: mean ~98, ~1% saturated.
WRIST_EXPOSURE = 60
REALSENSE_WARMUP_FRAMES = 15


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class UVCStream:
    """A plain V4L2 camera, e.g. the Sonix wrist module."""

    def __init__(self, path: str, width: int, height: int, exposure: int | None = WRIST_EXPOSURE):
        self.path = path
        if exposure is not None:
            self._set_exposure(path, exposure)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open UVC camera {path}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):  # let auto-gain settle
            self._cap.read()

    @staticmethod
    def _set_exposure(path: str, exposure: int) -> None:
        """Re-apply manual exposure. Best effort: a driver that refuses is not fatal."""
        for ctrl in (f"auto_exposure=1", f"exposure_time_absolute={exposure}"):
            subprocess.run(
                ["v4l2-ctl", "-d", path, f"--set-ctrl={ctrl}"],
                capture_output=True,
                check=False,
            )

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class RealSenseStream:
    """Colour stream from one RealSense D415, selected by serial number."""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self.serial = serial
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipe.start(cfg)
        for _ in range(REALSENSE_WARMUP_FRAMES):
            try:
                self._pipe.wait_for_frames(timeout_ms=3000)
            except Exception:  # noqa: BLE001
                break

    def read(self) -> np.ndarray | None:
        frames = self._pipe.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None

    def close(self) -> None:
        try:
            self._pipe.stop()
        except Exception:  # noqa: BLE001
            pass


def open_stream(scn: Scenario, role: str) -> FrameSource:
    """Open the camera configured under `cameras.<role>`."""
    kind = scn.require(f"cameras.{role}.kind")
    width = scn.require(f"cameras.{role}.width")
    height = scn.require(f"cameras.{role}.height")
    fps = scn.require(f"cameras.{role}.fps")

    if kind == "opencv":
        return UVCStream(str(scn.require(f"cameras.{role}.index_or_path")), width, height)
    if kind == "realsense":
        return RealSenseStream(str(scn.require(f"cameras.{role}.serial")), width, height, fps)
    raise ValueError(f"unknown camera kind {kind!r} for role {role!r}")


def grab_crop(scn: Scenario, stream: FrameSource) -> tuple[np.ndarray, np.ndarray]:
    """Return (full_frame, inspection_crop) from an already-open stream."""
    frame = stream.read()
    if frame is None:
        raise RuntimeError("camera returned no frame")
    x0, y0, x1, y1 = scn.require("inspection_station.crop")
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError(
            f"inspection_station.crop {[x0, y0, x1, y1]} is empty for a "
            f"{frame.shape[1]}x{frame.shape[0]} frame"
        )
    return frame, patch


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper; use it to focus a lens."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brick_fraction(patch) -> float:
    """Fraction of a crop that looks like a brick rather than bare mat.

    Brightness, not colour saturation. The mat is dark green and matte; every
    brick we sort is brighter than it regardless of hue. An earlier version
    tested `(s > 70) & (v > 60)`, which silently rejected the pale pink and
    yellow bricks -- their saturation is low even though they are plainly
    visible -- and quarantined two entire bricks out of the training set.

    Measured over 110 crops containing a brick, across 6 colours:
        v > 100   floor 0.0397   (every brick, every colour)
        s/v combo floor 0.0041   (pink and yellow indistinguishable from mat)
    Bare mat sits around 0.004, so the 0.025 gate has margin on both sides.
    """
    import cv2

    v = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 2]
    return float((v > 100).mean())
