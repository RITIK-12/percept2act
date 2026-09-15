"""Anomalib PatchCore inference on OpenVINO, with an explicit uncertainty band.

This is the DETECT stage. It turns an image crop into the structured defect
signal the robotics workflow consumes:

    {score, verdict, threshold, device, latency_ms}

`verdict` is one of good / defective / unsure. The `unsure` class exists because
the rubric scores "recovery from incorrect or uncertain detections" -- a score
sitting on the threshold is a real outcome, not something to round off silently.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from percept2act.config import Scenario


@dataclass
class Verdict:
    """The structured defect signal handed to the reasoning stage."""

    score: float
    verdict: str  # good | defective | unsure
    threshold: float
    band: float
    device: str
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def instruction_class(self) -> str:
        """The class whose instruction should drive the policy."""
        return self.verdict


class Detector:
    """PatchCore IR loaded through OpenVINO, pinned to a configured device."""

    def __init__(self, scn: Scenario, device: str | None = None, precision: str = "fp16"):
        self.scn = scn
        self.device = device or scn.device_for("detector")
        self.threshold = float(scn.get("detector.threshold") or 0.5)
        self.band = float(scn.require("detector.uncertainty_band"))
        self.input_h, self.input_w = scn.require("detector.input_size")

        self.model_path = self._resolve_ir(scn, precision)
        self._compiled, self._input, self._outputs = self._compile()

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def _resolve_ir(scn: Scenario, precision: str) -> Path:
        """Find the exported IR, preferring the manifest written by training."""
        import json

        root = Path(scn.abs_path("detector.export_dir"))
        manifest = root / "exports.json"
        if manifest.exists():
            exports = json.loads(manifest.read_text())
            if precision in exports:
                p = Path(exports[precision])
                xml = p if p.suffix == ".xml" else next(p.glob("**/*.xml"), None)
                if xml and xml.exists():
                    return xml
        xml = next(root.glob("**/*.xml"), None)
        if xml is None:
            raise FileNotFoundError(
                f"no OpenVINO IR under {root}. Train the detector first:\n"
                "  python scripts/07_train_anomalib.py"
            )
        return xml

    def _compile(self):
        import openvino as ov

        core = ov.Core()
        if self.device not in core.available_devices:
            raise RuntimeError(
                f"device {self.device!r} not available; OpenVINO sees "
                f"{core.available_devices}. Change openvino.devices.detector."
            )
        model = core.read_model(self.model_path)

        # The NPU plugin requires fully static shapes. Anomalib exports with a
        # dynamic batch dimension, so pin it to 1 before compiling or the NPU
        # silently refuses the model.
        try:
            model.reshape({0: [1, 3, self.input_h, self.input_w]})
        except Exception:  # noqa: BLE001
            pass  # already static, or a layout this does not apply to

        compiled = core.compile_model(model, self.device)
        return compiled, compiled.input(0), list(compiled.outputs)

    # -- inference -----------------------------------------------------------

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 crop -> normalized NCHW float32, matching Anomalib's export."""
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return np.transpose(arr, (2, 0, 1))[None, ...]

    def score(self, bgr: np.ndarray) -> Verdict:
        """Score one crop and classify it against the threshold and band."""
        tensor = self.preprocess(bgr)
        t0 = time.perf_counter()
        result = self._compiled({self._input: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw = self._extract_score(result)
        return Verdict(
            score=raw,
            verdict=self.classify(raw),
            threshold=self.threshold,
            band=self.band,
            device=self.device,
            latency_ms=latency_ms,
        )

    def _extract_score(self, result) -> float:
        """Pull the scalar image-level score out of the model's outputs.

        Anomalib's OpenVINO export emits an anomaly map and, depending on
        version, a separate scalar. Prefer the scalar; otherwise reduce the map
        by its max, which is the image-level score PatchCore is defined by.
        """
        scalars, maps = [], []
        for out in self._outputs:
            arr = np.asarray(result[out])
            (scalars if arr.size == 1 else maps).append(arr)
        if scalars:
            return float(scalars[0].reshape(-1)[0])
        if maps:
            return float(max(m.max() for m in maps))
        raise RuntimeError("detector produced no usable output")

    def classify(self, score: float) -> str:
        """Map a score to good / defective / unsure.

        The band is deliberately checked FIRST: a score inside it is uncertain
        regardless of which side of the threshold it falls on.
        """
        if abs(score - self.threshold) <= self.band:
            return "unsure"
        return "defective" if score > self.threshold else "good"

    def warmup(self, n: int = 3) -> None:
        """Run a few throwaway inferences.

        The first NPU inference includes graph compilation, so timing it would
        make the demo's latency table look far worse than steady state.
        """
        dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        for _ in range(n):
            self.score(dummy)

    def __repr__(self) -> str:
        return (
            f"Detector(device={self.device}, threshold={self.threshold:.4f}, "
            f"band={self.band}, ir={self.model_path.name})"
        )
