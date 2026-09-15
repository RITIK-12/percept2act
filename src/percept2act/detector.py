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
        """BGR uint8 crop -> NCHW float32 in [0, 1].

        Deliberately NOT ImageNet-normalized. Anomalib 2.x bakes its
        PreProcessor into the exported graph, so normalizing here too applies it
        twice and saturates every output to 1.0 for both classes -- the model
        looks broken while actually being fed garbage. Measured on this export:
        [0,1] input separates good 0.000 from defective 0.539; pre-normalized
        input gives 1.000 for both.
        """
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
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
        """Pull the image-level anomaly score out of the model's outputs.

        The export exposes `pred_score`, `pred_label`, `anomaly_map` and
        `pred_mask`. Select `pred_score` by name rather than by shape --
        `pred_label` is also a single element, and picking it would collapse
        every score to a hard 0/1 and make the uncertainty band meaningless.
        """
        for out in self._outputs:
            if "pred_score" in out.get_names():
                return float(np.asarray(result[out]).reshape(-1)[0])
        # Older exports omit the scalar; PatchCore's image score is the max of
        # its anomaly map, so fall back to that.
        for out in self._outputs:
            if "anomaly_map" in out.get_names():
                return float(np.asarray(result[out]).max())
        raise RuntimeError(
            f"no pred_score or anomaly_map output; got "
            f"{[o.get_names() for o in self._outputs]}"
        )

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
