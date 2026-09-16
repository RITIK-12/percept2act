"""The closed loop: PERCEIVE -> DETECT -> REASON -> ACT -> OPTIMIZE.

    for each brick:
        PERCEIVE  camera frame of the workspace
        DETECT    Anomalib PatchCore on OpenVINO/NPU  -> score
        REASON    score + band -> verdict -> instruction string
        ACT       SmolVLA, conditioned on that instruction, places the brick
        OPTIMIZE  every stage logs its device and latency as it goes

The reasoning step is deliberately small and legible: a verdict selects one of
two English sentences, and the policy does the rest. That is the seam between
the detector and the robot, and it is the thing to point at when explaining how
defect output reaches the manipulation workflow.

Run with scripts/12_demo.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from percept2act.cameras import grab_crop, open_stream
from percept2act.config import Scenario
from percept2act.detector import Detector, Verdict
from percept2act.executor import StubExecutor
from percept2act.latency import LatencyLog

log = logging.getLogger(__name__)


@dataclass
class BrickResult:
    index: int
    verdict: str
    score: float
    instruction: str
    plate: str | None
    steps: int
    reinspections: int = 0
    note: str = ""


@dataclass
class RunReport:
    results: list[BrickResult] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def summary(self) -> str:
        if not self.results:
            return "no bricks processed"
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
        elapsed = time.time() - self.started
        lines = [
            f"{len(self.results)} bricks in {elapsed:.1f}s "
            f"({elapsed / len(self.results):.1f}s per brick)",
            "  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            "",
            f"  {'#':>3} {'verdict':<10} {'score':>8} {'plate':<7} {'steps':>6}  note",
        ]
        for r in self.results:
            lines.append(
                f"  {r.index:>3} {r.verdict:<10} {r.score:>8.4f} "
                f"{str(r.plate or '-'):<7} {r.steps:>6}  {r.note}"
            )
        return "\n".join(lines)


class Orchestrator:
    """Sequences detection and manipulation for every brick in the workspace."""

    def __init__(
        self,
        scn: Scenario,
        robot: Any | None = None,
        executor: Any | None = None,
        detector: Detector | None = None,
        pause_between: bool = True,
    ):
        self.scn = scn
        self.robot = robot
        # Defaults to the stub so the loop is always runnable, even with no arm
        # connected and no policy trained.
        self.executor = executor or StubExecutor(scn)
        self.pause_between = pause_between
        self.latency = LatencyLog(
            scn.abs_path("runtime.latency_log"),
            enabled=bool(scn.get("openvino.log_latency", True)),
        )
        self.detector = detector or Detector(scn)
        self.detector.warmup()
        self.role = scn.require("cameras.detector_input")
        self.camera = open_stream(scn, self.role)
        self.max_bricks = int(scn.require("runtime.max_bricks_per_run"))

        # The robot is connected with cameras={} so the detector keeps exclusive
        # use of the wrist camera. A policy executor still needs that image, so
        # lend it this stream.
        if hasattr(self.executor, "set_frame_source"):
            self.executor.set_frame_source(self.policy_frames)

    def policy_frames(self) -> dict[str, Any]:
        """Wrist frame for the policy, in the colour order it was trained on.

        LeRobot's cameras default to ColorMode.RGB, so the recorded episodes are
        RGB while our streams are BGR. Skipping this conversion does not crash --
        it silently feeds the policy inverted colours, which is far harder to
        spot than a missing key.
        """
        import cv2

        frame = self.camera.read()
        if frame is None:
            return {}
        return {self.role: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)}

    # -- PERCEIVE: waiting for a brick ----------------------------------------

    def brick_fraction(self, patch: Any) -> float:
        """Fraction of the crop occupied by a brick. See cameras.brick_fraction."""
        from percept2act.cameras import brick_fraction

        return brick_fraction(patch)

    def wait_for_brick(
        self,
        present: bool = True,
        stable_frames: int = 12,
        timeout: float = 0.0,
        on_frame=None,
    ) -> bool:
        """Block until the crop is occupied (or cleared) and has stopped changing.

        `stable_frames` consecutive agreeing frames are required, which is what
        stops the cycle firing on the hand that is still placing the brick.
        Returns False on timeout.
        """
        threshold = float(self.scn.get("watch.brick_fraction", 0.025))
        run = 0
        started = time.time()

        while True:
            frame = self.camera.read()
            if frame is None:
                continue
            x0, y0, x1, y1 = self.scn.require("inspection_station.crop")
            patch = frame[y0:y1, x0:x1]
            frac = self.brick_fraction(patch)
            occupied = frac >= threshold

            run = run + 1 if occupied == present else 0
            if on_frame is not None:
                on_frame(frame, patch, frac, run, stable_frames)
            if run >= stable_frames:
                return True
            if timeout and time.time() - started > timeout:
                return False
            time.sleep(1.0 / 30)

    # -- DETECT + REASON ------------------------------------------------------

    def goto_inspect_pose(self) -> None:
        """Return the arm to the fixed inspect pose.

        The wrist camera moves with the arm, so scores are only comparable
        between bricks if every inspection happens from the same pose. Without
        this the detector would be comparing a close-up against a wide shot and
        calling the difference an anomaly.
        """
        pose = self.scn.get("inspection_station.pose")
        if pose is None or self.robot is None:
            return
        from percept2act.motion import ramp_to

        with self.latency.measure("act.goto_inspect", "CPU"):
            ramp_to(self.robot, pose, float(self.scn.get("replay.ramp_seconds", 2.0)))

    def inspect(self) -> tuple[Verdict, Any]:
        """Grab a frame and score the brick at the inspection crop."""
        with self.latency.measure("perceive.capture", "CPU"):
            _, patch = grab_crop(self.scn, self.camera)
        with self.latency.measure("detect.patchcore", self.detector.device) as m:
            verdict = self.detector.score(patch)
        m["score"] = round(verdict.score, 5)
        m["verdict"] = verdict.verdict
        return verdict, patch

    def resolve_verdict(self) -> tuple[Verdict, int]:
        """Inspect, re-inspecting while the score sits inside the uncertainty band.

        This is the recovery behaviour the reliability criterion asks for. An
        uncertain score is never silently rounded to the nearest class; it is
        re-examined, and only then resolved to the configured fallback.
        """
        max_reinspect = int(self.scn.require("classes.unsure.max_reinspect"))
        verdict, _ = self.inspect()
        attempts = 0

        while verdict.verdict == "unsure" and attempts < max_reinspect:
            attempts += 1
            behavior = self.scn.require("classes.unsure.behavior")
            log.warning(
                "score %.4f is within +/-%.3f of threshold %.4f -> %s (attempt %d/%d)",
                verdict.score, verdict.band, verdict.threshold, behavior, attempts, max_reinspect,
            )
            if behavior == "hold":
                break
            time.sleep(0.4)  # let the scene settle; a hand may still be in frame
            verdict, _ = self.inspect()

        return verdict, attempts

    # -- ACT ------------------------------------------------------------------

    def place(self, brick_class: str) -> int:
        """Hand the instruction to whichever executor is plugged in."""
        instruction = self.scn.instruction_for_class(brick_class)

        with self.latency.measure(f"act.{self.executor.name}", self.executor.device) as m:
            steps = self.executor.execute(brick_class, instruction)
        m["instruction"] = instruction
        m["steps"] = steps
        return steps

    # -- the loop -------------------------------------------------------------

    def run(self, bricks: int | None = None) -> RunReport:
        report = RunReport()
        limit = bricks or self.max_bricks

        for index in range(1, limit + 1):
            log.info("--- brick %d/%d ---", index, limit)

            with self.latency.measure("loop.brick_total", "CPU"):
                self.goto_inspect_pose()
                verdict, reinspections = self.resolve_verdict()

                brick_class = verdict.verdict
                note = ""
                if brick_class == "unsure":
                    brick_class = self.scn.require("classes.unsure.fallback_class")
                    note = f"unsure after {reinspections} re-inspection(s) -> {brick_class}"
                    log.warning(note)

                instruction = self.scn.instruction_for_class(brick_class)
                plate = self.scn.plate_for_class(brick_class)
                log.info(
                    "score=%.4f verdict=%s -> %s plate | %r",
                    verdict.score, verdict.verdict, plate, instruction,
                )

                steps = self.place(brick_class)

            report.results.append(
                BrickResult(
                    index=index,
                    verdict=verdict.verdict,
                    score=verdict.score,
                    instruction=instruction,
                    plate=plate,
                    steps=steps,
                    reinspections=reinspections,
                    note=note,
                )
            )

            if self.pause_between and index < limit:
                input("  reset the scene, then press Enter for the next brick... ")

        return report

    def close(self) -> None:
        self.camera.close()

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
