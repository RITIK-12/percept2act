"""Per-stage device + latency logging.

The rubric requires explaining workload placement and observed results out loud.
Reconstructing numbers at the end of the day does not work, so every inference
stage records device and wall-clock latency from the first run.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class LatencyLog:
    """Append-only JSONL log of {stage, device, ms}, plus a demo-ready table."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._samples: dict[tuple[str, str], list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str, device: str, **extra: Any) -> Iterator[dict]:
        """Time a block and record it under (stage, device)."""
        payload: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield payload
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.record(stage, device, ms, **{**extra, **payload})

    def record(self, stage: str, device: str, ms: float, **extra: Any) -> None:
        self._samples[(stage, device)].append(ms)
        if not self.enabled:
            return
        row = {"ts": time.time(), "stage": stage, "device": device, "ms": round(ms, 3)}
        row.update(extra)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def summary(self) -> list[dict[str, Any]]:
        out = []
        for (stage, device), samples in self._samples.items():
            ordered = sorted(samples)
            out.append(
                {
                    "stage": stage,
                    "device": device,
                    "n": len(samples),
                    "mean_ms": round(statistics.fmean(samples), 2),
                    "p50_ms": round(statistics.median(samples), 2),
                    "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
                }
            )
        return sorted(out, key=lambda r: -r["mean_ms"])

    def table(self) -> str:
        """The stage/device/latency table for the live demo."""
        rows = self.summary()
        if not rows:
            return "(no latency samples recorded)"
        head = f"{'stage':<22} {'device':<8} {'n':>5} {'mean':>9} {'p50':>9} {'p95':>9}"
        lines = [head, "-" * len(head)]
        for r in rows:
            lines.append(
                f"{r['stage']:<22} {r['device']:<8} {r['n']:>5} "
                f"{r['mean_ms']:>8.2f}ms {r['p50_ms']:>8.2f}ms {r['p95_ms']:>8.2f}ms"
            )
        return "\n".join(lines)

    @classmethod
    def from_file(cls, path: str | Path) -> "LatencyLog":
        """Rebuild a log from a previous run, for the demo writeup."""
        log = cls(path, enabled=False)
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                log._samples[(row["stage"], row["device"])].append(row["ms"])
        return log
