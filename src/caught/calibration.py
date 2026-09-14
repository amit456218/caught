"""Posture calibration: record a neutral upright baseline for a few seconds."""
from __future__ import annotations

import json
from pathlib import Path

from .features import Calibration, FrameFeatures


def load_calibration(path: Path) -> Calibration:
    if path.exists():
        try:
            return Calibration.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return Calibration()


def save_calibration(path: Path, cal: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.to_dict(), indent=2))


class CalibrationRecorder:
    """Collects feature frames for `seconds`, then produces a Calibration."""

    def __init__(self, seconds: float = 10.0):
        self.seconds = seconds
        self.samples: list[FrameFeatures] = []
        self.started: float | None = None
        self.result: Calibration | None = None

    @property
    def active(self) -> bool:
        return self.started is not None and self.result is None

    def start(self, ts: float) -> None:
        self.samples = []
        self.started = ts
        self.result = None

    def progress(self, ts: float) -> float:
        if self.started is None:
            return 0.0
        return min(1.0, (ts - self.started) / self.seconds)

    def feed(self, f: FrameFeatures) -> Calibration | None:
        if not self.active:
            return None
        if f.present:
            self.samples.append(f)
        if f.ts - (self.started or f.ts) >= self.seconds:
            self.result = Calibration.from_samples(self.samples)
            return self.result
        return None
