"""Detector interface: each habit gets an instantaneous 0..1 score per frame.

Temporal logic (smoothing, hysteresis, minimum durations, gap merging) lives in
episodes.py so that rule scores and learned-model scores go through the same path.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..features import FrameFeatures


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp from 0 at lo to 1 at hi, clamped."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


@dataclass
class Temporal:
    """Per-habit temporal parameters (seconds)."""
    ema_tau: float = 1.0          # exponential smoothing time constant
    on_threshold: float = 0.5
    off_threshold: float = 0.3
    min_on: float = 2.0           # score must stay above on_threshold this long to start
    min_off: float = 2.0          # score must stay below off_threshold this long to end
    min_duration: float = 4.0     # episodes shorter than this are discarded
    merge_gap: float = 8.0        # a restart within this gap extends the previous episode
    requires_present: bool = True


class Detector:
    habit: str = ""
    temporal: Temporal = Temporal()
    params: dict[str, float] = {}

    def __init__(self, overrides: dict[str, float] | None = None):
        self.params = dict(type(self).params)
        if overrides:
            for k, v in overrides.items():
                if k in self.params:
                    self.params[k] = float(v)
                elif hasattr(self.temporal, k):
                    self.temporal = Temporal(**{**self.temporal.__dict__, k: float(v)})

    def score(self, f: FrameFeatures) -> float:  # pragma: no cover - abstract
        raise NotImplementedError
