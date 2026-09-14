from .base import Detector, Temporal, ramp  # noqa: F401
from .rules import ALL_DETECTORS  # noqa: F401


def build_detectors(overrides: dict[str, dict[str, float]] | None = None) -> dict[str, Detector]:
    overrides = overrides or {}
    dets: dict[str, Detector] = {}
    for cls in ALL_DETECTORS:
        dets[cls.habit] = cls(overrides.get(cls.habit))
    return dets
