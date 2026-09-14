"""Adaptive frame-rate policy so an all-day process stays cool."""
from __future__ import annotations

from .config import Config


class PowerPolicy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = "active"
        self._still_frames = 0

    def update(self, present: bool, motion: float, any_active: bool) -> tuple[str, float, int]:
        """Return (mode, target_fps, object_detection_every_n_frames)."""
        c = self.cfg
        if not present:
            self.mode = "absent"
            self._still_frames = 0
            return self.mode, c.fps_absent, 6
        if motion < 0.006 and not any_active:
            self._still_frames += 1
        else:
            self._still_frames = 0
        if self._still_frames > 40:
            self.mode = "idle"
            return self.mode, c.fps_idle, c.object_every_n_frames * 2
        self.mode = "active"
        return self.mode, c.fps_active, c.object_every_n_frames
