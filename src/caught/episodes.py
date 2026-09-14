"""Turn per-frame habit scores into episodes.

Per habit: EMA smoothing -> hysteresis with minimum on/off dwell -> episode with
minimum duration -> gap merging. Episodes are committed (yielded) only once the
merge window has passed, so a brief pause does not split one phone session into two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .detectors.base import Temporal
from .features import FrameFeatures


@dataclass
class Episode:
    habit: str
    start: float
    end: float
    score_sum: float = 0.0
    n_frames: int = 0
    peak: float = 0.0
    source: str = "rule"
    thumbnails: list[bytes] = field(default_factory=list)   # tiny JPEGs, kept in memory until commit
    thumb_times: list[float] = field(default_factory=list)
    id: int | None = None
    rule_sum: float = 0.0
    model_sum: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def confidence(self) -> float:
        if not self.n_frames:
            return 0.0
        mean = self.score_sum / self.n_frames
        return min(1.0, 0.5 * mean + 0.5 * self.peak)

    @property
    def rule_conf(self) -> float:
        return self.rule_sum / self.n_frames if self.n_frames else 0.0

    @property
    def model_conf(self) -> float:
        return self.model_sum / self.n_frames if self.n_frames else 0.0


@dataclass
class HabitState:
    habit: str
    temporal: Temporal
    smoothed: float = 0.0
    active: bool = False
    above_since: float | None = None
    below_since: float | None = None
    on_credit: float = 0.0     # seconds of evidence for starting, leaks when the score dips
    off_credit: float = 0.0    # seconds of evidence for stopping
    current: Episode | None = None
    pending: Episode | None = None     # closed, waiting out the merge gap
    pending_closed_at: float = 0.0
    last_ts: float = 0.0

    def update(self, ts: float, score: float, rule_score: float, model_score: float | None,
               present: bool) -> list[tuple[str, Episode]]:
        """Advance the state machine. Returns events: ("start"|"end"|"commit"|"discard", episode)."""
        events: list[tuple[str, Episode]] = []
        t = self.temporal
        dt = ts - self.last_ts if self.last_ts else 0.0
        self.last_ts = ts
        if dt > 5.0:  # camera paused; reset smoothing but keep episode bookkeeping
            self.smoothed = score
        elif dt > 0:
            a = 1.0 - math.exp(-dt / max(t.ema_tau, 1e-3))
            self.smoothed += a * (score - self.smoothed)
        else:
            self.smoothed = score
        s = self.smoothed
        if t.requires_present and not present:
            s = 0.0

        step = min(dt, 1.0) if dt > 0 else 0.0
        if not self.active:
            # dwell with tolerance: one dropped frame does not restart the clock
            if s >= t.on_threshold:
                if self.on_credit <= 0:
                    self.above_since = ts
                self.on_credit += step
            else:
                self.on_credit = max(0.0, self.on_credit - step)
                if self.on_credit <= 0:
                    self.above_since = None
            if self.on_credit >= t.min_on and self.above_since is not None:
                self.active = True
                self.off_credit = 0.0
                self.below_since = None
                start_ts = self.above_since
                if self.pending is not None and start_ts - self.pending.end <= t.merge_gap:
                    self.current = self.pending
                    self.pending = None
                    self.current.end = ts
                    events.append(("resume", self.current))
                else:
                    self.current = Episode(self.habit, start_ts, ts)
                    events.append(("start", self.current))
        else:
            if s < t.off_threshold:
                if self.off_credit <= 0:
                    self.below_since = ts
                self.off_credit += step
            else:
                self.off_credit = max(0.0, self.off_credit - step)
                if self.off_credit <= 0:
                    self.below_since = None
            if self.off_credit >= t.min_off and self.below_since is not None:
                self.active = False
                self.on_credit = 0.0
                self.above_since = None
                ep = self.current
                assert ep is not None
                ep.end = self.below_since
                self.current = None
                self.pending = ep
                self.pending_closed_at = ts
                events.append(("end", ep))
        if self.active and self.current is not None:
            ep = self.current
            ep.end = ts
            ep.score_sum += s
            ep.rule_sum += rule_score
            if model_score is not None:
                ep.model_sum += model_score
            ep.n_frames += 1
            ep.peak = max(ep.peak, s)

        if self.pending is not None and ts - self.pending.end > t.merge_gap:
            ep = self.pending
            self.pending = None
            events.append(("commit" if ep.duration >= t.min_duration else "discard", ep))
        return events

    def flush(self, ts: float) -> list[tuple[str, Episode]]:
        """Close everything (shutdown)."""
        events: list[tuple[str, Episode]] = []
        if self.current is not None:
            ep = self.current
            ep.end = ts
            self.current = None
            self.active = False
            events.append(("end", ep))
            events.append(("commit" if ep.duration >= self.temporal.min_duration else "discard", ep))
        if self.pending is not None:
            ep = self.pending
            self.pending = None
            events.append(("commit" if ep.duration >= self.temporal.min_duration else "discard", ep))
        return events


# A more specific explanation wins: when the suppressor becomes active, a suppressed episode
# that began within RETRACT_S seconds is retracted (it was the same gesture), a longer one is ended.
SUPPRESSED_BY: dict[str, tuple[str, ...]] = {
    "hand_to_face": ("drink", "eat", "phone"),
    "eye_rub": ("drink",),
    "looking_down": ("phone",),
}
RETRACT_S = 12.0


class EpisodeTracker:
    def __init__(self, temporals: dict[str, Temporal]):
        self.states = {k: HabitState(k, t) for k, t in temporals.items()}

    def update(self, f: FrameFeatures, rule_scores: dict[str, float],
               model_scores: dict[str, float] | None, blend: float) -> list[tuple[str, Episode]]:
        events: list[tuple[str, Episode]] = []
        present = f.present > 0
        for k, st in self.states.items():
            r = rule_scores.get(k, 0.0)
            m = model_scores.get(k) if model_scores else None
            s = r if m is None else (1.0 - blend) * r + blend * m
            if any(self.states[sup].active for sup in SUPPRESSED_BY.get(k, ()) if sup in self.states):
                s = 0.0
            for ev in st.update(f.ts, s, r, m, present):
                if m is not None:
                    ev[1].source = "blend"
                events.append(ev)
        events.extend(self._arbitrate(f.ts))
        return events

    def _arbitrate(self, ts: float) -> list[tuple[str, Episode]]:
        out: list[tuple[str, Episode]] = []
        for k, sups in SUPPRESSED_BY.items():
            st = self.states.get(k)
            if st is None or not any(self.states[sup].active for sup in sups if sup in self.states):
                continue
            if st.current is not None:
                ep = st.current
                st.current = None
                st.active = False
                st.above_since = st.below_since = None
                st.on_credit = st.off_credit = 0.0
                ep.end = ts
                out.append(("end", ep))
                if ts - ep.start <= RETRACT_S:
                    out.append(("discard", ep))
                else:
                    st.pending = ep
                    st.pending_closed_at = ts
            if st.pending is not None and ts - st.pending.end <= RETRACT_S and st.pending.duration <= RETRACT_S:
                out.append(("discard", st.pending))
                st.pending = None
        return out

    def flush(self, ts: float) -> list[tuple[str, Episode]]:
        out: list[tuple[str, Episode]] = []
        for st in self.states.values():
            out.extend(st.flush(ts))
        return out

    def snapshot(self) -> dict[str, dict]:
        return {k: {"score": round(st.smoothed, 3), "active": st.active,
                    "since": st.current.start if st.current else None}
                for k, st in self.states.items()}
