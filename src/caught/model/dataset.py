"""Build training windows from stored feature history plus episode labels.

Supervision sources, strongest first:
  * human labels on episodes (confirmed = positive, rejected = negative), weight 5
  * unlabeled rule episodes (weak positives), weight 1
  * time outside any episode of a habit, while present (weak negatives), weight 0.3
Unknown targets are masked out of the loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HABIT_KEYS
from ..features import FEATURE_NAMES
from ..store import EpisodeRow, Store

WINDOW = 16      # frames
HZ = 2.0         # feature store cadence
PRESENT_IDX = FEATURE_NAMES.index("present")
NEG_MARGIN = 20.0  # seconds around an episode we refuse to call negative


@dataclass
class Windows:
    x: np.ndarray        # (N, WINDOW, F)
    y: np.ndarray        # (N, H) in {0, 1}
    w: np.ndarray        # (N, H) sample weights, 0 = unknown
    t: np.ndarray        # (N,) window end timestamps
    n_human: int


def _target_curves(ts: np.ndarray, episodes: list[EpisodeRow]) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestamp target and weight for each habit."""
    H = len(HABIT_KEYS)
    y = np.zeros((len(ts), H), np.float32)
    w = np.full((len(ts), H), 0.3, np.float32)  # default: weak negative
    idx = {k: i for i, k in enumerate(HABIT_KEYS)}
    # first pass: blank out margins so we never contradict a nearby episode
    for e in episodes:
        h = idx[e.habit]
        m = (ts >= e.start - NEG_MARGIN) & (ts < e.end + NEG_MARGIN)
        w[m, h] = 0.0
    # second pass: positives / explicit negatives
    for e in sorted(episodes, key=lambda e: 0 if e.label is None else 1):  # human labels last so they win
        h = idx[e.habit]
        m = (ts >= e.start) & (ts < e.end)
        if e.label is None:
            y[m, h] = 1.0
            w[m, h] = 1.0
        elif e.label == 1:
            y[m, h] = 1.0
            w[m, h] = 5.0
        else:
            y[m, h] = 0.0
            w[m, h] = 5.0
    return y, w


def build_windows(store: Store, start: float, end: float, max_windows: int = 60000,
                  seed: int = 0) -> Windows | None:
    ts, vecs = store.features_between(start, end)
    if len(ts) < WINDOW * 4:
        return None
    episodes = store.episodes_between(start, end, include_rejected=True)
    y_all, w_all = _target_curves(ts, episodes)
    present = vecs[:, PRESENT_IDX] > 0.5
    # absent habit is the exception: negatives require presence, positives require absence
    # for every other habit weak negatives are only meaningful while present
    for i, k in enumerate(HABIT_KEYS):
        if k != "absent":
            w_all[~present & (w_all[:, i] == 0.3), i] = 0.0

    # candidate window ends: every stored frame with a valid contiguous window (no big gaps)
    ends = np.arange(WINDOW - 1, len(ts))
    span = ts[ends] - ts[ends - (WINDOW - 1)]
    ok = span < (WINDOW / HZ) * 2.5
    ends = ends[ok]
    if len(ends) == 0:
        return None
    # oversample informative windows: anything with a strong label, then a random slice of the rest
    strong = ends[(w_all[ends] >= 1.0).any(axis=1)]
    weak = ends[~(w_all[ends] >= 1.0).any(axis=1)]
    rng = np.random.default_rng(seed)
    n_weak = min(len(weak), max(len(strong) * 3, 2000), max_windows - len(strong))
    chosen = np.concatenate([strong, rng.choice(weak, n_weak, replace=False) if n_weak > 0 else weak[:0]])
    if len(chosen) > max_windows:
        chosen = rng.choice(chosen, max_windows, replace=False)
    chosen.sort()
    x = np.stack([vecs[e - WINDOW + 1:e + 1] for e in chosen])
    n_human = sum(1 for e in episodes if e.label is not None)
    return Windows(x, y_all[chosen], w_all[chosen], ts[chosen], n_human)
