"""Active learning policy: which episodes to ask the human about, and when to retrain."""
from __future__ import annotations

import random
import time

from ..config import Config
from ..episodes import Episode
from ..store import Store


def should_review(ep: Episode, cfg: Config, rng: random.Random | None = None) -> bool:
    """Queue uncertain episodes, rule/model disagreements, and a random QA slice."""
    if not ep.thumbnails:
        return False
    rng = rng or random
    c = ep.confidence
    if cfg.review_uncertain_low <= c <= cfg.review_uncertain_high:
        return True
    if ep.model_sum and abs(ep.rule_conf - ep.model_conf) > 0.4:
        return True
    return rng.random() < cfg.review_random_rate


def should_retrain(store: Store, min_new_labels: int = 5, max_age_hours: float = 24.0) -> tuple[bool, str]:
    last = store.last_model_run()
    counts = store.label_counts()
    n_labels = counts["confirmed"] + counts["rejected"]
    if last is None:
        return (store.feature_count() > 4000, "no model yet")
    new_since = len(store.labeled(since=last["ts"] - 7 * 86400)) if n_labels else 0
    if new_since >= min_new_labels:
        return True, f"{new_since} new labels"
    if time.time() - last["ts"] > max_age_hours * 3600:
        return True, "model older than a day"
    return False, "up to date"
