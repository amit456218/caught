import time
from datetime import date

import numpy as np

from caught.config import Config
from caught.episodes import Episode
from caught.features import FEATURE_DIM
from caught.model.active import should_review
from caught.model.dataset import build_windows
from caught.model.infer import ModelScorer
from caught.model.train import train
from caught.report import build_week_report, render_html
from caught.simulate import simulate


def test_commit_and_query(store):
    now = time.time()
    ep = Episode("phone", now - 120, now - 60)
    ep.n_frames, ep.score_sum, ep.peak = 10, 7.0, 0.9
    ep.thumbnails = [b"\xff\xd8jpg"]
    ep.thumb_times = [now - 120]
    eid = store.commit_episode(ep, review=True)
    rows = store.episodes_between(now - 300, now)
    assert len(rows) == 1 and rows[0].id == eid and rows[0].has_samples
    assert store.thumbnail_count(eid) == 1
    q = store.review_queue()
    assert q and q[0].id == eid
    store.set_label(eid, False)
    assert store.review_queue() == []
    assert store.episodes_between(now - 300, now) == []  # rejected hidden by default
    assert len(store.episodes_between(now - 300, now, include_rejected=True)) == 1
    assert store.label_counts()["rejected"] == 1


def test_day_and_week_summaries(store):
    now = time.time()
    for i in range(3):
        ep = Episode("slouch", now - 3600 * (i + 1), now - 3600 * (i + 1) + 600)
        ep.n_frames, ep.score_sum, ep.peak = 5, 4.0, 0.9
        store.commit_episode(ep, False)
    d = store.day_summary(date.today())
    assert d["totals"]["slouch"]["count"] >= 1
    w = store.week_summary(date.today())
    assert len(w["days"]) == 7
    assert sum(sum(r) for r in w["heatmap"]["slouch"]) > 0


def test_feature_storage_and_prune(store):
    now = time.time()
    for i in range(50):
        store.add_feature(now - 100 + i, np.full(FEATURE_DIM, i, np.float32))
    store.flush()
    ts, vecs = store.features_between(now - 200, now + 1)
    assert len(ts) == 50 and vecs.shape == (50, FEATURE_DIM)
    store.add_feature(now - 40 * 86400, np.zeros(FEATURE_DIM, np.float32))
    store.flush()
    res = store.prune(retention_days=21)
    assert res["features_deleted"] == 1


def test_should_review_policy(cfg):
    ep = Episode("phone", 0, 10)
    ep.n_frames, ep.score_sum, ep.peak = 10, 5.0, 0.6
    assert not should_review(ep, cfg)  # no thumbnails
    ep.thumbnails = [b"x"]
    assert should_review(ep, cfg)  # uncertain
    ep.score_sum = 9.9
    ep.rule_sum, ep.model_sum = 9.9, 1.0
    assert should_review(ep, cfg)  # model disagrees


def test_simulate_train_infer_report(store, cfg):
    res = simulate(store, cfg, days=3, seed=1, log=lambda *_: None)
    assert res["episodes"] > 50 and res["features"] > 1000
    now = time.time()
    win = build_windows(store, now - 4 * 86400, now, max_windows=3000)
    assert win is not None and win.x.shape[1:] == (16, FEATURE_DIM)
    out = train(store, cfg.model_path, days=4, epochs=1, max_seconds=60, log=lambda *_: None)
    assert out["ok"] and cfg.model_path.exists()
    scorer = ModelScorer(cfg.model_path)
    assert scorer.ready
    scores = None
    for i in range(40):
        scores = scorer.push(now + i, np.zeros(FEATURE_DIM, np.float32))
    assert scores is not None and set(scores) >= {"phone", "slouch"}
    rep = build_week_report(store, cfg)
    assert rep["present_minutes"] > 0 and len(rep["habits"]) == 12
    html = render_html(rep)
    assert "Caught" in html and "minutes" in html
