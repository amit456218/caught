import time

from caught.episodes import Episode
from caught.focus import Sessions, score_window
from caught.store import EpisodeRow


def row(habit, start, end):
    return EpisodeRow(0, habit, start, end, 0.8, 0.9, 0.8, None, "rule", 10, None, None, False, False, 0.0)


def test_score_window_penalises_phone_and_finds_clean_stretch():
    t0 = 1000.0
    eps = [row("phone", t0 + 600, t0 + 900), row("slouch", t0 + 100, t0 + 400), row("yawn", t0 + 50, t0 + 53)]
    s = score_window(eps, t0, t0 + 3000)
    assert 0 <= s["score"] < 100
    assert s["phone_pickups"] == 1 and s["phone_minutes"] == 5.0
    assert s["longest_clean_minutes"] == 35.0  # 900 -> 3000
    assert s["fix_first"] == "phone"
    assert s["breakdown"][0]["habit"] == "phone"
    clean = score_window([], t0, t0 + 3000)
    assert clean["score"] == 100 and clean["fix_first"] is None


def test_sessions_lifecycle(store):
    sess = Sessions(store)
    assert sess.current() is None
    s = sess.start(25, "Reading")
    assert s["end"] is None and s["label"] == "Reading"
    time.sleep(0.05)
    ep = Episode("phone", s["start"] + 0.01, time.time())
    ep.n_frames, ep.score_sum, ep.peak = 10, 8.0, 0.9
    store.commit_episode(ep, False)
    live = sess.live()
    assert live["summary"]["phone_pickups"] == 1
    done = sess.end()
    assert done["end"] is not None and done["summary"]["score"] < 100
    assert sess.current() is None
    assert len(sess.recent()) == 1
    sess.delete(done["id"])
    assert sess.recent() == []
