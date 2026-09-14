"""The temporal core: smoothing, hysteresis, dwell, min duration, gap merging."""
from caught.detectors.base import Temporal
from caught.episodes import EpisodeTracker, HabitState
from caught.features import FrameFeatures


def run(state: HabitState, seq: list[tuple[float, float]], present=True):
    events = []
    for ts, score in seq:
        events.extend(state.update(ts, score, score, None, present))
    return events


def test_short_blip_does_not_start_episode():
    st = HabitState("phone", Temporal(ema_tau=0.2, min_on=2.0, min_off=1.0, min_duration=1.0, merge_gap=1.0))
    seq = [(t / 10, 1.0 if 5 <= t < 12 else 0.0) for t in range(60)]  # 0.7s of signal
    ev = run(st, seq)
    assert not any(k == "start" for k, _ in ev)


def test_sustained_signal_starts_ends_commits():
    st = HabitState("phone", Temporal(ema_tau=0.2, on_threshold=0.5, off_threshold=0.3, min_on=1.0, min_off=1.0,
                                      min_duration=2.0, merge_gap=1.0))
    seq = [(t / 10, 1.0 if 10 <= t < 80 else 0.0) for t in range(140)]
    ev = run(st, seq)
    kinds = [k for k, _ in ev]
    assert kinds == ["start", "end", "commit"]
    ep = ev[-1][1]
    assert 1.0 <= ep.start <= 1.6
    assert 7.5 <= ep.end <= 8.6
    assert ep.confidence > 0.8


def test_gap_shorter_than_merge_gap_extends_previous_episode():
    st = HabitState("phone", Temporal(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=1.0, merge_gap=5.0))
    seq = [(t / 10, 1.0 if (10 <= t < 50 or 70 <= t < 110) else 0.0) for t in range(200)]
    ev = run(st, seq)
    commits = [e for k, e in ev if k == "commit"]
    assert len(commits) == 1
    assert any(k == "resume" for k, _ in ev)
    assert commits[0].end - commits[0].start > 8.0


def test_gap_longer_than_merge_gap_gives_two_episodes():
    st = HabitState("phone", Temporal(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=1.0, merge_gap=1.0))
    seq = [(t / 10, 1.0 if (10 <= t < 50 or 90 <= t < 130) else 0.0) for t in range(200)]
    ev = run(st, seq)
    assert len([e for k, e in ev if k == "commit"]) == 2


def test_too_short_episode_is_discarded():
    st = HabitState("phone", Temporal(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=10.0, merge_gap=1.0))
    seq = [(t / 10, 1.0 if 10 <= t < 40 else 0.0) for t in range(100)]
    ev = run(st, seq)
    assert [k for k, _ in ev] == ["start", "end", "discard"]


def test_requires_present_gates_score():
    st = HabitState("slouch", Temporal(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=1.0, merge_gap=1.0))
    ev = run(st, [(t / 10, 1.0) for t in range(50)], present=False)
    assert ev == []


def test_flush_commits_open_episode():
    st = HabitState("phone", Temporal(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=1.0, merge_gap=1.0))
    run(st, [(t / 10, 1.0) for t in range(50)])
    ev = st.flush(6.0)
    assert [k for k, _ in ev] == ["end", "commit"]


def test_tracker_blends_model_scores():
    tr = EpisodeTracker({"phone": Temporal(ema_tau=0.1, min_on=0.3, min_off=0.3, min_duration=0.5, merge_gap=0.5)})
    f = FrameFeatures(present=1.0)
    events = []
    for t in range(40):
        f.ts = t / 10
        # rules say no, model says yes; with blend 1.0 the model wins
        events.extend(tr.update(f, {"phone": 0.0}, {"phone": 1.0}, blend=1.0))
    assert any(k == "start" for k, _ in events)
    assert events[0][1].source == "blend"


def test_drink_retracts_fresh_hand_to_face():
    from caught.detectors.base import Temporal as T
    tr = EpisodeTracker({"hand_to_face": T(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=1.0, merge_gap=1.0),
                         "drink": T(ema_tau=0.1, min_on=0.5, min_off=0.5, min_duration=0.5, merge_gap=1.0)})
    f = FrameFeatures(present=1.0)
    events = []
    for t in range(80):
        f.ts = t / 10
        htf = 1.0 if 5 <= t < 60 else 0.0      # hand goes up at 0.5 s
        drink = 1.0 if 30 <= t < 60 else 0.0   # the sip is recognised at 3.0 s
        events.extend(tr.update(f, {"hand_to_face": htf, "drink": drink}, None, 0.0))
    events.extend(tr.flush(8.0))
    kinds = [(k, e.habit) for k, e in events]
    assert ("start", "hand_to_face") in kinds
    assert ("discard", "hand_to_face") in kinds
    assert ("commit", "hand_to_face") not in kinds
    assert ("commit", "drink") in kinds
