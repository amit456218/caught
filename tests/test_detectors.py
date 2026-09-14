from caught.detectors import build_detectors
from caught.features import FAR, FrameFeatures


def base():
    return FrameFeatures(present=1.0, face_conf=1.0, face_size=0.3, n_hands=0.0)


def test_absent():
    d = build_detectors()["absent"]
    assert d.score(FrameFeatures(present=0.0)) == 1.0
    assert d.score(base()) == 0.0


def test_phone_needs_hand_or_face_proximity():
    d = build_detectors()["phone"]
    f = base()
    f.phone_conf = 0.8
    f.phone_dist_hand, f.phone_dist_face = FAR, FAR
    assert d.score(f) < 0.3  # phone on the desk
    f.phone_in_hand = 1.0
    assert d.score(f) > 0.9


def test_hand_to_face_suppressed_by_cup():
    d = build_detectors()["hand_to_face"]
    f = base()
    f.n_hands = 1.0
    f.hand_mouth_dist = 0.1
    s = d.score(f)
    assert s > 0.8
    f.drink_dist_mouth = 0.2
    f.drink_recent = 1.0
    assert d.score(f) < 0.3


def test_slouch_requires_calibration_signal():
    d = build_detectors()["slouch"]
    f = base()
    assert d.score(f) == 0.0
    f.posture_dev = 6.0
    assert d.score(f) == 1.0


def test_looking_down_prefers_hidden_hands():
    d = build_detectors()["looking_down"]
    f = base()
    f.pitch = 30.0
    f.wrists_visible = 0.0
    hidden = d.score(f)
    f.n_hands = 2.0
    f.wrists_visible = 0.9
    f.hand_y_min = 0.7
    typing = d.score(f)
    assert hidden > typing > 0


def test_drowsy_ignores_looking_down():
    d = build_detectors()["drowsy"]
    f = base()
    f.eye_closed = 0.9
    assert d.score(f) == 1.0
    f.pitch = 35.0
    assert d.score(f) == 0.0


def test_overrides_apply_to_params_and_temporal():
    dets = build_detectors({"phone": {"conf_lo": 0.5, "min_on": 9.0}})
    assert dets["phone"].params["conf_lo"] == 0.5
    assert dets["phone"].temporal.min_on == 9.0
    assert build_detectors()["phone"].temporal.min_on != 9.0  # class default untouched


def test_all_scores_in_unit_range():
    dets = build_detectors()
    f = base()
    f.phone_conf = f.drink_conf = f.food_conf = 1.0
    f.phone_in_hand = f.drink_in_hand = 1.0
    f.hand_mouth_dist = f.hand_eye_dist = f.drink_dist_mouth = f.food_dist_mouth = 0.0
    f.n_hands = 2.0
    f.eye_closed = f.jaw_open = f.mouth_open = 1.0
    f.motion = 1.0
    f.pitch = 60.0
    f.yaw = 80.0
    f.posture_dev = 10.0
    for k, d in dets.items():
        s = d.score(f)
        assert 0.0 <= s <= 1.0, k
