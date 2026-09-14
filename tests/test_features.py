import numpy as np

from caught.features import FEATURE_DIM, FEATURE_NAMES, Calibration, FeatureExtractor, FrameFeatures
from caught.perception.landmarks import Perception
from caught.perception.objects import Detection


def test_vector_roundtrip():
    f = FrameFeatures(pitch=12.5, phone_conf=0.7)
    v = f.to_vector()
    assert v.shape == (FEATURE_DIM,)
    g = FrameFeatures.from_vector(v)
    assert g.pitch == 12.5 and g.phone_conf == np.float32(0.7)
    assert FEATURE_NAMES[0] == "ts"


def fake_face(cx=0.5, cy=0.45, size=0.3):
    face = np.zeros((478, 3), np.float32)
    face[:, 0] = cx
    face[:, 1] = cy
    face[152] = [cx, cy + size / 2, 0]      # chin
    face[10] = [cx, cy - size / 2, 0]       # forehead
    face[1] = [cx, cy, 0]                   # nose
    face[61] = [cx - 0.05, cy + 0.1, 0]     # mouth L
    face[291] = [cx + 0.05, cy + 0.1, 0]    # mouth R
    face[13] = [cx, cy + 0.09, 0]
    face[14] = [cx, cy + 0.11, 0]
    for i in (159, 145):
        face[i] = [cx - 0.06, cy - 0.05, 0]
    for i in (386, 374):
        face[i] = [cx + 0.06, cy - 0.05, 0]
    face[234] = [cx - 0.15, cy, 0]
    face[454] = [cx + 0.15, cy, 0]
    return face


def test_hand_near_mouth_distance_is_small():
    p = Perception(ts_ms=0, width=640, height=360, face=fake_face(), hands_ran=True)
    hand = np.zeros((21, 3), np.float32)
    hand[:, 0] = 0.5 * 360 / 640 * (640 / 360)  # will be scaled by aspect inside; keep normalized x=0.5
    hand[:, 0] = 0.5
    hand[:, 1] = 0.9
    hand[8] = [0.5, 0.55, 0]  # index tip at the mouth
    p.hands = [hand]
    fx = FeatureExtractor()
    f = fx.extract(np.zeros((360, 640, 3), np.uint8), p, [], ts=1000.0)
    assert f.present == 1.0 and f.n_hands == 1.0
    assert f.hand_mouth_dist < 0.2
    assert f.hand_eye_dist > f.hand_mouth_dist


def test_phone_in_hand_flag():
    p = Perception(ts_ms=0, width=640, height=360, face=fake_face(), hands_ran=True)
    hand = np.zeros((21, 3), np.float32)
    hand[:, 0] = 0.5
    hand[:, 1] = 0.8
    p.hands = [hand]
    det = [Detection("cell phone", 0.9, 0.45, 0.7, 0.55, 0.9)]
    f = FeatureExtractor().extract(np.zeros((360, 640, 3), np.uint8), p, det, ts=1000.0)
    assert f.phone_conf == 0.9
    assert f.phone_in_hand == 1.0
    assert f.phone_dist_hand == 0.0


def test_calibration_and_posture_deviation():
    samples = []
    for i in range(60):
        f = FrameFeatures(present=1.0, face_conf=1.0, face_size=0.3 + 0.002 * (i % 3), nose_y=0.4, shoulder_y=0.8,
                          shoulder_w=0.5, nose_shoulder=0.8, pitch=2.0, torso_lean=0.1)
        samples.append(f)
    cal = Calibration.from_samples(samples)
    assert cal.valid and cal.n_samples == 60
    fx = FeatureExtractor(cal)
    p = Perception(ts_ms=0, width=640, height=360, face=fake_face(size=0.3))
    pose = np.zeros((33, 4), np.float32)
    pose[:, 3] = 1.0
    pose[11] = [0.3, 0.8, 0.0, 1.0]
    pose[12] = [0.7, 0.8, 0.0, 1.0]
    pose[0] = [0.5, 0.45, -0.1, 1.0]
    p.pose = pose
    upright = fx.extract(np.zeros((360, 640, 3), np.uint8), p, [], ts=1.0)
    # now sink the head toward the shoulders and lean in
    p2 = Perception(ts_ms=1, width=640, height=360, face=fake_face(cy=0.6, size=0.4), pose=pose.copy())
    slouched = fx.extract(np.zeros((360, 640, 3), np.uint8), p2, [], ts=2.0)
    assert slouched.posture_dev > upright.posture_dev + 2.0


def test_motion_energy_changes_with_content():
    fx = FeatureExtractor()
    a = np.zeros((360, 640, 3), np.uint8)
    b = np.full((360, 640, 3), 200, np.uint8)
    assert fx.motion_energy(a) == 0.0
    assert fx.motion_energy(b) > 0.5
    assert fx.motion_energy(b) == 0.0
