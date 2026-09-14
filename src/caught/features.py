"""Per-frame feature vector: geometry, blendshapes, hand-face relations, objects, motion.

Everything downstream (rule detectors, the learned model, the review UI) consumes
FrameFeatures. The field order of the dataclass is the vector order.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields

import cv2
import numpy as np

from .perception.landmarks import (
    CHIN, FINGER_PIPS, FINGERTIPS, LEFT_CHEEK, LEFT_EYE_BOT, LEFT_EYE_TOP, LIP_BOT, LIP_TOP,
    MOUTH_L, MOUTH_R, NOSE_TIP, P_LEAR, P_LEYE, P_LSHOULDER, P_LWRIST, P_NOSE, P_REAR, P_REYE,
    P_RSHOULDER, P_RWRIST, RIGHT_CHEEK, RIGHT_EYE_BOT, RIGHT_EYE_TOP, WRIST, LandmarkEngine, Perception,
)

P_MOUTH_L, P_MOUTH_R = 9, 10
RECENT_CAP = 30.0  # seconds; "not seen lately"
from .perception.objects import DRINK, FOOD, Detection

FAR = 3.0  # "no such thing in frame" distance, in face-heights


@dataclass
class FrameFeatures:
    ts: float = 0.0
    present: float = 0.0
    face_conf: float = 0.0
    face_size: float = 0.0
    face_cx: float = 0.5
    face_cy: float = 0.5
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    nose_y: float = 0.5
    shoulder_y: float = 0.0
    shoulder_w: float = 0.0
    nose_shoulder: float = 0.0
    shoulder_tilt: float = 0.0
    torso_lean: float = 0.0
    eye_closed: float = 0.0
    squint: float = 0.0
    gaze_h: float = 0.0
    gaze_v: float = 0.0
    jaw_open: float = 0.0
    mouth_open: float = 0.0
    mouth_pucker: float = 0.0
    mouth_funnel: float = 0.0
    cheek_puff: float = 0.0
    brow_down: float = 0.0
    brow_inner_up: float = 0.0
    n_hands: float = 0.0
    hand_mouth_dist: float = FAR
    hand_eye_dist: float = FAR
    hand_cheek_dist: float = FAR
    hand_face_dist: float = FAR
    hand_y_min: float = 1.0
    hand_curl: float = 0.0
    hand_speed: float = 0.0
    wrists_visible: float = 0.0
    wrist_y_mean: float = 1.0
    phone_conf: float = 0.0
    phone_dist_face: float = FAR
    phone_dist_hand: float = FAR
    phone_in_hand: float = 0.0
    phone_cy: float = 0.0
    phone_size: float = 0.0
    phone_aspect: float = 1.0        # box height / width in pixels; phones are oblong, mugs are square
    drink_conf: float = 0.0
    drink_dist_mouth: float = FAR
    drink_in_hand: float = 0.0
    food_conf: float = 0.0
    food_dist_mouth: float = FAR
    book_conf: float = 0.0
    motion: float = 0.0
    posture_dev: float = 0.0
    pitch_dev: float = 0.0
    hour_sin: float = 0.0
    hour_cos: float = 1.0
    # --- added after the first bench pass ---
    wrist_eye_dist: float = FAR      # pose wrist to pose eye, in face-heights (works when fingers are occluded)
    wrist_mouth_dist: float = FAR
    wrist_ear_dist: float = FAR
    wrist_face_dist: float = FAR
    face_touch: float = 0.0          # best evidence that a hand is on the face, 0..1
    drink_recent: float = RECENT_CAP  # seconds since a cup/bottle was seen near hand or mouth
    food_recent: float = RECENT_CAP
    phone_recent: float = RECENT_CAP
    chew: float = 0.0                # jaw_open variability over the last 3 s
    jaw_max: float = 0.0             # largest jaw_open in the last 3 s (a yawn is not chewing)
    mouth_visits: float = 0.0        # hand-to-mouth visits in the last 60 s
    face_from_crop: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float32)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "FrameFeatures":
        return cls(**{f.name: float(x) for f, x in zip(fields(cls), v)})

    def to_dict(self) -> dict[str, float]:
        return {f.name: round(float(getattr(self, f.name)), 4) for f in fields(self)}


FEATURE_NAMES: list[str] = [f.name for f in fields(FrameFeatures)]
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class Calibration:
    """Neutral-posture baseline recorded while sitting upright."""
    nose_shoulder: float = 0.0
    face_size: float = 0.0
    pitch: float = 0.0
    nose_y: float = 0.0
    shoulder_y: float = 0.0
    torso_lean: float = 0.0
    nose_shoulder_mad: float = 0.05
    face_size_mad: float = 0.02
    pitch_mad: float = 4.0
    nose_y_mad: float = 0.02
    torso_lean_mad: float = 0.05
    n_samples: int = 0
    recorded_at: float = 0.0
    auto: bool = False   # derived from the first minutes of a session, not an explicit calibration

    @property
    def valid(self) -> bool:
        return self.n_samples >= 20 and self.face_size > 0

    @classmethod
    def from_samples(cls, samples: list["FrameFeatures"]) -> "Calibration":
        good = [s for s in samples if s.present > 0 and s.face_size > 0]
        if len(good) < 20:
            return cls(n_samples=len(good))

        def med_mad(vals: list[float], floor: float) -> tuple[float, float]:
            a = np.asarray(vals, dtype=np.float64)
            m = float(np.median(a))
            mad = float(np.median(np.abs(a - m))) * 1.4826
            return m, max(mad, floor)

        ns, ns_mad = med_mad([s.nose_shoulder for s in good], 0.03)
        fs, fs_mad = med_mad([s.face_size for s in good], 0.01)
        pi, pi_mad = med_mad([s.pitch for s in good], 3.0)
        ny, ny_mad = med_mad([s.nose_y for s in good], 0.015)
        sy, _ = med_mad([s.shoulder_y for s in good], 0.02)
        tl, tl_mad = med_mad([s.torso_lean for s in good], 0.03)
        return cls(ns, fs, pi, ny, sy, tl, ns_mad, fs_mad, pi_mad, ny_mad, tl_mad, len(good), time.time())

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        c = cls()
        for f in fields(cls):
            if f.name in d:
                setattr(c, f.name, d[f.name])
        return c


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _bbox_dist(px: float, py: float, d: Detection) -> float:
    dx = max(d.x1 - px, 0.0, px - d.x2)
    dy = max(d.y1 - py, 0.0, py - d.y2)
    return math.hypot(dx, dy)


class FeatureExtractor:
    def __init__(self, calibration: Calibration | None = None):
        self.calibration = calibration or Calibration()
        self._prev_gray: np.ndarray | None = None
        self._prev_wrists: list[tuple[float, float]] = []
        self._prev_ts: float = 0.0
        self._last_detections: list[Detection] = []
        self._last_seen = {"drink": -1e9, "food": -1e9, "phone": -1e9}
        self._holding = {"drink": False, "food": False, "phone": False}
        self._last_hand_up = -1e9
        self._jaw_hist: list[tuple[float, float]] = []
        self._mouth_visits: list[float] = []
        self._hand_at_mouth = False

    def set_calibration(self, cal: Calibration) -> None:
        self.calibration = cal

    def motion_energy(self, bgr: np.ndarray) -> float:
        small = cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        m = float(np.mean(np.abs(gray - self._prev_gray)) / 255.0)
        self._prev_gray = gray
        return m

    def extract(self, bgr: np.ndarray, p: Perception, detections: list[Detection] | None, ts: float) -> FrameFeatures:
        f = FrameFeatures(ts=ts)
        if detections is not None:
            self._last_detections = detections
        dets = self._last_detections
        aspect = p.width / max(p.height, 1)  # to keep distances isotropic in normalized coords

        lt = time.localtime(ts)
        hour = lt.tm_hour + lt.tm_min / 60
        f.hour_sin = math.sin(2 * math.pi * hour / 24)
        f.hour_cos = math.cos(2 * math.pi * hour / 24)
        f.motion = self.motion_energy(bgr)
        f.present = 1.0 if p.present else 0.0

        # ---- face geometry -------------------------------------------------
        mouth = eyes_l = eyes_r = cheek_l = cheek_r = None
        face_bbox = None
        if p.face is not None:
            face = p.face.copy()
            face[:, 0] *= aspect  # x in "height units" so distances are isotropic
            f.face_conf = 1.0
            x1, y1 = face[:, 0].min(), face[:, 1].min()
            x2, y2 = face[:, 0].max(), face[:, 1].max()
            face_bbox = (x1, y1, x2, y2)
            f.face_size = float(y2 - y1)
            f.face_cx = float((x1 + x2) / 2 / aspect)
            f.face_cy = float((y1 + y2) / 2)
            f.nose_y = float(face[NOSE_TIP, 1])
            pitch, yaw, roll = LandmarkEngine.head_pose(p)
            f.pitch, f.yaw, f.roll = pitch, yaw, roll
            mouth = (face[MOUTH_L] + face[MOUTH_R]) / 2
            eyes_l = (face[LEFT_EYE_TOP] + face[LEFT_EYE_BOT]) / 2
            eyes_r = (face[RIGHT_EYE_TOP] + face[RIGHT_EYE_BOT]) / 2
            cheek_l, cheek_r = face[LEFT_CHEEK], face[RIGHT_CHEEK]
            lip_gap = _dist(face[LIP_TOP], face[LIP_BOT])
            f.mouth_open = float(lip_gap / max(f.face_size, 1e-3))

            b = p.blendshapes
            g = lambda k: float(b.get(k, 0.0))  # noqa: E731
            f.eye_closed = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2
            f.squint = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
            f.gaze_h = ((g("eyeLookOutLeft") + g("eyeLookInRight")) - (g("eyeLookInLeft") + g("eyeLookOutRight"))) / 2
            f.gaze_v = ((g("eyeLookUpLeft") + g("eyeLookUpRight")) - (g("eyeLookDownLeft") + g("eyeLookDownRight"))) / 2
            f.jaw_open = g("jawOpen")
            f.mouth_pucker = g("mouthPucker")
            f.mouth_funnel = g("mouthFunnel")
            f.cheek_puff = g("cheekPuff")
            f.brow_down = (g("browDownLeft") + g("browDownRight")) / 2
            f.brow_inner_up = g("browInnerUp")

        # ---- body -----------------------------------------------------------
        if p.pose is not None:
            pose = p.pose.copy()
            pose[:, 0] *= aspect
            ls, rs = pose[P_LSHOULDER], pose[P_RSHOULDER]
            if ls[3] > 0.4 and rs[3] > 0.4:
                f.shoulder_y = float((ls[1] + rs[1]) / 2)
                f.shoulder_w = float(abs(ls[0] - rs[0]))
                f.shoulder_tilt = float((ls[1] - rs[1]) / max(f.shoulder_w, 1e-3))
                nose = pose[P_NOSE]
                if nose[3] > 0.4:
                    f.nose_shoulder = float((f.shoulder_y - nose[1]) / max(f.shoulder_w, 1e-3))
                    f.torso_lean = float(((ls[2] + rs[2]) / 2) - nose[2])
                    if p.face is None:
                        f.nose_y = float(nose[1])
            lw, rw = pose[P_LWRIST], pose[P_RWRIST]
            f.wrists_visible = float((lw[3] + rw[3]) / 2)
            vis = [w for w in (lw, rw) if w[3] > 0.4]
            if vis:
                f.wrist_y_mean = float(np.mean([w[1] for w in vis]))
            # wrist-to-head distances from the body model: robust when fingers are occluded
            head_pts = {k: pose[i] for k, i in (("le", P_LEYE), ("re", P_REYE), ("lm", P_MOUTH_L), ("rm", P_MOUTH_R),
                                                ("lear", P_LEAR), ("rear", P_REAR)) if pose[i, 3] > 0.4}
            fs_pose = f.face_size if f.face_size > 0 else (0.45 * f.shoulder_w if f.shoulder_w > 0 else 0.15)
            for wr in vis:
                if "le" in head_pts or "re" in head_pts:
                    d = min(_dist(wr, head_pts[k]) for k in ("le", "re") if k in head_pts) / fs_pose
                    f.wrist_eye_dist = min(f.wrist_eye_dist, d)
                if "lm" in head_pts or "rm" in head_pts:
                    d = min(_dist(wr, head_pts[k]) for k in ("lm", "rm") if k in head_pts) / fs_pose
                    f.wrist_mouth_dist = min(f.wrist_mouth_dist, d)
                if "lear" in head_pts or "rear" in head_pts:
                    d = min(_dist(wr, head_pts[k]) for k in ("lear", "rear") if k in head_pts) / fs_pose
                    f.wrist_ear_dist = min(f.wrist_ear_dist, d)
            if vis and head_pts:
                hx = [float(v[0]) for v in head_pts.values()]
                hy = [float(v[1]) for v in head_pts.values()]
                x1, x2 = min(hx) - 0.3 * fs_pose, max(hx) + 0.3 * fs_pose
                y1, y2 = min(hy) - 0.5 * fs_pose, max(hy) + 0.4 * fs_pose
                for wr in vis:
                    d = math.hypot(max(x1 - wr[0], 0, wr[0] - x2), max(y1 - wr[1], 0, wr[1] - y2)) / fs_pose
                    f.wrist_face_dist = min(f.wrist_face_dist, d)

        # ---- hands ----------------------------------------------------------
        f.n_hands = float(len(p.hands))
        fs = max(f.face_size, 0.12)  # fall back to a plausible face height if no face
        best_curl = 0.0
        best_face_d = FAR
        wrists_now: list[tuple[float, float]] = []
        for hand in p.hands:
            h = hand.copy()
            h[:, 0] *= aspect
            wrists_now.append((float(h[WRIST, 0]), float(h[WRIST, 1])))
            f.hand_y_min = min(f.hand_y_min, float(h[:, 1].min()))
            tips = h[FINGERTIPS]
            # finger curl: fingertip closer to wrist than the PIP joint means curled
            curls = []
            for tip_i, pip_i in zip(FINGERTIPS[1:], FINGER_PIPS[1:]):
                d_tip = _dist(h[tip_i], h[WRIST])
                d_pip = _dist(h[pip_i], h[WRIST])
                curls.append(1.0 if d_tip < d_pip * 1.05 else 0.0)
            curl = float(np.mean(curls)) if curls else 0.0
            if mouth is not None:
                dm = min(_dist(t, mouth) for t in tips) / fs
                de = min(min(_dist(t, eyes_l), _dist(t, eyes_r)) for t in tips) / fs
                dc = min(min(_dist(t, cheek_l), _dist(t, cheek_r)) for t in tips) / fs
                f.hand_mouth_dist = min(f.hand_mouth_dist, dm)
                f.hand_eye_dist = min(f.hand_eye_dist, de)
                f.hand_cheek_dist = min(f.hand_cheek_dist, dc)
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                dfb = min(math.hypot(max(x1 - t[0], 0, t[0] - x2), max(y1 - t[1], 0, t[1] - y2)) for t in tips) / fs
                if dfb < best_face_d:
                    best_face_d, best_curl = dfb, curl
        if p.hands:
            f.hand_face_dist = best_face_d
            f.hand_curl = best_curl
        # hand speed (face-heights per second), matched greedily to previous wrists
        if wrists_now and self._prev_wrists and self._prev_ts and ts > self._prev_ts:
            dt = ts - self._prev_ts
            speeds = []
            for w in wrists_now:
                d = min(math.hypot(w[0] - pw[0], w[1] - pw[1]) for pw in self._prev_wrists)
                speeds.append(d / fs / dt)
            f.hand_speed = float(min(np.mean(speeds), 20.0))
        if p.hands_ran:
            self._prev_wrists = wrists_now
            self._prev_ts = ts

        # ---- objects --------------------------------------------------------
        for d in dets:
            # detections are normalized to width/height; convert x to height units
            dx1, dx2 = d.x1 * aspect, d.x2 * aspect
            cx, cy = (dx1 + dx2) / 2, d.cy
            dd = Detection(d.name, d.conf, dx1, d.y1, dx2, d.y2)
            if d.name == "cell phone" and d.conf > f.phone_conf:
                f.phone_conf = d.conf
                f.phone_cy = cy
                f.phone_size = float(max(d.h, dd.w))
                f.phone_aspect = float((d.h / max(d.w, 1e-3)) / aspect)
                if face_bbox is not None:
                    fcx, fcy = (face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2
                    f.phone_dist_face = math.hypot(cx - fcx, cy - fcy) / fs
                if p.hands:
                    dh = FAR
                    in_hand = 0.0
                    for hand in p.hands:
                        h = hand.copy()
                        h[:, 0] *= aspect
                        for pt in h:
                            dh = min(dh, _bbox_dist(float(pt[0]), float(pt[1]), dd) / fs)
                        pad = 0.15 * fs
                        inside = sum(1 for pt in h if dd.contains(float(pt[0]), float(pt[1]), pad))
                        if inside >= 4:
                            in_hand = 1.0
                    f.phone_dist_hand = dh
                    f.phone_in_hand = in_hand
            elif d.name in DRINK and d.conf > f.drink_conf:
                f.drink_conf = d.conf
                if mouth is not None:
                    f.drink_dist_mouth = _bbox_dist(float(mouth[0]), float(mouth[1]), dd) / fs
                if p.hands:
                    f.drink_in_hand = 1.0 if any(
                        sum(1 for pt in (hh * np.array([aspect, 1, 1], np.float32)) if dd.contains(float(pt[0]), float(pt[1]), 0.1 * fs)) >= 3
                        for hh in p.hands) else 0.0
            elif d.name in FOOD and d.conf > f.food_conf:
                f.food_conf = d.conf
                if mouth is not None:
                    f.food_dist_mouth = _bbox_dist(float(mouth[0]), float(mouth[1]), dd) / fs
            elif d.name == "book":
                f.book_conf = max(f.book_conf, d.conf)

        # ---- memory of objects, hand-on-face evidence, chewing, hand-to-mouth rate ----
        # An object counts as "in play" only when it is at the hand or the face, never just on the desk.
        pursed = max(f.mouth_pucker, f.mouth_funnel)
        squarish = 0.75 < f.phone_aspect < 1.35
        mug_like = (f.phone_dist_face < 0.6 and pursed > 0.15) or (squarish and f.phone_conf < 0.6)
        seen = {
            "phone": f.phone_conf > 0.3 and not mug_like
                     and (f.phone_in_hand or f.phone_dist_hand < 0.5 or f.phone_dist_face < 1.2),
            "drink": f.drink_conf > 0.25 and (f.drink_in_hand or f.drink_dist_mouth < 1.2),
            "food": f.food_conf > 0.25 and f.food_dist_mouth < 1.5,
        }
        # A hand raised to the face keeps a previously seen object "held" until the hand drops:
        # the cup at the mouth is exactly when the detector stops seeing it.
        hand_up = f.wrist_face_dist < 1.2 or f.hand_face_dist < 1.0
        if hand_up:
            self._last_hand_up = ts
        sip_signature = (f.hand_mouth_dist < 0.5 or f.wrist_mouth_dist < 0.9) and pursed > 0.3 and f.pitch < -8
        if sip_signature and f.phone_conf <= 0:
            self._holding["phone"] = False
            self._last_seen["phone"] = -1e9
            self._last_seen["drink"] = ts  # the shape of a sip is evidence of a cup the detector cannot see
        # confidence needed to *start* holding something (a flicker should not start a hold)
        start_conf = {"phone": 0.4, "drink": 0.35, "food": 0.35}
        conf_now = {"phone": f.phone_conf, "drink": f.drink_conf, "food": f.food_conf}
        for k, now in seen.items():
            if now:
                self._last_seen[k] = ts
                if conf_now[k] >= start_conf[k]:
                    if not self._holding[k]:
                        # one thing in the hand at a time: a phone seen now replaces a remembered cup
                        for other in self._holding:
                            if other != k:
                                self._holding[other] = False
                    self._holding[k] = True
            elif self._holding[k]:
                if ts - self._last_hand_up <= 2.5:
                    self._last_seen[k] = ts
                else:
                    self._holding[k] = False
        f.phone_recent = float(min(RECENT_CAP, ts - self._last_seen["phone"]))
        f.drink_recent = float(min(RECENT_CAP, ts - self._last_seen["drink"]))
        f.food_recent = float(min(RECENT_CAP, ts - self._last_seen["food"]))
        hand_touch = max(0.0, min(1.0, (0.45 - f.hand_face_dist) / 0.3)) if p.hands else 0.0
        wrist_touch = max(0.0, min(1.0, (0.9 - f.wrist_face_dist) / 0.5))
        f.face_touch = float(max(hand_touch, 0.85 * wrist_touch))
        f.face_from_crop = 1.0 if p.face_from_crop else 0.0
        if f.face_conf > 0:
            self._jaw_hist.append((ts, f.jaw_open))
            self._jaw_hist = [(t, j) for (t, j) in self._jaw_hist if ts - t <= 3.0]
            if len(self._jaw_hist) >= 4:
                f.chew = float(np.std([j for _, j in self._jaw_hist]))
                f.jaw_max = float(max(j for _, j in self._jaw_hist))
        near = f.hand_mouth_dist < 0.5 or f.wrist_mouth_dist < 0.9
        far = f.hand_mouth_dist > 0.8 and f.wrist_mouth_dist > 1.3
        if near and not self._hand_at_mouth:
            self._hand_at_mouth = True
            self._mouth_visits.append(ts)
        elif far:
            self._hand_at_mouth = False
        self._mouth_visits = [t for t in self._mouth_visits if ts - t <= 60.0]
        f.mouth_visits = float(len(self._mouth_visits))

        # ---- posture deviation vs calibration -------------------------------
        cal = self.calibration
        if cal.valid and f.present and f.face_size > 0:
            z_ns = (cal.nose_shoulder - f.nose_shoulder) / cal.nose_shoulder_mad if f.shoulder_w > 0 else 0.0
            z_fs = (f.face_size - cal.face_size) / cal.face_size_mad
            z_ny = (f.nose_y - cal.nose_y) / cal.nose_y_mad
            z_pi = (cal.pitch - f.pitch) / cal.pitch_mad
            z_tl = (f.torso_lean - cal.torso_lean) / cal.torso_lean_mad if f.shoulder_w > 0 else 0.0
            # positive = worse. Head sinking toward shoulders, face growing (leaning in),
            # nose dropping in frame, chin tucking, torso leaning forward.
            f.posture_dev = float(0.35 * z_ns + 0.25 * z_fs + 0.2 * z_ny + 0.1 * z_pi + 0.1 * z_tl)
            f.pitch_dev = float(f.pitch - cal.pitch)
        return f
