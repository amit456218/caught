"""Hand-written detectors. Each returns an instantaneous score in [0, 1].

Distances are in face-heights (see features.py), angles in degrees.
Head pitch convention (verified on real footage): positive = chin toward chest (looking down).
"""
from __future__ import annotations

from ..features import FAR, FrameFeatures
from .base import Detector, Temporal, ramp


class AbsentDetector(Detector):
    habit = "absent"
    temporal = Temporal(ema_tau=0.5, on_threshold=0.6, off_threshold=0.4, min_on=3.0, min_off=2.0,
                        min_duration=20.0, merge_gap=5.0, requires_present=False)

    def score(self, f: FrameFeatures) -> float:
        return 0.0 if f.present else 1.0


class PhoneDetector(Detector):
    """Phone seen in hand or at the face, or a phone seen moments ago plus a hand held at the ear."""
    habit = "phone"
    temporal = Temporal(ema_tau=1.0, on_threshold=0.35, off_threshold=0.15, min_on=1.5, min_off=4.0,
                        min_duration=5.0, merge_gap=15.0)
    params = {"conf_lo": 0.2, "conf_hi": 0.5, "hand_dist": 0.35, "face_dist": 1.2,
              "ear_dist": 0.55, "recent_s": 12.0, "ear_weight": 0.7}

    def score(self, f: FrameFeatures) -> float:
        p = self.params
        s = 0.0
        if f.phone_conf > 0:
            conf = ramp(f.phone_conf, p["conf_lo"], p["conf_hi"])
            in_hand = max(f.phone_in_hand, ramp(p["hand_dist"] - f.phone_dist_hand, 0.0, p["hand_dist"]))
            near_face = ramp(p["face_dist"] - f.phone_dist_face, 0.0, p["face_dist"] * 0.6)
            # a phone lying flat on the desk far from any hand or face is not a pickup
            s = conf * max(in_hand, 0.9 * near_face, 0.15)
            if f.phone_dist_face < 0.6 and max(f.mouth_pucker, f.mouth_funnel) > 0.15:
                s *= 0.3  # pressed to pursed lips: that is a cup the detector mislabeled
            if 0.75 < f.phone_aspect < 1.35 and f.phone_conf < 0.6:
                s *= 0.5  # a square low-confidence "phone" is usually a mug or a wallet
        # phone call: the detector rarely sees a phone pressed to the ear, but the body model sees the wrist there
        ear = ramp(p["ear_dist"] - f.wrist_ear_dist, 0.0, p["ear_dist"] * 0.6)
        if ear > 0 and f.present:
            memory = 1.0 if f.phone_recent < p["recent_s"] else 0.75
            s = max(s, p["ear_weight"] * ear * memory)
        return min(1.0, s)


class LookingDownDetector(Detector):
    habit = "looking_down"
    temporal = Temporal(ema_tau=1.5, on_threshold=0.5, off_threshold=0.3, min_on=5.0, min_off=3.0,
                        min_duration=15.0, merge_gap=10.0)
    params = {"pitch_lo": 14.0, "pitch_hi": 30.0, "hands_hidden_weight": 0.65, "book_penalty": 0.7}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0:
            return 0.0
        p = self.params
        pitch = f.pitch_dev if f.pitch_dev != 0.0 else f.pitch
        down = ramp(pitch, p["pitch_lo"], p["pitch_hi"])
        if down <= 0:
            return 0.0
        hands_hidden = 1.0 if (f.n_hands == 0 and f.wrists_visible < 0.5) else (1.0 if f.hand_y_min > 0.85 else 0.0)
        gaze_down = ramp(-f.gaze_v, 0.1, 0.5)
        s = down * ((1 - p["hands_hidden_weight"]) + p["hands_hidden_weight"] * max(hands_hidden, 0.5 * gaze_down))
        s *= 1.0 - p["book_penalty"] * ramp(f.book_conf, 0.3, 0.6)
        if f.phone_conf > 0.4:
            s *= 0.5  # the phone detector owns this one
        # a hand over the face wrecks the head-pose estimate; do not trust it
        s *= 1.0 - f.face_touch
        return s


class SlouchDetector(Detector):
    habit = "slouch"
    temporal = Temporal(ema_tau=3.0, on_threshold=0.5, off_threshold=0.3, min_on=8.0, min_off=6.0,
                        min_duration=30.0, merge_gap=20.0)
    params = {"z_lo": 2.0, "z_hi": 5.0}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0 or f.posture_dev == 0.0:
            return 0.0
        if f.face_touch > 0.5:
            return 0.0  # leaning on a hand changes the geometry; handled by hand_to_face
        return ramp(f.posture_dev, self.params["z_lo"], self.params["z_hi"])


class HandToFaceDetector(Detector):
    habit = "hand_to_face"
    temporal = Temporal(ema_tau=0.8, on_threshold=0.5, off_threshold=0.3, min_on=2.0, min_off=2.0,
                        min_duration=3.0, merge_gap=5.0)
    params = {"mouth_dist": 0.35, "cheek_dist": 0.3, "curl_bonus": 0.25, "wrist_weight": 0.7,
              "object_recent_s": 4.0}

    def score(self, f: FrameFeatures) -> float:
        if not f.present:
            return 0.0
        p = self.params
        s = 0.0
        if f.n_hands > 0 and f.face_conf > 0:
            mouth = ramp(p["mouth_dist"] - f.hand_mouth_dist, 0.0, p["mouth_dist"] * 0.8)
            cheek = 0.85 * ramp(p["cheek_dist"] - f.hand_cheek_dist, 0.0, p["cheek_dist"] * 0.8)
            on_face = 0.8 * ramp(0.2 - f.hand_face_dist, 0.0, 0.15)  # fingers inside the face box (forehead, eyes)
            s = max(mouth, cheek, on_face)
            if s > 0:
                s = min(1.0, s + p["curl_bonus"] * f.hand_curl * mouth)
        # fingers occluded (rubbing the face, head in hands): fall back to the wrist from the body model
        wrist = ramp(0.75 - f.wrist_face_dist, 0.0, 0.45)
        s = max(s, p["wrist_weight"] * wrist)
        if s <= 0:
            return 0.0
        # holding a cup, food, or a phone at the face is a different habit
        if f.drink_recent < p["object_recent_s"] or f.food_recent < p["object_recent_s"]:
            s *= 0.15
        if f.phone_dist_face < 0.9 or (f.phone_recent < p["object_recent_s"] and f.wrist_ear_dist < 0.6):
            s *= 0.3
        if f.jaw_open > 0.45:
            s *= 0.4  # hand over a yawn
        return s


class EyeRubDetector(Detector):
    habit = "eye_rub"
    temporal = Temporal(ema_tau=0.4, on_threshold=0.5, off_threshold=0.25, min_on=0.7, min_off=1.5,
                        min_duration=1.0, merge_gap=3.0)
    params = {"eye_dist": 0.3, "closed_lo": 0.2, "closed_hi": 0.6, "wrist_eye_dist": 0.6}

    def score(self, f: FrameFeatures) -> float:
        if not f.present:
            return 0.0
        p = self.params
        near = 0.0
        if f.n_hands > 0 and f.face_conf > 0:
            near = ramp(p["eye_dist"] - f.hand_eye_dist, 0.0, p["eye_dist"] * 0.8)
        # hands covering the eyes hide the fingers from the hand model; the wrist is still visible
        wrist_near = ramp(p["wrist_eye_dist"] - f.wrist_eye_dist, 0.0, p["wrist_eye_dist"] * 0.6)
        near = max(near, 0.9 * wrist_near)
        if near <= 0:
            return 0.0
        closed = ramp(f.eye_closed, p["closed_lo"], p["closed_hi"])
        occluded = 1.0 if f.face_conf <= 0 else 0.0  # face lost behind the hands
        return near * max(closed, occluded * 0.8, 0.4 * near)


class LookingAwayDetector(Detector):
    habit = "looking_away"
    temporal = Temporal(ema_tau=1.5, on_threshold=0.5, off_threshold=0.3, min_on=6.0, min_off=3.0,
                        min_duration=10.0, merge_gap=8.0)
    params = {"yaw_lo": 25.0, "yaw_hi": 45.0, "gaze_lo": 0.35, "gaze_hi": 0.7}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0:
            return 0.0
        p = self.params
        yaw = ramp(abs(f.yaw), p["yaw_lo"], p["yaw_hi"])
        gaze = ramp(abs(f.gaze_h), p["gaze_lo"], p["gaze_hi"]) * ramp(abs(f.yaw), 10.0, 25.0)
        return max(yaw, 0.8 * gaze)


class DrinkDetector(Detector):
    """A cup or bottle near the mouth; or one seen seconds ago and the hand is still at the mouth;
    or, with no object visible at all, a hand at the mouth with pursed lips and the head tipped back."""
    habit = "drink"
    temporal = Temporal(ema_tau=0.5, on_threshold=0.45, off_threshold=0.2, min_on=0.6, min_off=2.0,
                        min_duration=0.8, merge_gap=8.0)
    params = {"conf_lo": 0.25, "conf_hi": 0.55, "mouth_dist": 0.8, "recent_s": 3.0, "sip_weight": 0.8}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0:
            return 0.0
        p = self.params
        s = 0.0
        phone_in_play = f.phone_recent < 5
        if f.drink_conf > 0:
            # with a phone in the hand, a low-confidence "bottle" is that phone
            conf = ramp(f.drink_conf, 0.45, 0.7) if phone_in_play else ramp(f.drink_conf, p["conf_lo"], p["conf_hi"])
            near = ramp(p["mouth_dist"] - f.drink_dist_mouth, 0.0, p["mouth_dist"] * 0.6)
            lifted = 1.0 if f.drink_in_hand else 0.7
            s = conf * near * lifted
        hand_at_mouth = max(ramp(0.5 - f.hand_mouth_dist, 0.0, 0.3), 0.8 * ramp(0.9 - f.wrist_mouth_dist, 0.0, 0.5))
        if f.drink_recent < p["recent_s"] and hand_at_mouth > 0 and not phone_in_play:
            s = max(s, 0.85 * hand_at_mouth)  # the cup is at the mouth and occluded
        elif hand_at_mouth > 0 and not phone_in_play and f.food_recent > 5:
            pursed = ramp(max(f.mouth_pucker, f.mouth_funnel), 0.15, 0.4)
            tipped = ramp(-(f.pitch_dev if f.pitch_dev != 0.0 else f.pitch), 2.0, 12.0)
            # pursed lips + head tipped back + hand at the mouth is the shape of a sip even with no cup seen
            sip = hand_at_mouth * min(1.0, max(pursed, 0.7 * tipped) + 0.4 * pursed * tipped)
            s = max(s, p["sip_weight"] * sip)
        return min(1.0, s)


class EatDetector(Detector):
    """Food or cutlery near the mouth, or repeated hand-to-mouth visits with chewing."""
    habit = "eat"
    temporal = Temporal(ema_tau=2.0, on_threshold=0.45, off_threshold=0.2, min_on=2.0, min_off=10.0,
                        min_duration=10.0, merge_gap=45.0)
    params = {"conf_lo": 0.25, "conf_hi": 0.55, "mouth_dist": 0.9, "hand_mouth": 0.9,
              "chew_lo": 0.02, "chew_hi": 0.06, "visits_lo": 0.5, "visits_hi": 2.5, "chew_weight": 0.8}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0:
            return 0.0
        p = self.params
        if f.drink_recent < 5 or f.phone_recent < 5:
            return 0.0
        # what is being eaten: a recognised food item near the mouth or hand (not the apple on the desk),
        # or the jaw working in small movements (chewing, not a yawn)
        food_now = ramp(f.food_conf, p["conf_lo"], p["conf_hi"]) * ramp(1.6 - f.food_dist_mouth, 0.0, 0.8)
        food_ev = max(food_now, 0.6 if f.food_recent < 20 else 0.0)
        small_jaw = ramp(0.45 - f.jaw_max, 0.0, 0.15)
        chew_ev = p["chew_weight"] * ramp(f.chew, p["chew_lo"], p["chew_hi"]) * small_jaw
        # how it gets there: hand at the mouth now, or repeated visits in the last minute
        hand_now = max(ramp(p["hand_mouth"] - f.hand_mouth_dist, 0.0, p["hand_mouth"] * 0.55),
                       0.8 * ramp(1.2 - f.wrist_mouth_dist, 0.0, 0.6))
        visits = ramp(f.mouth_visits, p["visits_lo"], p["visits_hi"])
        near_food = ramp(p["mouth_dist"] - f.food_dist_mouth, 0.0, p["mouth_dist"] * 0.6) if f.food_conf > 0 else 0.0
        # a hand merely resting at the mouth is not eating; chewing needs the hand to come and go
        s = max(food_ev * max(hand_now, visits, near_food), chew_ev * max(visits, 0.5 * hand_now))
        return min(1.0, s)


class YawnDetector(Detector):
    habit = "yawn"
    temporal = Temporal(ema_tau=0.4, on_threshold=0.5, off_threshold=0.3, min_on=1.2, min_off=0.8,
                        min_duration=1.2, merge_gap=4.0)
    params = {"jaw_lo": 0.4, "jaw_hi": 0.7, "mouth_open_min": 0.1}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0:
            return 0.0
        p = self.params
        if f.mouth_open < p["mouth_open_min"] and f.jaw_open < 0.6:
            return 0.0
        s = ramp(f.jaw_open, p["jaw_lo"], p["jaw_hi"])
        if f.drink_dist_mouth < 0.6 or f.food_dist_mouth < 0.6:
            s *= 0.2
        return s


class DrowsyDetector(Detector):
    """Both eyes closed for a couple of seconds, while not looking down, yawning, or rubbing them.
    The eye-closure blendshape rises when the gaze drops (reading, eating), so the gaze gate matters."""
    habit = "drowsy"
    temporal = Temporal(ema_tau=0.5, on_threshold=0.5, off_threshold=0.3, min_on=2.5, min_off=1.0,
                        min_duration=2.5, merge_gap=5.0)
    params = {"closed_lo": 0.6, "closed_hi": 0.85, "max_pitch": 18.0, "gaze_gate": -0.3}

    def score(self, f: FrameFeatures) -> float:
        if not f.present or f.face_conf <= 0:
            return 0.0
        p = self.params
        pitch = f.pitch_dev if f.pitch_dev != 0.0 else f.pitch
        if pitch > p["max_pitch"] or f.gaze_v < p["gaze_gate"]:
            return 0.0  # looking down reads as closed eyes to the blendshape model
        if f.hand_eye_dist < 0.35 or f.wrist_eye_dist < 0.6 or f.face_touch > 0.5:
            return 0.0  # that is an eye rub or a head in hands
        if f.jaw_open > 0.3 or f.mouth_open > 0.12:
            return 0.0  # yawning
        return ramp(f.eye_closed, p["closed_lo"], p["closed_hi"])


class RestlessDetector(Detector):
    habit = "restless"
    temporal = Temporal(ema_tau=5.0, on_threshold=0.5, off_threshold=0.3, min_on=20.0, min_off=15.0,
                        min_duration=30.0, merge_gap=30.0)
    params = {"motion_lo": 0.02, "motion_hi": 0.06, "speed_lo": 2.0, "speed_hi": 5.0}

    def score(self, f: FrameFeatures) -> float:
        if not f.present:
            return 0.0
        p = self.params
        return max(ramp(f.motion, p["motion_lo"], p["motion_hi"]), 0.8 * ramp(f.hand_speed, p["speed_lo"], p["speed_hi"]))


ALL_DETECTORS: list[type[Detector]] = [
    AbsentDetector, PhoneDetector, LookingDownDetector, SlouchDetector, HandToFaceDetector,
    EyeRubDetector, LookingAwayDetector, DrinkDetector, EatDetector, YawnDetector,
    DrowsyDetector, RestlessDetector,
]
