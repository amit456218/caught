"""Face, hand, and body landmarks via MediaPipe Tasks (VIDEO mode)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import ensure_models, models_dir

# Face mesh indices we care about
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
LEFT_EYE_TOP, LEFT_EYE_BOT = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOT = 386, 374
MOUTH_L, MOUTH_R = 61, 291
LIP_TOP, LIP_BOT = 13, 14
LEFT_CHEEK, RIGHT_CHEEK = 234, 454

# Hand indices
WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

# Pose indices
P_NOSE, P_LEYE, P_REYE, P_LEAR, P_REAR = 0, 2, 5, 7, 8
P_LSHOULDER, P_RSHOULDER = 11, 12
P_LELBOW, P_RELBOW = 13, 14
P_LWRIST, P_RWRIST = 15, 16
P_LHIP, P_RHIP = 23, 24

BLENDSHAPES = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "jawOpen", "mouthClose", "mouthPucker", "mouthFunnel", "mouthRollLower",
    "cheekPuff", "browDownLeft", "browDownRight", "browInnerUp",
]


@dataclass
class Perception:
    ts_ms: int
    width: int
    height: int
    face: np.ndarray | None = None          # (478, 3) normalized xyz
    face_matrix: np.ndarray | None = None   # (4, 4)
    blendshapes: dict[str, float] = field(default_factory=dict)
    hands: list[np.ndarray] = field(default_factory=list)   # each (21, 3) normalized
    handedness: list[str] = field(default_factory=list)
    pose: np.ndarray | None = None          # (33, 4) normalized xyz + visibility
    hands_ran: bool = False
    face_from_crop: bool = False            # face came from the upscaled head crop fallback

    @property
    def present(self) -> bool:
        return self.face is not None or (self.pose is not None and float(self.pose[P_NOSE, 3]) > 0.5)


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees from a 4x4 transform. Pitch>0 = looking up."""
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class LandmarkEngine:
    """Owns the three MediaPipe landmarkers. Not thread-safe; call from one thread."""

    def __init__(self, run_hands: bool = True, run_pose: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        md = ensure_models(log=lambda *_: None)
        base = mpp.BaseOptions
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(md / "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(md / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)) if run_hands else None
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(md / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)) if run_pose else None
        # second face landmarker in IMAGE mode, used on an upscaled head crop when the person
        # sits far from the camera and the full-frame pass finds nothing
        self.face_crop = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(md / "face_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4))
        self._last_ms = -1
        self.crop_size = 320

    def process(self, bgr: np.ndarray, ts_ms: int, want_hands: bool = True) -> Perception:
        import cv2

        if ts_ms <= self._last_ms:
            ts_ms = self._last_ms + 1
        self._last_ms = ts_ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        h, w = bgr.shape[:2]
        out = Perception(ts_ms=ts_ms, width=w, height=h)

        fr = self.face.detect_for_video(img, ts_ms)
        if fr.face_landmarks:
            lm = fr.face_landmarks[0]
            out.face = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            if fr.facial_transformation_matrixes:
                out.face_matrix = np.array(fr.facial_transformation_matrixes[0], dtype=np.float32)
            if fr.face_blendshapes:
                out.blendshapes = {c.category_name: float(c.score) for c in fr.face_blendshapes[0]
                                   if c.category_name in BLENDSHAPES}

        if self.pose is not None:
            pr = self.pose.detect_for_video(img, ts_ms)
            if pr.pose_landmarks:
                lm = pr.pose_landmarks[0]
                out.pose = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)

        if out.face is None and out.pose is not None:
            self._face_from_head_crop(rgb, out)

        if self.hand is not None and want_hands:
            hr = self.hand.detect_for_video(img, ts_ms)
            out.hands_ran = True
            for i, lm in enumerate(hr.hand_landmarks or []):
                out.hands.append(np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32))
                side = hr.handedness[i][0].category_name if hr.handedness and hr.handedness[i] else "?"
                out.handedness.append(side)
        return out

    def _face_from_head_crop(self, rgb: np.ndarray, out: Perception) -> None:
        """Run the face landmarker on an upscaled crop around the pose head landmarks."""
        import cv2

        pose = out.pose
        assert pose is not None
        h, w = rgb.shape[:2]
        nose = pose[P_NOSE]
        if nose[3] < 0.5:
            return
        pts = [pose[i] for i in (P_NOSE, P_LEYE, P_REYE, P_LEAR, P_REAR) if pose[i, 3] > 0.4]
        if len(pts) < 3:
            return
        xs = [p[0] * w for p in pts]
        ys = [p[1] * h for p in pts]
        span = max(max(xs) - min(xs), 8.0)
        side = int(min(max(span * 3.2, 48), 0.9 * min(h, w)))
        if side > 0.55 * h:
            return  # the face is big; the miss is not a resolution problem
        cx, cy = nose[0] * w, nose[1] * h - span * 0.35
        x0 = int(max(0, min(w - side, cx - side / 2)))
        y0 = int(max(0, min(h - side, cy - side / 2)))
        crop = rgb[y0:y0 + side, x0:x0 + side]
        if crop.size == 0:
            return
        crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        res = self.face_crop.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        if not res.face_landmarks:
            return
        lm = res.face_landmarks[0]
        face = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        face[:, 0] = (x0 + face[:, 0] * side) / w
        face[:, 1] = (y0 + face[:, 1] * side) / h
        face[:, 2] = face[:, 2] * side / w
        out.face = face
        out.face_from_crop = True
        if res.facial_transformation_matrixes:
            out.face_matrix = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
        if res.face_blendshapes:
            out.blendshapes = {c.category_name: float(c.score) for c in res.face_blendshapes[0]
                               if c.category_name in BLENDSHAPES}

    @staticmethod
    def head_pose(p: Perception) -> tuple[float, float, float]:
        if p.face_matrix is None:
            return 0.0, 0.0, 0.0
        return _euler_from_matrix(p.face_matrix)

    def close(self) -> None:
        for lm in (self.face, self.hand, self.pose, self.face_crop):
            if lm is not None:
                lm.close()
