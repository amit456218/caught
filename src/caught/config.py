"""Configuration, paths, and habit definitions."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    env = os.environ.get("CAUGHT_MODELS")
    return Path(env) if env else PROJECT_ROOT / "models"


MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "yolo11n.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
}


def ensure_models(log=print) -> Path:
    """Download the pretrained weights on first use (about 23 MB total)."""
    import urllib.request

    md = models_dir()
    md.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = md / name
        if dest.exists() and dest.stat().st_size > 100_000:
            continue
        log(f"downloading {name} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return md


def default_data_dir() -> Path:
    env = os.environ.get("CAUGHT_DATA")
    return Path(env) if env else Path.home() / ".caught"


# ---------------------------------------------------------------------------
# Habit registry. Every habit the system can catch, with display metadata and
# a daily budget (minutes or count) used for streaks and the weekly report.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HabitSpec:
    key: str
    label: str
    kind: str  # "duration" (minutes matter) or "count" (occurrences matter)
    color: str
    description: str
    daily_budget: float  # minutes for duration habits, count for count habits
    icon: str = ""


HABITS: dict[str, HabitSpec] = {h.key: h for h in [
    HabitSpec("phone", "Phone in hand", "duration", "#ef5350",
              "A phone is visible in your hand or at your face.", 30, "📱"),
    HabitSpec("looking_down", "Looking down (probably phone in lap)", "duration", "#ff7043",
              "Head pitched down for a sustained stretch with hands out of view.", 20, "👀"),
    HabitSpec("slouch", "Slouching", "duration", "#ab47bc",
              "Posture drifted well below your calibrated baseline.", 60, "🪑"),
    HabitSpec("hand_to_face", "Hand at face", "duration", "#ffa726",
              "Fingers at your mouth or cheek for more than a few seconds.", 10, "🤚"),
    HabitSpec("eye_rub", "Eye rubbing", "count", "#26c6da",
              "Hand at the eyes while they are closed.", 5, "👁️"),
    HabitSpec("looking_away", "Looking away", "duration", "#8d6e63",
              "Head turned well away from the screen for a sustained stretch.", 45, "↩️"),
    HabitSpec("drink", "Drinks", "count", "#42a5f5",
              "A cup or bottle raised to your mouth.", 8, "☕"),
    HabitSpec("eat", "Eating at desk", "duration", "#9ccc65",
              "Food at your mouth, or repeated hand-to-mouth with food present.", 20, "🍕"),
    HabitSpec("yawn", "Yawns", "count", "#7e57c2",
              "Mouth wide open for over a second.", 6, "🥱"),
    HabitSpec("drowsy", "Eyes closed", "count", "#5c6bc0",
              "Both eyes closed for over two seconds while present.", 3, "😴"),
    HabitSpec("restless", "Restless", "duration", "#ec407a",
              "Sustained high motion with no task-like stillness.", 30, "🌀"),
    HabitSpec("absent", "Away from desk", "duration", "#78909c",
              "Nobody in front of the camera.", 240, "🚪"),
]}

HABIT_KEYS: list[str] = list(HABITS.keys())


@dataclass
class Config:
    data_dir: str = str(default_data_dir())
    camera_index: int = 0
    process_width: int = 640          # frames are downscaled to this width before inference
    fps_active: float = 8.0           # present and moving
    fps_idle: float = 3.0             # present but still
    fps_absent: float = 1.0           # nobody there
    object_every_n_frames: int = 3    # YOLO cadence while present
    feature_store_hz: float = 2.0     # how often feature vectors are persisted for training
    store_review_clips: bool = True   # keep tiny thumbnail strips for uncertain episodes
    thumbnail_width: int = 160
    review_uncertain_low: float = 0.35
    review_uncertain_high: float = 0.8
    review_random_rate: float = 0.08  # fraction of confident episodes also queued for QA
    retention_days: int = 21          # features + samples older than this are pruned
    model_blend: float = 0.5          # weight of learned model vs rules once trained
    server_host: str = "127.0.0.1"
    server_port: int = 8787
    preview_in_dashboard: bool = True
    notifications: bool = False       # it does not nag by default
    auto_train_hour: int = 3          # nightly retrain when running continuously
    roast_enabled: bool = False       # optional Claude-written weekly roast
    goals: dict[str, float] = field(default_factory=lambda: {k: h.daily_budget for k, h in HABITS.items()})
    detector_params: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def db_path(self) -> Path:
        return self.data_path / "caught.sqlite3"

    @property
    def samples_path(self) -> Path:
        return self.data_path / "samples"

    @property
    def model_path(self) -> Path:
        return self.data_path / "model.pt"

    @property
    def calibration_path(self) -> Path:
        return self.data_path / "calibration.json"

    @property
    def config_path(self) -> Path:
        return self.data_path / "config.json"

    def ensure_dirs(self) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.samples_path.mkdir(parents=True, exist_ok=True)
        (self.data_path / "reports").mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.ensure_dirs()
        self.config_path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> "Config":
        cfg = cls()
        if data_dir:
            cfg.data_dir = str(data_dir)
        path = cfg.config_path
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                for k, v in raw.items():
                    if k == "data_dir" and data_dir:
                        continue
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except (OSError, ValueError):
                pass
        for k, h in HABITS.items():
            cfg.goals.setdefault(k, h.daily_budget)
        return cfg

    def update(self, patch: dict) -> None:
        for k, v in patch.items():
            if k in ("data_dir",):
                continue
            if hasattr(self, k):
                setattr(self, k, v)
