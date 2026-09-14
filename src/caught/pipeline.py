"""The always-on loop: frames -> landmarks/objects -> features -> scores -> episodes -> store.

Runs on its own thread. Shares a status dict and an optional preview JPEG with the
server. Raw frames never leave memory; only tiny thumbnails of flagged episodes are
written to disk, and only when review clips are enabled.
"""
from __future__ import annotations

import random
import threading
import time
import traceback
from datetime import date

import cv2
import numpy as np

from .calibration import CalibrationRecorder, load_calibration, save_calibration
from .capture import FrameSource
from .config import HABITS, Config
from .detectors import build_detectors
from .episodes import Episode, EpisodeTracker
from .features import FeatureExtractor, FrameFeatures
from .model import ModelScorer, should_retrain, should_review
from .notify import notify
from .perception.landmarks import P_LWRIST, P_RWRIST, LandmarkEngine, Perception
from .perception.objects import Detection, ObjectDetector
from .power import PowerPolicy
from .store import Store

THUMB_SCHEDULE = [0.0, 3.0, 8.0, 15.0, 30.0, 60.0]  # seconds into an episode


class Pipeline:
    def __init__(self, cfg: Config, source: FrameSource, store: Store, use_objects: bool = True,
                 use_model: bool = True, preview: bool = True, log=print):
        self.cfg = cfg
        self.source = source
        self.store = store
        self.log = log
        self.use_objects = use_objects
        self.preview_enabled = preview
        self.detectors = build_detectors(cfg.detector_params)
        self.tracker = EpisodeTracker({k: d.temporal for k, d in self.detectors.items()})
        self.extractor = FeatureExtractor(load_calibration(cfg.calibration_path))
        self.scorer = ModelScorer(cfg.model_path) if use_model else None
        self.power = PowerPolicy(cfg)
        self.calibrator = CalibrationRecorder()
        self._auto_samples: list[FrameFeatures] = []
        self._auto_started = 0.0
        self.rng = random.Random()
        self._engine: LandmarkEngine | None = None
        self._objects: ObjectDetector | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._preview_jpeg: bytes | None = None
        self._preview_ts = 0.0
        self._last_store_ts = 0.0
        self._frame_i = 0
        self._last_detections: list[Detection] = []
        self._fps_hist: list[float] = []
        self._latency_ms = 0.0
        self._last_prune_day: date | None = None
        self._last_train_day: date | None = None
        self._training = False
        self._session_check_ts = 0.0
        self._session_notified: int | None = None
        self.finished = False
        self.error: str | None = None
        self.started_at = 0.0
        self.status: dict = {"running": False}
        self.today_cache: dict = {"date": None, "totals": {}}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def pause(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start_calibration(self, seconds: float = 10.0) -> None:
        self.calibrator = CalibrationRecorder(seconds)
        self.calibrator.start(time.time())

    def preview_jpeg(self) -> bytes | None:
        with self._lock:
            return self._preview_jpeg

    def reload_model(self) -> bool:
        if self.scorer is None:
            return False
        ok = self.scorer.reload()
        if ok:
            self.scorer.reset()
        return ok

    # ------------------------------------------------------------------ main loop
    def _run(self) -> None:
        try:
            self.log("pipeline: loading models")
            self._engine = LandmarkEngine()
            if self.use_objects:
                self._objects = ObjectDetector()
            self.source.start()
            self.started_at = time.time()
            self.store.log_event("start", {"camera": getattr(self.source, "index", None)})
            self.log("pipeline: running")
            self.status["running"] = True
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.2)
                    self.status["mode"] = "paused"
                    continue
                frame = self.source.read()
                if frame is None:
                    if getattr(self.source, "path", None):  # video replay finished
                        break
                    time.sleep(0.01)
                    continue
                t0 = time.monotonic()
                self._step(frame.image, frame.ts, frame.ts_ms)
                self._latency_ms = (time.monotonic() - t0) * 1000
                self._fps_hist.append(frame.ts)
                self._fps_hist = [t for t in self._fps_hist if frame.ts - t < 5.0]
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.log("pipeline error:\n" + traceback.format_exc())
        finally:
            ts = time.time()
            for kind, ep in self.tracker.flush(ts):
                self._handle_event(kind, ep, None)
            self.store.flush()
            self.store.log_event("stop")
            try:
                self.source.stop()
            except Exception:  # noqa: BLE001
                pass
            if self._engine:
                self._engine.close()
            self.status["running"] = False
            self.finished = True
            self.log("pipeline: stopped")

    def _step(self, img: np.ndarray, ts: float, ts_ms: int) -> None:
        assert self._engine is not None
        self._frame_i += 1
        # hands are the most expensive landmarker: run when the body says a wrist is raised,
        # or periodically so we never miss a hand entering from below
        want_hands = (self._frame_i % 4 == 0)
        p_prev = getattr(self, "_p_prev", None)
        if p_prev is not None and p_prev.pose is not None:
            for wi in (P_LWRIST, P_RWRIST):
                if p_prev.pose[wi, 3] > 0.5 and p_prev.pose[wi, 1] < 0.95:
                    want_hands = True
        if p_prev is not None and p_prev.hands:
            want_hands = True
        p = self._engine.process(img, ts_ms, want_hands=want_hands)
        self._p_prev = p

        mode, fps, obj_every = self.power.update(p.present, self.extractor.motion_energy(img) if False else 0.0,
                                                 any(s.active for s in self.tracker.states.values()))
        dets: list[Detection] | None = None
        if self._objects is not None and p.present and self._frame_i % obj_every == 0:
            dets = self._objects.detect(img)
            self._last_detections = dets
        elif not p.present:
            self._last_detections = []
            dets = []
        f = self.extractor.extract(img, p, dets, ts)
        mode, fps, _ = self.power.update(p.present, f.motion, any(s.active for s in self.tracker.states.values()))
        self.source.set_target_fps(fps)

        if self.calibrator.active:
            cal = self.calibrator.feed(f)
            if cal is not None:
                if cal.valid:
                    self.extractor.set_calibration(cal)
                    save_calibration(self.cfg.calibration_path, cal)
                    self.store.log_event("calibrate", cal.to_dict())
                    self.log(f"calibrated posture baseline from {cal.n_samples} frames")
                else:
                    self.log("calibration failed: face not visible enough")

        self._maybe_auto_calibrate(f)
        rule_scores = {k: d.score(f) for k, d in self.detectors.items()}
        model_scores = self.scorer.push(ts, f.to_vector()) if self.scorer else None
        blend = self.cfg.model_blend if model_scores else 0.0
        events = self.tracker.update(f, rule_scores, model_scores, blend)
        for kind, ep in events:
            self._handle_event(kind, ep, img)
        self._collect_thumbnails(img, ts)

        # persistence at a fixed cadence (matches the training window cadence)
        if ts - self._last_store_ts >= 1.0 / self.cfg.feature_store_hz:
            self._last_store_ts = ts
            self.store.add_feature(ts, f.to_vector())
        self.store.add_presence(ts, p.present)

        self._update_status(f, p, rule_scores, model_scores, mode, fps)
        if self.preview_enabled and ts - self._preview_ts > 0.5:
            self._preview_ts = ts
            self._render_preview(img, p, f, rule_scores)
        self._housekeeping(ts)

    def _maybe_auto_calibrate(self, f: FrameFeatures) -> None:
        """No explicit calibration yet: take the first ~90 s of face-visible frames as the baseline.
        Marked auto so the dashboard can nudge the user to do a real one."""
        if self.extractor.calibration.valid or self.calibrator.active:
            return
        if not (f.present and f.face_conf > 0 and f.shoulder_w > 0):
            return
        if not self._auto_started:
            self._auto_started = f.ts
        self._auto_samples.append(f)
        if len(self._auto_samples) >= 200 and f.ts - self._auto_started >= 90:
            from .features import Calibration
            cal = Calibration.from_samples(self._auto_samples)
            self._auto_samples = []
            self._auto_started = 0.0
            if cal.valid:
                cal.auto = True
                self.extractor.set_calibration(cal)
                save_calibration(self.cfg.calibration_path, cal)
                self.store.log_event("auto_calibrate", cal.to_dict())
                self.log("posture baseline set automatically from the first 90 s; run a real calibration when you can")

    # ------------------------------------------------------------------ episodes
    def _handle_event(self, kind: str, ep: Episode, img: np.ndarray | None) -> None:
        if kind == "start":
            if img is not None:
                self._add_thumb(ep, img, ep.start)
            if self.cfg.notifications and ep.habit in ("phone", "slouch"):
                notify("Caught", f"{HABITS[ep.habit].label}")
        elif kind == "commit":
            review = should_review(ep, self.cfg, self.rng) if self.cfg.store_review_clips else False
            if not self.cfg.store_review_clips:
                ep.thumbnails, ep.thumb_times = [], []
            eid = self.store.commit_episode(ep, review)
            self.log(f"episode #{eid} {ep.habit} {ep.duration:.0f}s conf={ep.confidence:.2f}"
                     + (" [review]" if review else ""))
            self.today_cache["date"] = None  # invalidate
        elif kind == "discard":
            pass

    def _collect_thumbnails(self, img: np.ndarray, ts: float) -> None:
        if not self.cfg.store_review_clips:
            return
        for st in self.tracker.states.values():
            ep = st.current
            if ep is None or len(ep.thumbnails) >= len(THUMB_SCHEDULE):
                continue
            due = THUMB_SCHEDULE[len(ep.thumbnails)]
            if ts - ep.start >= due:
                self._add_thumb(ep, img, ts)

    def _add_thumb(self, ep: Episode, img: np.ndarray, ts: float) -> None:
        h, w = img.shape[:2]
        tw = self.cfg.thumbnail_width
        small = cv2.resize(img, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            ep.thumbnails.append(buf.tobytes())
            ep.thumb_times.append(ts)

    # ------------------------------------------------------------------ status / preview
    def _update_status(self, f: FrameFeatures, p: Perception, rules: dict[str, float],
                       model: dict[str, float] | None, mode: str, fps: float) -> None:
        now = time.time()
        actual_fps = len(self._fps_hist) / 5.0 if len(self._fps_hist) > 1 else 0.0
        snap = self.tracker.snapshot()
        self.status = {
            "running": True, "mode": mode, "target_fps": fps, "fps": round(actual_fps, 1),
            "latency_ms": round(self._latency_ms, 1), "present": bool(p.present),
            "face": p.face is not None, "hands": len(p.hands), "pose": p.pose is not None,
            "uptime_s": int(now - self.started_at) if self.started_at else 0,
            "habits": snap, "rule_scores": {k: round(v, 3) for k, v in rules.items()},
            "model_scores": {k: round(v, 3) for k, v in model.items()} if model else None,
            "model_ready": bool(self.scorer and self.scorer.ready),
            "model_info": self.scorer.info if self.scorer else {},
            "calibrated": self.extractor.calibration.valid,
            "calibration_auto": bool(self.extractor.calibration.valid and self.extractor.calibration.auto),
            "calibrating": self.calibrator.active,
            "calibration_progress": self.calibrator.progress(f.ts) if self.calibrator.active else None,
            "features": {k: f.to_dict()[k] for k in ("pitch", "yaw", "posture_dev", "pitch_dev", "hand_mouth_dist",
                                                     "phone_conf", "eye_closed", "jaw_open", "motion", "n_hands")},
            "objects": [{"name": d.name, "conf": round(d.conf, 2)} for d in self._last_detections],
            "training": self._training, "error": self.error, "paused": self.paused,
        }

    def _render_preview(self, img: np.ndarray, p: Perception, f: FrameFeatures, rules: dict[str, float]) -> None:
        out = img.copy()
        h, w = out.shape[:2]
        if p.face is not None:
            xs, ys = p.face[:, 0] * w, p.face[:, 1] * h
            cv2.rectangle(out, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (80, 200, 255), 1)
        for hand in p.hands:
            for x, y, _ in hand:
                cv2.circle(out, (int(x * w), int(y * h)), 2, (0, 255, 120), -1)
        if p.pose is not None:
            for i in (11, 12, 13, 14, 15, 16):
                if p.pose[i, 3] > 0.4:
                    cv2.circle(out, (int(p.pose[i, 0] * w), int(p.pose[i, 1] * h)), 4, (255, 180, 0), -1)
        for d in self._last_detections:
            cv2.rectangle(out, (int(d.x1 * w), int(d.y1 * h)), (int(d.x2 * w), int(d.y2 * h)), (60, 60, 255), 2)
            cv2.putText(out, f"{d.name} {d.conf:.2f}", (int(d.x1 * w), max(12, int(d.y1 * h) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255), 1)
        y = 18
        for k, st in self.tracker.states.items():
            if st.active or st.smoothed > 0.15:
                txt = f"{HABITS[k].label}: {st.smoothed:.2f}" + (" ACTIVE" if st.active else "")
                cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if st.active else (200, 200, 200), 1)
                y += 18
        cv2.putText(out, f"pitch {f.pitch:+.0f} yaw {f.yaw:+.0f} posture {f.posture_dev:+.1f}", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        if self.calibrator.active:
            pr = self.calibrator.progress(f.ts)
            cv2.rectangle(out, (0, 0), (int(w * pr), 6), (0, 200, 0), -1)
            cv2.putText(out, "CALIBRATING: sit upright, look at the screen", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._preview_jpeg = buf.tobytes()

    # ------------------------------------------------------------------ housekeeping
    def _housekeeping(self, ts: float) -> None:
        if ts - self._session_check_ts >= 5.0:
            self._session_check_ts = ts
            self._check_session(ts)
        today = date.fromtimestamp(ts)
        if self._last_prune_day != today:
            self._last_prune_day = today
            try:
                self.store.prune(self.cfg.retention_days)
            except Exception as e:  # noqa: BLE001
                self.log(f"prune failed: {e}")
        hour = time.localtime(ts).tm_hour
        if (hour == self.cfg.auto_train_hour and self._last_train_day != today and not self._training
                and self.scorer is not None):
            self._last_train_day = today
            ok, why = should_retrain(self.store)
            if ok:
                self.train_async(reason=f"nightly ({why})")

    def _check_session(self, ts: float) -> None:
        """One notification when a focus session's planned time is up. Always on: the user set the timer."""
        try:
            from .focus import Sessions
            sess = Sessions(self.store)
            cur = sess.current()
        except Exception:  # noqa: BLE001
            return
        if not cur:
            self._session_notified = None
            return
        if cur["id"] == self._session_notified:
            return
        if ts - cur["start"] >= cur["planned_minutes"] * 60:
            self._session_notified = cur["id"]
            live = sess.live() or {}
            sm = live.get("summary") or {}
            notify("Caught", f"{cur['planned_minutes']:.0f} min are up. {sm.get('phone_pickups', 0)} phone pickups, "
                             f"score so far {sm.get('score', '–')}. End the session in the dashboard.")

    def train_async(self, reason: str = "manual") -> bool:
        if self._training:
            return False
        self._training = True

        def _job() -> None:
            from .model.train import train
            try:
                self.log(f"training: {reason}")
                res = train(self.store, self.cfg.model_path, log=self.log)
                if res.get("ok"):
                    self.reload_model()
            except Exception as e:  # noqa: BLE001
                self.log(f"training failed: {e}")
            finally:
                self._training = False

        threading.Thread(target=_job, name="train", daemon=True).start()
        return True
