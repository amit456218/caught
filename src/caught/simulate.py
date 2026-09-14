"""Generate a realistic synthetic history so the dashboard, review queue, training, and
report can be exercised without a camera. Also used by the tests."""
from __future__ import annotations

import io
import math
import random
import time
from datetime import date, datetime, timedelta

import numpy as np

from .config import HABIT_KEYS, Config
from .episodes import Episode
from .features import FAR, FrameFeatures
from .store import Store

# (habit, episodes per desk-hour, duration seconds (lognormal median, sigma), evening multiplier)
PROFILE = {
    "phone": (1.6, 90.0, 0.9, 2.2),
    "looking_down": (0.5, 120.0, 0.6, 1.8),
    "slouch": (0.5, 400.0, 0.8, 1.6),
    "hand_to_face": (1.2, 12.0, 0.6, 1.2),
    "eye_rub": (0.35, 2.5, 0.3, 2.5),
    "looking_away": (0.6, 45.0, 0.7, 1.0),
    "drink": (0.8, 3.0, 0.3, 0.8),
    "eat": (0.08, 700.0, 0.5, 1.4),
    "yawn": (0.3, 3.0, 0.2, 3.0),
    "drowsy": (0.12, 4.0, 0.3, 4.0),
    "restless": (0.25, 120.0, 0.6, 1.5),
}


def _thumb(color: tuple[int, int, int], text: str) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (160, 90), color)
    ImageDraw.Draw(im).text((8, 36), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def _feature_for(ts: float, active: set[str], present: bool, rng: random.Random) -> FrameFeatures:
    f = FrameFeatures(ts=ts)
    lt = time.localtime(ts)
    hour = lt.tm_hour + lt.tm_min / 60
    f.hour_sin, f.hour_cos = math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)
    if not present:
        f.present = 0.0
        f.motion = rng.uniform(0, 0.004)
        return f
    f.present = f.face_conf = 1.0
    f.face_size = rng.gauss(0.3, 0.02)
    f.face_cx, f.face_cy = rng.gauss(0.5, 0.03), rng.gauss(0.45, 0.03)
    f.pitch, f.yaw, f.roll = rng.gauss(2, 4), rng.gauss(0, 5), rng.gauss(0, 3)
    f.nose_y = rng.gauss(0.42, 0.02)
    f.shoulder_y, f.shoulder_w = rng.gauss(0.8, 0.02), rng.gauss(0.5, 0.02)
    f.nose_shoulder = (f.shoulder_y - f.nose_y) / f.shoulder_w
    f.eye_closed = max(0.0, rng.gauss(0.08, 0.05))
    f.jaw_open = max(0.0, rng.gauss(0.05, 0.03))
    f.mouth_open = max(0.0, rng.gauss(0.03, 0.02))
    f.motion = max(0.0, rng.gauss(0.012, 0.006))
    f.n_hands = float(rng.random() < 0.4)
    f.wrists_visible = rng.uniform(0.3, 0.9)
    f.wrist_y_mean = rng.uniform(0.75, 0.98)
    f.posture_dev = rng.gauss(0.0, 0.8)
    f.pitch_dev = f.pitch - 2
    if "phone" in active:
        f.phone_conf = rng.uniform(0.5, 0.9)
        f.phone_in_hand = 1.0
        f.phone_dist_hand = rng.uniform(0, 0.2)
        f.phone_dist_face = rng.uniform(0.6, 1.5)
        f.n_hands = 1.0
        f.phone_cy, f.phone_size = rng.uniform(0.5, 0.8), rng.uniform(0.1, 0.2)
        f.pitch += rng.uniform(8, 20)
    if "looking_down" in active:
        f.pitch += rng.uniform(18, 32)
        f.pitch_dev = f.pitch - 2
        f.n_hands = 0.0
        f.wrists_visible = rng.uniform(0, 0.3)
        f.gaze_v = -rng.uniform(0.2, 0.5)
    if "slouch" in active:
        f.posture_dev = rng.gauss(4.0, 0.8)
        f.nose_shoulder -= 0.25
        f.face_size += 0.05
    if "hand_to_face" in active:
        f.n_hands = 1.0
        f.hand_mouth_dist = rng.uniform(0.05, 0.25)
        f.hand_cheek_dist = rng.uniform(0.1, 0.35)
        f.hand_face_dist = 0.0
        f.hand_curl = rng.choice([0.5, 0.75, 1.0])
    if "eye_rub" in active:
        f.n_hands = 1.0
        f.hand_eye_dist = rng.uniform(0.02, 0.2)
        f.eye_closed = rng.uniform(0.5, 0.9)
    if "looking_away" in active:
        f.yaw = rng.choice([-1, 1]) * rng.uniform(35, 60)
    if "drink" in active:
        f.drink_conf = rng.uniform(0.5, 0.9)
        f.drink_dist_mouth = rng.uniform(0.0, 0.3)
        f.drink_in_hand = 1.0
        f.n_hands = 1.0
    if "eat" in active:
        f.food_conf = rng.uniform(0.4, 0.8)
        f.food_dist_mouth = rng.uniform(0.1, 0.8)
        f.hand_mouth_dist = rng.uniform(0.1, 0.5)
        f.jaw_open = rng.uniform(0.1, 0.4)
    if "yawn" in active:
        f.jaw_open = rng.uniform(0.55, 0.85)
        f.mouth_open = rng.uniform(0.12, 0.25)
    if "drowsy" in active:
        f.eye_closed = rng.uniform(0.6, 0.95)
    if "restless" in active:
        f.motion = rng.uniform(0.04, 0.09)
        f.hand_speed = rng.uniform(2, 6)
    for k in ("hand_mouth_dist", "hand_eye_dist", "hand_cheek_dist", "hand_face_dist", "phone_dist_face",
              "phone_dist_hand", "drink_dist_mouth", "food_dist_mouth"):
        if getattr(f, k) == FAR and rng.random() < 0.3:
            setattr(f, k, rng.uniform(1.0, FAR))
    return f


def simulate(store: Store, cfg: Config, days: int = 7, seed: int = 7, features: bool = True,
             end_day: date | None = None, log=print) -> dict:
    rng = random.Random(seed)
    end_day = end_day or date.today()
    n_eps = 0
    n_feat = 0
    now = time.time()
    for di in range(days - 1, -1, -1):
        d = end_day - timedelta(days=di)
        day_start = datetime(d.year, d.month, d.day).timestamp()
        weekend = d.weekday() >= 5
        # desk sessions: morning block, afternoon block, evening block, with breaks
        blocks = []
        t = day_start + (10.5 if weekend else 9.0) * 3600 + rng.uniform(-1800, 1800)
        end_t = day_start + (23.5 if not weekend else 24.0) * 3600
        while t < end_t:
            length = rng.uniform(40, 130) * 60
            blocks.append((t, min(t + length, end_t)))
            t += length + rng.uniform(5, 45) * 60
        blocks = [(a, b) for a, b in blocks if b < now]
        episodes: list[Episode] = []
        for (a, b) in blocks:
            hours = (b - a) / 3600
            for k, (rate, med, sigma, evening) in PROFILE.items():
                mult = evening if (a - day_start) / 3600 >= 18 else 1.0
                n = np.random.default_rng(rng.randrange(1 << 30)).poisson(rate * hours * mult)
                for _ in range(int(n)):
                    dur = float(np.exp(np.log(med) + sigma * rng.gauss(0, 1)))
                    s = rng.uniform(a, max(a, b - dur))
                    ep = Episode(k, s, min(s + dur, b))
                    ep.n_frames = max(1, int(ep.duration * 4))
                    conf = min(0.99, max(0.2, rng.gauss(0.72, 0.18)))
                    ep.score_sum = conf * ep.n_frames
                    ep.rule_sum = conf * ep.n_frames
                    ep.peak = min(1.0, conf + 0.15)
                    episodes.append(ep)
        # gaps between blocks are absences
        for (a1, b1), (a2, b2) in zip(blocks, blocks[1:]):
            ep = Episode("absent", b1, a2)
            ep.n_frames = int(ep.duration)
            ep.score_sum = ep.rule_sum = 0.98 * ep.n_frames
            ep.peak = 1.0
            episodes.append(ep)
        for ep in episodes:
            review = 0.3 < ep.confidence < 0.8 and rng.random() < 0.5
            if review:
                col = (rng.randrange(40, 120), rng.randrange(40, 120), rng.randrange(60, 140))
                for i in range(rng.randint(2, 5)):
                    ep.thumbnails.append(_thumb(col, f"{ep.habit} #{i}"))
                    ep.thumb_times.append(ep.start + i * 3)
            store.commit_episode(ep, review)
            n_eps += 1
        # presence + features
        for (a, b) in blocks:
            for m in range(int(a // 60), int(b // 60)):
                store._presence_buf[m] = [10, 10]
                if len(store._presence_buf) > 500:
                    store.flush()
        if features:
            hz = cfg.feature_store_hz
            active_by_time = sorted(((e.start, e.end, e.habit) for e in episodes if e.habit != "absent"))
            for (a, b) in blocks:
                t = a
                j = 0
                while t < b:
                    active = {h for (s, e, h) in active_by_time if s <= t < e}
                    f = _feature_for(t, active, True, rng)
                    store.add_feature(t, f.to_vector())
                    n_feat += 1
                    t += 1.0 / hz
                    j += 1
            # a little absence context between blocks
            for (a1, b1), (a2, b2) in zip(blocks, blocks[1:]):
                t = b1
                while t < min(a2, b1 + 120):
                    store.add_feature(t, _feature_for(t, set(), False, rng).to_vector())
                    n_feat += 1
                    t += 1.0 / hz
        store.flush()
        # a few focus sessions inside the desk blocks
        from .focus import Sessions, score_window
        sess = Sessions(store)
        labels = ["Reading", "Problem set", "Essay", "Lecture notes", "Revision", None]
        for (a, b) in blocks:
            if b - a < 40 * 60 or rng.random() < 0.3:
                continue
            planned = rng.choice([25, 25, 50, 50, 90])
            s0 = a + rng.uniform(0, 10 * 60)
            s1 = min(b, s0 + planned * 60 + rng.uniform(-5 * 60, 8 * 60))
            eps_rows = store.episodes_between(s0, s1)
            summary = score_window(eps_rows, s0, s1, present_minutes=(s1 - s0) / 60)
            with store._lock:
                store._conn.execute("INSERT INTO sessions(start, end, planned_minutes, label, summary) VALUES (?, ?, ?, ?, ?)",
                                    (s0, s1, planned, rng.choice(labels), __import__("json").dumps(summary)))
                store._conn.commit()
        log(f"simulated {d.isoformat()}: {len(episodes)} episodes over {len(blocks)} desk blocks")
    store.log_event("simulate", {"days": days, "episodes": n_eps, "features": n_feat})
    return {"episodes": n_eps, "features": n_feat, "days": days}
