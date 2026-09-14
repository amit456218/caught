"""SQLite persistence: episodes, labels, feature history, review samples, system events."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .config import HABIT_KEYS, HABITS
from .episodes import Episode
from .features import FEATURE_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    habit TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    confidence REAL NOT NULL,
    peak REAL NOT NULL,
    rule_conf REAL NOT NULL DEFAULT 0,
    model_conf REAL,
    source TEXT NOT NULL,
    n_frames INTEGER NOT NULL,
    label INTEGER,
    labeled_at REAL,
    has_samples INTEGER NOT NULL DEFAULT 0,
    review INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_start ON episodes(start);
CREATE INDEX IF NOT EXISTS idx_episodes_habit ON episodes(habit, start);
CREATE TABLE IF NOT EXISTS features (
    ts REAL PRIMARY KEY,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    metrics TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS presence (
    minute INTEGER PRIMARY KEY,
    frames INTEGER NOT NULL,
    present INTEGER NOT NULL
);
"""


@dataclass
class EpisodeRow:
    id: int
    habit: str
    start: float
    end: float
    confidence: float
    peak: float
    rule_conf: float
    model_conf: float | None
    source: str
    n_frames: int
    label: int | None
    labeled_at: float | None
    has_samples: bool
    review: bool
    created: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["duration"] = self.duration
        d["label"] = None if self.label is None else bool(self.label)
        return d


def _row(r: sqlite3.Row) -> EpisodeRow:
    return EpisodeRow(r["id"], r["habit"], r["start"], r["end"], r["confidence"], r["peak"],
                      r["rule_conf"], r["model_conf"], r["source"], r["n_frames"], r["label"],
                      r["labeled_at"], bool(r["has_samples"]), bool(r["review"]), r["created"])


def day_bounds(d: date) -> tuple[float, float]:
    start = datetime(d.year, d.month, d.day).timestamp()
    return start, start + 86400


class Store:
    def __init__(self, db_path: Path, samples_path: Path):
        self.db_path = db_path
        self.samples_path = samples_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        samples_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._feature_buf: list[tuple[float, bytes]] = []
        self._presence_buf: dict[int, list[int]] = {}

    def close(self) -> None:
        with self._lock:
            self.flush()
            self._conn.close()

    # ------------------------------------------------------------------ episodes
    def commit_episode(self, ep: Episode, review: bool) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes(habit,start,end,confidence,peak,rule_conf,model_conf,source,n_frames,"
                "has_samples,review,created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ep.habit, ep.start, ep.end, ep.confidence, ep.peak, ep.rule_conf,
                 ep.model_conf if ep.model_sum else None, ep.source, ep.n_frames,
                 1 if ep.thumbnails else 0, 1 if review else 0, time.time()))
            eid = int(cur.lastrowid)
            self._conn.commit()
        if ep.thumbnails:
            d = self.samples_path / str(eid)
            d.mkdir(parents=True, exist_ok=True)
            for i, (jpg, t) in enumerate(zip(ep.thumbnails, ep.thumb_times)):
                (d / f"{i}.jpg").write_bytes(jpg)
            (d / "meta.json").write_text(json.dumps({"times": ep.thumb_times, "habit": ep.habit}))
        ep.id = eid
        return eid

    def episodes_between(self, start: float, end: float, habit: str | None = None,
                         include_rejected: bool = False) -> list[EpisodeRow]:
        q = "SELECT * FROM episodes WHERE end > ? AND start < ?"
        args: list = [start, end]
        if habit:
            q += " AND habit = ?"
            args.append(habit)
        if not include_rejected:
            q += " AND (label IS NULL OR label = 1)"
        q += " ORDER BY start"
        with self._lock:
            return [_row(r) for r in self._conn.execute(q, args)]

    def episode(self, eid: int) -> EpisodeRow | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone()
        return _row(r) if r else None

    def set_label(self, eid: int, label: bool | None) -> None:
        with self._lock:
            self._conn.execute("UPDATE episodes SET label = ?, labeled_at = ?, review = 0 WHERE id = ?",
                               (None if label is None else int(label), time.time(), eid))
            self._conn.commit()

    def review_queue(self, limit: int = 20) -> list[EpisodeRow]:
        """Unlabeled episodes with thumbnails, most uncertain first."""
        q = ("SELECT * FROM episodes WHERE label IS NULL AND has_samples = 1 "
             "ORDER BY review DESC, ABS(confidence - 0.5) ASC, start DESC LIMIT ?")
        with self._lock:
            return [_row(r) for r in self._conn.execute(q, (limit,))]

    def labeled(self, since: float | None = None) -> list[EpisodeRow]:
        q = "SELECT * FROM episodes WHERE label IS NOT NULL"
        args: list = []
        if since:
            q += " AND start > ?"
            args.append(since)
        with self._lock:
            return [_row(r) for r in self._conn.execute(q, args)]

    def label_counts(self) -> dict[str, int]:
        with self._lock:
            r = self._conn.execute(
                "SELECT SUM(label = 1) AS pos, SUM(label = 0) AS neg, SUM(label IS NULL AND has_samples = 1) AS pending "
                "FROM episodes").fetchone()
        return {"confirmed": int(r["pos"] or 0), "rejected": int(r["neg"] or 0), "pending": int(r["pending"] or 0)}

    def thumbnail_path(self, eid: int, i: int) -> Path:
        return self.samples_path / str(eid) / f"{i}.jpg"

    def thumbnail_count(self, eid: int) -> int:
        d = self.samples_path / str(eid)
        return len(list(d.glob("*.jpg"))) if d.exists() else 0

    # ------------------------------------------------------------------ aggregates
    def day_summary(self, d: date) -> dict:
        start, end = day_bounds(d)
        eps = self.episodes_between(start, end)
        totals: dict[str, dict] = {k: {"minutes": 0.0, "count": 0} for k in HABIT_KEYS}
        for e in eps:
            s, t = max(e.start, start), min(e.end, end)
            totals[e.habit]["minutes"] += (t - s) / 60
            totals[e.habit]["count"] += 1
        present_min = self.presence_minutes(start, end)
        return {"date": d.isoformat(), "totals": totals, "present_minutes": present_min,
                "episodes": [e.to_dict() for e in eps]}

    def week_summary(self, end_day: date) -> dict:
        days = [end_day - timedelta(days=i) for i in range(6, -1, -1)]
        per_day = []
        heat: dict[str, list[list[float]]] = {k: [[0.0] * 24 for _ in days] for k in HABIT_KEYS}
        for di, d in enumerate(days):
            start, end = day_bounds(d)
            eps = self.episodes_between(start, end)
            totals = {k: {"minutes": 0.0, "count": 0} for k in HABIT_KEYS}
            for e in eps:
                s, t = max(e.start, start), min(e.end, end)
                totals[e.habit]["minutes"] += (t - s) / 60
                totals[e.habit]["count"] += 1
                # spread across hour buckets
                cur = s
                while cur < t:
                    hour_start = start + int((cur - start) // 3600) * 3600
                    nxt = min(t, hour_start + 3600)
                    h = int((hour_start - start) // 3600)
                    if 0 <= h < 24:
                        heat[e.habit][di][h] += (nxt - cur) / 60 if HABITS[e.habit].kind == "duration" else 1.0
                    cur = nxt
            per_day.append({"date": d.isoformat(), "totals": totals,
                            "present_minutes": self.presence_minutes(start, end)})
        return {"days": per_day, "heatmap": heat}

    def presence_minutes(self, start: float, end: float) -> float:
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*) AS n FROM presence WHERE minute >= ? AND minute < ? AND present > frames / 2",
                (int(start // 60), int(end // 60))).fetchone()
        return float(r["n"] or 0)

    def presence_span(self, start: float, end: float) -> tuple[float, float] | None:
        """First and last minute (as timestamps) with someone at the desk inside [start, end)."""
        with self._lock:
            r = self._conn.execute(
                "SELECT MIN(minute) AS a, MAX(minute) AS b FROM presence WHERE minute >= ? AND minute < ? AND present > frames / 2",
                (int(start // 60), int(end // 60))).fetchone()
        if not r or r["a"] is None:
            return None
        return float(r["a"] * 60), float(r["b"] * 60 + 60)

    def first_day(self) -> date | None:
        with self._lock:
            r = self._conn.execute("SELECT MIN(start) AS s FROM episodes").fetchone()
        return date.fromtimestamp(r["s"]) if r and r["s"] else None

    # ------------------------------------------------------------------ features
    def add_feature(self, ts: float, vec: np.ndarray) -> None:
        self._feature_buf.append((ts, vec.astype(np.float32).tobytes()))
        if len(self._feature_buf) >= 40:
            self.flush()

    def add_presence(self, ts: float, present: bool) -> None:
        m = int(ts // 60)
        b = self._presence_buf.setdefault(m, [0, 0])
        b[0] += 1
        b[1] += 1 if present else 0
        if len(self._presence_buf) > 2:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if self._feature_buf:
                self._conn.executemany("INSERT OR REPLACE INTO features(ts, vec) VALUES (?, ?)", self._feature_buf)
                self._feature_buf = []
            if self._presence_buf:
                rows = [(m, v[0], v[1]) for m, v in self._presence_buf.items()]
                self._conn.executemany(
                    "INSERT INTO presence(minute, frames, present) VALUES (?, ?, ?) "
                    "ON CONFLICT(minute) DO UPDATE SET frames = frames + excluded.frames, present = present + excluded.present",
                    rows)
                self._presence_buf = {}
            self._conn.commit()

    def features_between(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            rows = self._conn.execute("SELECT ts, vec FROM features WHERE ts >= ? AND ts < ? ORDER BY ts",
                                      (start, end)).fetchall()
        if not rows:
            return np.zeros(0), np.zeros((0, FEATURE_DIM), np.float32)
        ts = np.array([r["ts"] for r in rows])
        vecs = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return ts, vecs

    def feature_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) AS n FROM features").fetchone()["n"])

    def feature_span(self) -> tuple[float, float] | None:
        with self._lock:
            r = self._conn.execute("SELECT MIN(ts) AS a, MAX(ts) AS b FROM features").fetchone()
        return (r["a"], r["b"]) if r and r["a"] is not None else None

    # ------------------------------------------------------------------ misc
    def log_event(self, kind: str, detail: str | dict | None = None) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO events(ts, kind, detail) VALUES (?, ?, ?)",
                               (time.time(), kind, json.dumps(detail) if isinstance(detail, dict) else detail))
            self._conn.commit()

    def events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def record_model_run(self, metrics: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO model_runs(ts, metrics) VALUES (?, ?)", (time.time(), json.dumps(metrics)))
            self._conn.commit()

    def last_model_run(self) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM model_runs ORDER BY ts DESC LIMIT 1").fetchone()
        return {"ts": r["ts"], **json.loads(r["metrics"])} if r else None

    def prune(self, retention_days: int) -> dict:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            n_feat = self._conn.execute("DELETE FROM features WHERE ts < ?", (cutoff,)).rowcount
            old = self._conn.execute("SELECT id FROM episodes WHERE has_samples = 1 AND label IS NULL AND start < ?",
                                     (cutoff,)).fetchall()
            for r in old:
                d = self.samples_path / str(r["id"])
                for p in d.glob("*"):
                    p.unlink(missing_ok=True)
                if d.exists():
                    d.rmdir()
            self._conn.execute("UPDATE episodes SET has_samples = 0 WHERE has_samples = 1 AND label IS NULL AND start < ?",
                               (cutoff,))
            self._conn.commit()
        return {"features_deleted": n_feat, "sample_dirs_deleted": len(old)}

    def wipe(self) -> None:
        with self._lock:
            for t in ("episodes", "features", "events", "model_runs", "presence"):
                self._conn.execute(f"DELETE FROM {t}")
            self._conn.commit()
        for d in self.samples_path.glob("*"):
            if d.is_dir():
                for p in d.glob("*"):
                    p.unlink(missing_ok=True)
                d.rmdir()
