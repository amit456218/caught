"""Focus sessions and the focus score.

A session is a stretch of intended work (a pomodoro, a lecture, an essay). At the end it
gets a summary: what happened, a 0-100 focus score with a visible breakdown, the longest
clean stretch, and the one thing to fix first. The same scoring is applied to a whole day.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import HABITS
from .store import EpisodeRow, Store

# Penalty per 100% share of the window spent in a habit (points), and per count for count habits.
SHARE_PENALTY = {"phone": 110.0, "looking_down": 60.0, "absent": 70.0, "slouch": 25.0,
                 "hand_to_face": 15.0, "looking_away": 30.0, "restless": 20.0, "eat": 10.0}
COUNT_PENALTY = {"drowsy": 4.0, "yawn": 1.0, "eye_rub": 1.0}
DISTRACTIONS = ("phone", "looking_down", "absent")
FIX_FIRST_ORDER = ("phone", "looking_down", "absent", "slouch", "looking_away", "hand_to_face", "restless",
                   "drowsy", "eat", "yawn", "eye_rub", "drink")


@dataclass
class Breakdown:
    habit: str
    label: str
    minutes: float
    count: int
    penalty: float


def _clip(a: float, b: float, s: float, e: float) -> float:
    return max(0.0, min(b, e) - max(a, s))


def score_window(episodes: list[EpisodeRow], start: float, end: float, present_minutes: float | None = None,
                 count_absence: bool = True) -> dict:
    """Score a window. Absence counts inside a focus session (you said you would be there) but
    not for a whole day, where breaks are normal."""
    span = max(end - start, 1.0)
    per: dict[str, dict] = {k: {"minutes": 0.0, "count": 0} for k in HABITS}
    for e in episodes:
        overlap = _clip(start, end, e.start, e.end)
        if overlap <= 0:
            continue
        per[e.habit]["minutes"] += overlap / 60
        per[e.habit]["count"] += 1
    total = 0.0
    breakdown: list[Breakdown] = []
    for k, w in SHARE_PENALTY.items():
        if k == "absent" and not count_absence:
            continue
        share = per[k]["minutes"] * 60 / span
        pen = min(w * share, w * 0.8)
        if pen > 0:
            breakdown.append(Breakdown(k, HABITS[k].label, per[k]["minutes"], per[k]["count"], round(pen, 1)))
        total += pen
    for k, w in COUNT_PENALTY.items():
        pen = min(w * per[k]["count"], 20.0)
        if pen > 0:
            breakdown.append(Breakdown(k, HABITS[k].label, per[k]["minutes"], per[k]["count"], round(pen, 1)))
        total += pen
    score = max(0, min(100, round(100 - total)))
    breakdown.sort(key=lambda b: -b.penalty)
    # longest clean stretch: no phone, looking_down, or absence
    distracting = DISTRACTIONS if count_absence else tuple(k for k in DISTRACTIONS if k != "absent")
    marks = sorted((max(start, e.start), min(end, e.end)) for e in episodes if e.habit in distracting and e.end > start and e.start < end)
    longest, cur = 0.0, start
    for s, e in marks:
        longest = max(longest, s - cur)
        cur = max(cur, e)
    longest = max(longest, end - cur)
    distractions = sum(1 for e in episodes if e.habit in ("phone", "looking_down") and e.end > start and e.start < end)
    fix = next((k for k in FIX_FIRST_ORDER if k in per and (k != "absent" or count_absence) and (
        (HABITS[k].kind == "duration" and per[k]["minutes"] >= 2) or (HABITS[k].kind == "count" and per[k]["count"] >= 3))), None)
    return {
        "score": score, "minutes": round(span / 60, 1),
        "present_minutes": present_minutes if present_minutes is not None else None,
        "breakdown": [b.__dict__ for b in breakdown],
        "per_habit": per, "longest_clean_minutes": round(longest / 60, 1),
        "distractions": distractions,
        "phone_minutes": round(per["phone"]["minutes"], 1), "phone_pickups": per["phone"]["count"],
        "fix_first": fix, "fix_first_label": HABITS[fix].label if fix else None,
    }


SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start REAL NOT NULL,
    end REAL,
    planned_minutes REAL NOT NULL,
    label TEXT,
    summary TEXT
);
"""


class Sessions:
    def __init__(self, store: Store):
        self.store = store
        with store._lock:
            store._conn.executescript(SESSION_SCHEMA)

    def current(self) -> dict | None:
        with self.store._lock:
            r = self.store._conn.execute("SELECT * FROM sessions WHERE end IS NULL ORDER BY start DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def start(self, planned_minutes: float, label: str | None = None) -> dict:
        cur = self.current()
        if cur:
            self.end()
        with self.store._lock:
            c = self.store._conn.execute("INSERT INTO sessions(start, planned_minutes, label) VALUES (?, ?, ?)",
                                         (time.time(), float(planned_minutes), label))
            self.store._conn.commit()
            sid = int(c.lastrowid)
        self.store.log_event("session_start", {"id": sid, "planned": planned_minutes, "label": label})
        return self.get(sid)

    def live(self) -> dict | None:
        """The running session with a provisional summary."""
        cur = self.current()
        if not cur:
            return None
        now = time.time()
        eps = self.store.episodes_between(cur["start"], now)
        summary = score_window(eps, cur["start"], now)
        last_distraction = max((e.end for e in eps if e.habit in DISTRACTIONS), default=cur["start"])
        summary["clean_for_minutes"] = round((now - last_distraction) / 60, 1)
        summary["elapsed_minutes"] = round((now - cur["start"]) / 60, 1)
        summary["remaining_minutes"] = round(max(0.0, cur["planned_minutes"] - (now - cur["start"]) / 60), 1)
        return {**cur, "summary": summary}

    def end(self) -> dict | None:
        cur = self.current()
        if not cur:
            return None
        now = time.time()
        self.store.flush()
        eps = self.store.episodes_between(cur["start"], now)
        summary = score_window(eps, cur["start"], now, present_minutes=self.store.presence_minutes(cur["start"], now))
        with self.store._lock:
            self.store._conn.execute("UPDATE sessions SET end = ?, summary = ? WHERE id = ?",
                                     (now, json.dumps(summary), cur["id"]))
            self.store._conn.commit()
        self.store.log_event("session_end", {"id": cur["id"], "score": summary["score"]})
        return self.get(cur["id"])

    def get(self, sid: int) -> dict:
        with self.store._lock:
            r = self.store._conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        d = dict(r)
        d["summary"] = json.loads(d["summary"]) if d.get("summary") else None
        return d

    def recent(self, days: float = 7.0, limit: int = 20) -> list[dict]:
        since = time.time() - days * 86400
        with self.store._lock:
            rows = self.store._conn.execute("SELECT * FROM sessions WHERE start > ? AND end IS NOT NULL ORDER BY start DESC LIMIT ?",
                                            (since, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d["summary"]) if d.get("summary") else None
            out.append(d)
        return out

    def delete(self, sid: int) -> None:
        with self.store._lock:
            self.store._conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            self.store._conn.commit()
