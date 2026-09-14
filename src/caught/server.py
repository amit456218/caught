"""Local HTTP API + dashboard. Binds to 127.0.0.1 only."""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import time

from .config import HABIT_KEYS, HABITS, Config
from .focus import Sessions, score_window
from .report import build_week_report, has_anthropic_credential, save_report, streaks, write_roast
from .store import Store, day_bounds

DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"


def _parse_day(s: str | None) -> date:
    if not s:
        return date.today()
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise HTTPException(400, "bad date") from e


def create_app(cfg: Config, store: Store, pipeline=None) -> FastAPI:
    app = FastAPI(title="Caught", docs_url=None, redoc_url=None)
    state = {"pipeline": pipeline, "report_cache": {}}
    sessions = Sessions(store)

    def today_score() -> dict | None:
        start, end = day_bounds(date.today())
        span = store.presence_span(start, end)
        if span is None:
            return None
        s0, s1 = span
        p = state["pipeline"]
        if p is not None and p.status.get("running"):
            s1 = max(s1, time.time())
        eps = store.episodes_between(s0, s1)
        out = score_window(eps, s0, s1, present_minutes=store.presence_minutes(s0, s1), count_absence=False)
        out["window_start"], out["window_end"] = s0, s1
        return out

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return DASHBOARD.read_text()

    @app.get("/api/habits")
    def habits() -> dict:
        return {k: {"label": h.label, "kind": h.kind, "icon": h.icon, "description": h.description,
                    "goal": cfg.goals.get(k, h.daily_budget)} for k, h in HABITS.items()}

    @app.get("/api/status")
    def status() -> dict:
        p = state["pipeline"]
        st = dict(p.status) if p else {"running": False}
        if p and p.error:
            st["error"] = p.error
        st["labels"] = store.label_counts()
        st["features_stored"] = store.feature_count()
        st["last_train"] = store.last_model_run()
        st["config"] = {"store_review_clips": cfg.store_review_clips, "notifications": cfg.notifications,
                        "roast_enabled": cfg.roast_enabled, "model_blend": cfg.model_blend,
                        "preview": cfg.preview_in_dashboard}
        st["has_credential"] = has_anthropic_credential()
        return st

    @app.get("/api/home")
    def home() -> dict:
        p = state["pipeline"]
        st = p.status if p else {}
        y0, y1 = day_bounds(date.today())
        today = today_score()
        yesterday = None
        ys = store.presence_span(y0 - 86400, y0)
        if ys:
            yesterday = score_window(store.episodes_between(ys[0], ys[1]), ys[0], ys[1], count_absence=False)
        wk = streaks(store, cfg, date.today())
        return {
            "ready": {"camera_running": bool(st.get("running")), "paused": bool(st.get("paused")),
                      "calibrated": bool(st.get("calibrated")) or cfg.calibration_path.exists(),
                      "calibration_auto": bool(st.get("calibration_auto")),
                      "model_ready": bool(st.get("model_ready")), "present": bool(st.get("present")),
                      "pending_reviews": store.label_counts()["pending"], "error": st.get("error")},
            "today": today, "yesterday_score": yesterday["score"] if yesterday else None,
            "session": sessions.live(), "recent_sessions": sessions.recent(days=7, limit=6),
            "goals": cfg.goals, "streaks": wk,
            "active": [k for k, v in (st.get("habits") or {}).items() if v.get("active")],
        }

    @app.post("/api/session/start")
    async def session_start(req: Request) -> dict:
        body = await req.json() if await req.body() else {}
        minutes = float(body.get("minutes", 25))
        if not (1 <= minutes <= 600):
            raise HTTPException(400, "minutes must be between 1 and 600")
        return sessions.start(minutes, (body.get("label") or "").strip()[:60] or None)

    @app.post("/api/session/end")
    def session_end() -> dict:
        s = sessions.end()
        if s is None:
            raise HTTPException(409, "no session running")
        return s

    @app.get("/api/session/current")
    def session_current() -> dict:
        return sessions.live() or {}

    @app.get("/api/sessions")
    def session_list(days: float = 7) -> list[dict]:
        return sessions.recent(days=days, limit=50)

    @app.delete("/api/session/{sid}")
    def session_delete(sid: int) -> dict:
        sessions.delete(sid)
        return {"ok": True}

    @app.get("/api/day")
    def day(date: str | None = None) -> dict:  # noqa: A002
        d = _parse_day(date)
        out = store.day_summary(d)
        y = store.day_summary(d - timedelta(days=1))
        out["yesterday_totals"] = y["totals"]
        out["yesterday_present_minutes"] = y["present_minutes"]
        out["goals"] = cfg.goals
        return out

    @app.get("/api/week")
    def week(end: str | None = None) -> dict:
        d = _parse_day(end)
        out = store.week_summary(d)
        prev = store.week_summary(d - timedelta(days=7))
        out["prev_days"] = prev["days"]
        out["streaks"] = streaks(store, cfg, d)
        out["goals"] = cfg.goals
        return out

    @app.get("/api/episodes")
    def episodes(start: float, end: float, habit: str | None = None, rejected: bool = False) -> list[dict]:
        return [e.to_dict() for e in store.episodes_between(start, end, habit, include_rejected=rejected)]

    @app.get("/api/review")
    def review(limit: int = 12) -> list[dict]:
        out = []
        for e in store.review_queue(limit):
            d = e.to_dict()
            d["thumbs"] = [f"/api/samples/{e.id}/{i}.jpg" for i in range(store.thumbnail_count(e.id))]
            d["label_text"] = HABITS[e.habit].label
            d["when"] = datetime.fromtimestamp(e.start).strftime("%a %H:%M")
            out.append(d)
        return out

    @app.post("/api/label")
    async def label(req: Request) -> dict:
        body = await req.json()
        eid = int(body.get("id"))
        lab = body.get("label")
        if store.episode(eid) is None:
            raise HTTPException(404, "no such episode")
        store.set_label(eid, None if lab is None else bool(lab))
        return {"ok": True, "labels": store.label_counts()}

    @app.get("/api/samples/{eid}/{i}.jpg")
    def sample(eid: int, i: int) -> FileResponse:
        p = store.thumbnail_path(eid, i)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p), media_type="image/jpeg")

    @app.get("/api/preview.jpg")
    def preview() -> Response:
        p = state["pipeline"]
        if not cfg.preview_in_dashboard or p is None:
            raise HTTPException(404, "preview disabled")
        jpg = p.preview_jpeg()
        if not jpg:
            raise HTTPException(404, "no frame yet")
        return Response(content=jpg, media_type="image/jpeg")

    @app.post("/api/calibrate")
    async def calibrate(req: Request) -> dict:
        p = state["pipeline"]
        if p is None:
            raise HTTPException(409, "pipeline not running")
        body = await req.json() if await req.body() else {}
        p.start_calibration(float(body.get("seconds", 10)))
        return {"ok": True}

    @app.post("/api/pause")
    async def pause(req: Request) -> dict:
        p = state["pipeline"]
        if p is None:
            raise HTTPException(409, "pipeline not running")
        body = await req.json()
        p.pause(bool(body.get("paused")))
        return {"ok": True, "paused": p.paused}

    @app.post("/api/train")
    def train() -> dict:
        p = state["pipeline"]
        if p is not None:
            started = p.train_async("dashboard")
            return {"ok": started, "async": True}
        from .model.train import train as _train
        res = _train(store, cfg.model_path)
        return {"ok": res.get("ok", False), "async": False, "result": res}

    @app.get("/api/settings")
    def get_settings() -> dict:
        return {"store_review_clips": cfg.store_review_clips, "notifications": cfg.notifications,
                "roast_enabled": cfg.roast_enabled, "model_blend": cfg.model_blend,
                "preview_in_dashboard": cfg.preview_in_dashboard, "goals": cfg.goals,
                "retention_days": cfg.retention_days, "fps_active": cfg.fps_active}

    @app.post("/api/settings")
    async def set_settings(req: Request) -> dict:
        body = await req.json()
        allowed = {"store_review_clips", "notifications", "roast_enabled", "model_blend",
                   "preview_in_dashboard", "retention_days", "fps_active"}
        patch = {k: v for k, v in body.items() if k in allowed}
        if "goals" in body and isinstance(body["goals"], dict):
            for k, v in body["goals"].items():
                if k in HABIT_KEYS:
                    cfg.goals[k] = float(v)
        cfg.update(patch)
        cfg.save()
        state["report_cache"].clear()
        return get_settings()

    @app.get("/api/report/week")
    def report_week(end: str | None = None, roast: bool = False, refresh: bool = False) -> dict:
        d = _parse_day(end)
        key = (d.isoformat(), roast)
        if not refresh and key in state["report_cache"]:
            return state["report_cache"][key]
        rep = build_week_report(store, cfg, d)
        if roast:
            rep = write_roast(rep, cfg)
        rep["path"] = save_report(rep, cfg)
        state["report_cache"][key] = rep
        return rep

    @app.get("/api/events")
    def events(limit: int = 50) -> list[dict]:
        return store.events(limit)

    @app.post("/api/wipe")
    async def wipe(req: Request) -> dict:
        body = await req.json()
        if body.get("confirm") != "wipe":
            raise HTTPException(400, "send {confirm: 'wipe'}")
        store.wipe()
        state["report_cache"].clear()
        return {"ok": True}

    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def serve_in_thread(app: FastAPI, host: str, port: int):
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, name="server", daemon=True)
    t.start()
    return server, t
