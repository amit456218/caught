from fastapi.testclient import TestClient

from caught.server import create_app
from caught.simulate import simulate


def test_endpoints(store, cfg):
    simulate(store, cfg, days=2, seed=3, features=False, log=lambda *_: None)
    c = TestClient(create_app(cfg, store, None))
    assert c.get("/").status_code == 200 and "Caught" in c.get("/").text
    assert "phone" in c.get("/api/habits").json()
    st = c.get("/api/status").json()
    assert st["running"] is False and "labels" in st
    d = c.get("/api/day").json()
    assert "totals" in d and "yesterday_totals" in d
    w = c.get("/api/week").json()
    assert len(w["days"]) == 7 and "streaks" in w
    rv = c.get("/api/review?limit=2").json()
    if rv:
        assert c.get(rv[0]["thumbs"][0]).headers["content-type"] == "image/jpeg"
        assert c.post("/api/label", json={"id": rv[0]["id"], "label": False}).json()["ok"]
    assert c.get("/api/day?date=bad").status_code == 400
    assert c.post("/api/calibrate", json={}).status_code == 409
    s = c.post("/api/settings", json={"model_blend": 0.3, "goals": {"phone": 12}}).json()
    assert s["model_blend"] == 0.3 and s["goals"]["phone"] == 12.0
    rep = c.get("/api/report/week").json()
    assert rep["habits"] and rep["path"].endswith(".html")
    assert c.post("/api/wipe", json={"confirm": "no"}).status_code == 400
    h = c.get("/api/home").json()
    assert "ready" in h and "recent_sessions" in h
    s = c.post("/api/session/start", json={"minutes": 25, "label": "Test"}).json()
    assert s["end"] is None
    assert c.get("/api/session/current").json()["id"] == s["id"]
    e = c.post("/api/session/end").json()
    assert e["summary"]["score"] <= 100
    assert c.post("/api/session/end").status_code == 409
    assert c.post("/api/session/start", json={"minutes": 0}).status_code == 400
