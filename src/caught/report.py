"""Weekly report: the numbers, the projections, and (optionally) a Claude-written roast.

Only aggregates leave the machine, and only if roast_enabled is on and an Anthropic
credential is available. No images, no per-episode data.
"""
from __future__ import annotations

import html
import json
import os
from datetime import date, timedelta

from .config import HABIT_KEYS, HABITS, Config
from .store import Store, day_bounds

PROMPT = """You write the Sunday summary for "Caught", a desk camera that quietly logs one person's
habits for a week and then shows them the chart. Your job: one short, dry, funny paragraph
(4 to 6 sentences) and then exactly three one-line observations. Be specific, use the numbers,
name the worst hour and the best day. Roast gently; the reader chose this. No advice, no
lectures, no emojis, no headings. Plain text only."""


def _totals(store: Store, d: date) -> tuple[dict[str, dict], float]:
    start, end = day_bounds(d)
    eps = store.episodes_between(start, end)
    t = {k: {"minutes": 0.0, "count": 0} for k in HABIT_KEYS}
    for e in eps:
        s, f = max(e.start, start), min(e.end, end)
        t[e.habit]["minutes"] += (f - s) / 60
        t[e.habit]["count"] += 1
    return t, store.presence_minutes(start, end)


def streaks(store: Store, cfg: Config, end_day: date, max_days: int = 60) -> dict[str, int]:
    """Consecutive days (ending end_day) under goal, counting only days with >= 60 present minutes."""
    out = {k: 0 for k in HABIT_KEYS}
    alive = {k: True for k in HABIT_KEYS}
    for i in range(max_days):
        d = end_day - timedelta(days=i)
        t, present = _totals(store, d)
        if present < 60:
            if i == 0:
                continue  # today may simply not have started
            break
        for k in HABIT_KEYS:
            if not alive[k]:
                continue
            v = t[k]["minutes"] if HABITS[k].kind == "duration" else t[k]["count"]
            if v <= cfg.goals.get(k, HABITS[k].daily_budget):
                out[k] += 1
            else:
                alive[k] = False
        if not any(alive.values()):
            break
    return out


def build_week_report(store: Store, cfg: Config, end_day: date | None = None) -> dict:
    end_day = end_day or date.today()
    days = [end_day - timedelta(days=i) for i in range(6, -1, -1)]
    prev_days = [d - timedelta(days=7) for d in days]
    week = store.week_summary(end_day)
    this_t = {k: {"minutes": 0.0, "count": 0} for k in HABIT_KEYS}
    prev_t = {k: {"minutes": 0.0, "count": 0} for k in HABIT_KEYS}
    present = 0.0
    prev_present = 0.0
    for d in days:
        t, p = _totals(store, d)
        present += p
        for k in HABIT_KEYS:
            this_t[k]["minutes"] += t[k]["minutes"]
            this_t[k]["count"] += t[k]["count"]
    for d in prev_days:
        t, p = _totals(store, d)
        prev_present += p
        for k in HABIT_KEYS:
            prev_t[k]["minutes"] += t[k]["minutes"]
            prev_t[k]["count"] += t[k]["count"]

    habits = []
    for k in HABIT_KEYS:
        h = HABITS[k]
        val = this_t[k]["minutes"] if h.kind == "duration" else this_t[k]["count"]
        prev = prev_t[k]["minutes"] if h.kind == "duration" else prev_t[k]["count"]
        heat = week["heatmap"][k]
        by_hour = [sum(heat[di][hh] for di in range(7)) for hh in range(24)]
        by_day = [sum(heat[di]) for di in range(7)]
        worst_hour = int(max(range(24), key=lambda i: by_hour[i])) if max(by_hour) > 0 else None
        worst_day = days[int(max(range(7), key=lambda i: by_day[i]))].isoformat() if max(by_day) > 0 else None
        best_day = days[int(min(range(7), key=lambda i: by_day[i]))].isoformat() if max(by_day) > 0 else None
        share = (this_t[k]["minutes"] / present) if present and h.kind == "duration" and k != "absent" else None
        habits.append({
            "key": k, "label": h.label, "kind": h.kind, "icon": h.icon,
            "value": round(val, 1), "prev": round(prev, 1),
            "delta": round(val - prev, 1),
            "minutes": round(this_t[k]["minutes"], 1), "count": this_t[k]["count"],
            "goal_per_day": cfg.goals.get(k, h.daily_budget),
            "per_year": round(val * 52, 0),
            "per_year_hours": round(this_t[k]["minutes"] * 52 / 60, 1),
            "worst_hour": worst_hour, "worst_day": worst_day, "best_day": best_day,
            "share_of_desk_time": round(share, 3) if share is not None else None,
        })
    habit_minutes = sum(h["minutes"] for h in habits if h["kind"] == "duration" and h["key"] != "absent")
    report = {
        "start": days[0].isoformat(), "end": days[-1].isoformat(),
        "present_minutes": round(present, 1), "prev_present_minutes": round(prev_present, 1),
        "habit_minutes": round(habit_minutes, 1),
        "habit_share": round(habit_minutes / present, 3) if present else None,
        "habits": habits, "streaks": streaks(store, cfg, end_day),
        "days": week["days"], "generated_at": date.today().isoformat(),
        "roast": None, "roast_error": None,
    }
    return report


def write_roast(report: dict, cfg: Config, model: str = "claude-opus-5") -> dict:
    """Ask Claude for the paragraph. Sends aggregates only. Never raises."""
    if not cfg.roast_enabled:
        report["roast_error"] = "roast disabled in settings"
        return report
    try:
        import anthropic
    except ImportError:
        report["roast_error"] = "anthropic package not installed"
        return report
    facts = {
        "week": f"{report['start']} to {report['end']}",
        "hours_at_desk": round(report["present_minutes"] / 60, 1),
        "hours_at_desk_last_week": round(report["prev_present_minutes"] / 60, 1),
        "share_of_desk_time_spent_in_habits": report["habit_share"],
        "habits": [{k: v for k, v in h.items() if k in ("label", "kind", "value", "prev", "worst_hour",
                                                          "worst_day", "best_day", "per_year_hours")}
                   for h in report["habits"] if h["value"] > 0],
        "streaks_days_under_goal": report["streaks"],
    }
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=1500, system=PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": "This week's numbers:\n" + json.dumps(facts, indent=1)}],
        )
        if resp.stop_reason == "refusal":
            report["roast_error"] = "model declined"
            return report
        report["roast"] = "".join(b.text for b in resp.content if b.type == "text").strip()
    except anthropic.AuthenticationError:
        report["roast_error"] = "no Anthropic credential (set ANTHROPIC_API_KEY or run `ant auth login`)"
    except anthropic.APIConnectionError:
        report["roast_error"] = "network error reaching the API"
    except anthropic.APIStatusError as e:
        report["roast_error"] = f"API error {e.status_code}"
    except Exception as e:  # noqa: BLE001
        report["roast_error"] = f"{type(e).__name__}: {e}"
    return report


def render_html(report: dict) -> str:
    def esc(x: object) -> str:
        return html.escape(str(x))

    rows = []
    worst = max((h for h in report["habits"] if h["kind"] == "duration" and h["key"] != "absent"),
                key=lambda h: h["minutes"], default=None)
    ordered = sorted((h for h in report["habits"] if h["key"] != "absent"),
                     key=lambda h: -h["minutes"] if h["kind"] == "duration" else -h["count"])
    ordered += [h for h in report["habits"] if h["key"] == "absent"]
    for h in ordered:
        unit = "min" if h["kind"] == "duration" else "×"
        delta = h["delta"]
        sign = "+" if delta > 0 else ""
        hour = f"{h['worst_hour']:02d}:00" if h["worst_hour"] is not None else "–"
        rows.append(f"<tr><td>{esc(h['icon'])} {esc(h['label'])}</td><td class=n>{h['value']:.0f} {unit}</td>"
                    f"<td class=n>{sign}{delta:.0f}</td><td>{hour}</td><td class=n>{h['per_year_hours']:.0f} h</td>"
                    f"<td class=n>{report['streaks'].get(h['key'], 0)} d</td></tr>")
    roast = f"<blockquote>{esc(report['roast'])}</blockquote>" if report.get("roast") else \
        (f"<p class=muted>Roast unavailable: {esc(report['roast_error'])}</p>" if report.get("roast_error") else "")
    hero = ""
    if worst and worst["minutes"] > 0:
        hero = (f"<p class=hero>{worst['minutes']:.0f} minutes</p><p class=sub>{esc(worst['label'].lower())} this week. "
                f"At this rate that is {worst['per_year_hours']:.0f} hours a year.</p>")
    share = f"{report['habit_share'] * 100:.0f}%" if report["habit_share"] is not None else "–"
    return f"""<!doctype html><meta charset=utf-8><title>Caught · week of {esc(report['start'])}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#0b0b0b;background:#f9f9f7}}
h1{{font-size:20px;font-weight:600;margin:0 0 4px}} .muted{{color:#898781}} .hero{{font-size:56px;font-weight:600;margin:24px 0 0;line-height:1}}
.sub{{margin:4px 0 24px;color:#52514e}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
td,th{{padding:8px 6px;border-bottom:1px solid #e1e0d9;text-align:left}} th{{color:#898781;font-weight:500;font-size:12px}} .n{{text-align:right}}
blockquote{{margin:24px 0;padding:16px 20px;border-left:3px solid #2a78d6;background:#fcfcfb;white-space:pre-line}}
</style>
<h1>Caught · {esc(report['start'])} to {esc(report['end'])}</h1>
<p class=muted>{report['present_minutes'] / 60:.1f} hours at the desk · {share} of it in a logged habit</p>
{hero}{roast}
<table><tr><th>Habit</th><th class=n>This week</th><th class=n>vs last</th><th>Worst hour</th><th class=n>Per year</th><th class=n>Streak</th></tr>
{''.join(rows)}</table>
<p class=muted>Generated {esc(report['generated_at'])}. Everything computed locally.</p>"""


def save_report(report: dict, cfg: Config) -> str:
    cfg.ensure_dirs()
    path = cfg.data_path / "reports" / f"week-{report['end']}.html"
    path.write_text(render_html(report))
    (cfg.data_path / "reports" / f"week-{report['end']}.json").write_text(json.dumps(report, indent=1))
    return str(path)


def has_anthropic_credential() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (os.path.expanduser("~/.config/anthropic") and os.path.isdir(os.path.expanduser("~/.config/anthropic")))
