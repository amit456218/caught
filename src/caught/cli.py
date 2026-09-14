"""Command line entry point."""
from __future__ import annotations

import argparse
import signal
import sys
import time
import webbrowser
from datetime import date

from .config import Config


def _store(cfg: Config):
    from .store import Store

    cfg.ensure_dirs()
    return Store(cfg.db_path, cfg.samples_path)


def cmd_run(args: argparse.Namespace) -> int:
    from .capture import CameraSource, VideoFileSource
    from .pipeline import Pipeline
    from .server import create_app, serve_in_thread

    cfg = Config.load(args.data)
    if args.camera is not None:
        cfg.camera_index = args.camera
    store = _store(cfg)
    if args.video:
        source = VideoFileSource(args.video, cfg.process_width, realtime=args.realtime)
    else:
        source = CameraSource(cfg.camera_index, cfg.process_width)
    pipe = Pipeline(cfg, source, store, use_objects=not args.no_objects, use_model=not args.no_model,
                    preview=cfg.preview_in_dashboard or args.window)
    app = create_app(cfg, store, pipe)
    port = args.port or cfg.server_port
    server, _ = serve_in_thread(app, cfg.server_host, port)
    pipe.start()
    url = f"http://{cfg.server_host}:{port}"
    print(f"caught: dashboard at {url}  (Ctrl-C to stop)")
    if args.open:
        webbrowser.open(url)

    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        if args.window:
            import cv2
            while not stop["flag"] and not pipe.finished:
                jpg = pipe.preview_jpeg()
                if jpg:
                    import numpy as np
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow("caught", img)
                if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                    break
            cv2.destroyAllWindows()
        else:
            while not stop["flag"] and not pipe.finished:
                time.sleep(0.25)
    finally:
        print("caught: stopping")
        pipe.stop()
        server.should_exit = True
        store.close()
    if pipe.error:
        print(f"caught: pipeline ended with error: {pipe.error}", file=sys.stderr)
        return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    cfg = Config.load(args.data)
    store = _store(cfg)
    app = create_app(cfg, store, None)
    port = args.port or cfg.server_port
    print(f"caught: dashboard (read-only, no camera) at http://{cfg.server_host}:{port}")
    if args.open:
        webbrowser.open(f"http://{cfg.server_host}:{port}")
    uvicorn.run(app, host=cfg.server_host, port=port, log_level="warning")
    store.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .capture import CameraSource
    from .pipeline import Pipeline

    cfg = Config.load(args.data)
    store = _store(cfg)
    pipe = Pipeline(cfg, CameraSource(cfg.camera_index, cfg.process_width), store,
                    use_objects=False, use_model=False, preview=False)
    pipe.start()
    print("caught: sit upright, look at the screen. Recording baseline in 3 seconds...")
    while not pipe.status.get("running") and not pipe.finished:
        time.sleep(0.1)
    time.sleep(3)
    pipe.start_calibration(args.seconds)
    while pipe.calibrator.active and not pipe.finished:
        time.sleep(0.2)
    pipe.stop()
    store.close()
    if pipe.calibrator.result and pipe.calibrator.result.valid:
        print(f"caught: calibrated from {pipe.calibrator.result.n_samples} frames -> {cfg.calibration_path}")
        return 0
    print("caught: calibration failed (face not visible enough)", file=sys.stderr)
    return 1


def cmd_train(args: argparse.Namespace) -> int:
    from .model.train import train

    cfg = Config.load(args.data)
    store = _store(cfg)
    res = train(store, cfg.model_path, days=args.days, epochs=args.epochs)
    store.close()
    return 0 if res.get("ok") else 1


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_week_report, save_report, write_roast

    cfg = Config.load(args.data)
    store = _store(cfg)
    end = date.fromisoformat(args.end) if args.end else date.today()
    rep = build_week_report(store, cfg, end)
    if args.roast:
        cfg.roast_enabled = True
        rep = write_roast(rep, cfg)
        if rep.get("roast_error"):
            print(f"roast unavailable: {rep['roast_error']}", file=sys.stderr)
    path = save_report(rep, cfg)
    store.close()
    print(f"caught: report written to {path}")
    for h in sorted(rep["habits"], key=lambda h: (h["key"] == "absent", -h["minutes"], -h["count"])):
        if h["value"] > 0:
            unit = "min" if h["kind"] == "duration" else "x"
            print(f"  {h['label']:<40} {h['value']:>7.0f} {unit}   ({h['delta']:+.0f} vs last week)")
    if rep.get("roast"):
        print("\n" + rep["roast"])
    if args.open:
        webbrowser.open(f"file://{path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .simulate import simulate

    cfg = Config.load(args.data)
    store = _store(cfg)
    if args.wipe:
        store.wipe()
    res = simulate(store, cfg, days=args.days, seed=args.seed, features=not args.no_features)
    store.close()
    print(f"caught: simulated {res['episodes']} episodes and {res['features']} feature rows over {res['days']} days")
    return 0


def cmd_cameras(args: argparse.Namespace) -> int:
    from .capture import list_cameras

    cams = list_cameras()
    if not cams:
        print("no cameras found (check macOS camera permission for your terminal)")
        return 1
    for c in cams:
        print(f"  index {c['index']}: {c['width']}x{c['height']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = Config.load(args.data)
    store = _store(cfg)
    today = store.day_summary(date.today())
    print(f"data: {cfg.data_path}")
    print(f"episodes today: {len(today['episodes'])}, present {today['present_minutes']:.0f} min")
    print(f"labels: {store.label_counts()}")
    print(f"features stored: {store.feature_count()}")
    print(f"last training: {store.last_model_run()}")
    print(f"calibration: {'yes' if cfg.calibration_path.exists() else 'no'}")
    store.close()
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    cfg = Config.load(args.data)
    if not args.yes:
        print("this deletes every episode, feature row, and review clip. re-run with --yes")
        return 1
    store = _store(cfg)
    store.wipe()
    store.close()
    for p in (cfg.model_path, cfg.calibration_path):
        p.unlink(missing_ok=True)
    print("caught: wiped")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .bench import format_table, run_bench, save_results

    cfg = Config.load(args.data)
    bench_dir = Path(args.dir).expanduser()
    if not bench_dir.exists():
        print(f"no such directory: {bench_dir}", file=sys.stderr)
        return 1
    results = run_bench(cfg, bench_dir, only=args.only, use_objects=not args.no_objects, trace=args.trace)
    print()
    print(format_table(results))
    strict = [r for r in results if not r.optional]
    soft = [r for r in results if r.optional]
    passed = sum(r.passed for r in strict)
    print(f"\n{passed}/{len(strict)} strict items pass; {sum(r.passed for r in soft)}/{len(soft)} optional items pass")
    out = bench_dir / "results.json"
    save_results(results, out)
    print(f"details written to {out}")
    return 0 if passed == len(strict) else 2


def cmd_record(args: argparse.Namespace) -> int:
    """Record a labeled clip from the camera into the bench folder."""
    import json
    from pathlib import Path

    import cv2

    from .capture import CameraSource

    cfg = Config.load(args.data)
    bench_dir = Path(args.dir).expanduser()
    bench_dir.mkdir(parents=True, exist_ok=True)
    src = CameraSource(cfg.camera_index, process_width=960)
    src.set_target_fps(10)
    src.start()
    stem = f"{args.label}_{int(time.time())}"
    path = bench_dir / f"{stem}.mp4"
    writer = None
    print(f"caught: recording {args.seconds:.0f}s as '{args.label}' -> {path}. Do the thing now.")
    t0 = time.time()
    frames = 0
    try:
        while time.time() - t0 < args.seconds:
            fr = src.read()
            if fr is None:
                time.sleep(0.01)
                continue
            if writer is None:
                h, w = fr.image.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            writer.write(fr.image)
            frames += 1
    finally:
        src.stop()
        if writer is not None:
            writer.release()
    mf = bench_dir / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {"items": []}
    manifest["items"].append({"file": path.name, "expect": [args.label] if args.label != "nothing" else [],
                              "reject": [h for h in args.reject.split(",") if h] if args.reject else []})
    mf.write_text(json.dumps(manifest, indent=1))
    print(f"caught: saved {frames} frames; manifest updated. Run: caught bench {bench_dir}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .config import ensure_models

    md = ensure_models()
    print(f"caught: models ready in {md}")
    return 0


def cmd_menubar(args: argparse.Namespace) -> int:
    from .menubar import run_menubar

    return run_menubar(Config.load(args.data))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="caught", description="A desk camera that learns your bad habits.")
    p.add_argument("--data", help="data directory (default ~/.caught or $CAUGHT_DATA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="watch the camera and serve the dashboard")
    r.add_argument("--camera", type=int)
    r.add_argument("--port", type=int)
    r.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    r.add_argument("--window", action="store_true", help="also show a live preview window")
    r.add_argument("--no-objects", action="store_true", help="skip YOLO (no phone/cup detection)")
    r.add_argument("--no-model", action="store_true", help="rules only, ignore the trained model")
    r.add_argument("--video", help="replay a video file instead of the camera")
    r.add_argument("--realtime", action="store_true", help="replay video at real speed")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="dashboard only, no camera")
    s.add_argument("--port", type=int)
    s.add_argument("--open", action="store_true")
    s.set_defaults(fn=cmd_serve)

    c = sub.add_parser("calibrate", help="record your upright posture baseline")
    c.add_argument("--seconds", type=float, default=10.0)
    c.set_defaults(fn=cmd_calibrate)

    t = sub.add_parser("train", help="train the personal model on stored history")
    t.add_argument("--days", type=float, default=21.0)
    t.add_argument("--epochs", type=int, default=12)
    t.set_defaults(fn=cmd_train)

    rp = sub.add_parser("report", help="build the weekly report")
    rp.add_argument("--end", help="last day of the week, YYYY-MM-DD")
    rp.add_argument("--roast", action="store_true", help="ask Claude for the paragraph (aggregates only)")
    rp.add_argument("--open", action="store_true")
    rp.set_defaults(fn=cmd_report)

    sm = sub.add_parser("simulate", help="generate synthetic history for a demo")
    sm.add_argument("--days", type=int, default=7)
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--wipe", action="store_true")
    sm.add_argument("--no-features", action="store_true")
    sm.set_defaults(fn=cmd_simulate)

    b = sub.add_parser("bench", help="backtest the detectors on labeled clips and photos")
    b.add_argument("dir", nargs="?", default="bench", help="folder with media + manifest.json")
    b.add_argument("--only", help="substring filter on file names")
    b.add_argument("--no-objects", action="store_true")
    b.add_argument("--trace", action="store_true", help="write per-frame feature/score CSVs next to the media")
    b.set_defaults(fn=cmd_bench)

    rc = sub.add_parser("record", help="record a labeled clip from the camera for the bench")
    rc.add_argument("label", help="habit key you are about to perform, or 'nothing' for a clean negative")
    rc.add_argument("--seconds", type=float, default=20.0)
    rc.add_argument("--reject", help="comma-separated habits that must NOT fire", default="")
    rc.add_argument("--dir", default="bench")
    rc.set_defaults(fn=cmd_record)

    sub.add_parser("models", help="download the pretrained weights (done automatically on first run)").set_defaults(fn=cmd_models)
    sub.add_parser("cameras", help="list cameras").set_defaults(fn=cmd_cameras)
    sub.add_parser("status", help="print a summary").set_defaults(fn=cmd_status)
    w = sub.add_parser("wipe", help="delete all data")
    w.add_argument("--yes", action="store_true")
    w.set_defaults(fn=cmd_wipe)
    sub.add_parser("menubar", help="run as a macOS menu bar app").set_defaults(fn=cmd_menubar)
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"run", "serve", "calibrate", "train", "report", "simulate", "bench", "record", "cameras", "status",
             "wipe", "menubar", "models"}
    if "-h" not in argv and "--help" not in argv:
        # bare `caught` (optionally with --data DIR) just starts everything
        i = 0
        while i < len(argv) and argv[i] == "--data" and i + 1 < len(argv):
            i += 2
        if i >= len(argv) or argv[i] not in known:
            argv = argv[:i] + ["run", "--open"] + argv[i:]
    args = build_parser().parse_args(argv)
    sys.exit(args.fn(args))
