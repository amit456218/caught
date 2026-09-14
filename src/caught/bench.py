"""Backtesting harness: run the perception + detector stack over labeled clips and images.

A bench directory holds media files plus a manifest.json:

    {"items": [
        {"file": "phone_lap.mp4", "expect": ["phone", "looking_down"], "reject": ["hand_to_face"]},
        {"file": "coffee.jpg",    "expect": ["drink"],                 "reject": ["hand_to_face", "eat"]}
    ]}

For each item we report the peak instantaneous score and the peak smoothed score of every
detector, whether an episode would have fired (video only), and pass/fail against the labels.
Nothing touches the real history: the store is in memory.

`caught record --label phone --seconds 20` captures a labeled clip from the camera into the
bench directory so the harness grows with the user's own footage.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calibration import load_calibration
from .config import HABIT_KEYS, HABITS, Config
from .detectors import build_detectors
from .episodes import EpisodeTracker
from .features import FeatureExtractor, FrameFeatures
from .perception.landmarks import LandmarkEngine
from .perception.objects import ObjectDetector

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


@dataclass
class ItemResult:
    file: str
    kind: str
    frames: int
    seconds: float
    present_frac: float
    peak_raw: dict[str, float]
    peak_smooth: dict[str, float]
    fired: dict[str, bool]
    expect: list[str]
    reject: list[str]
    passed: bool
    failures: list[str]
    expect_any: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)   # medians over present frames
    note: str = ""
    optional: bool = False


class Bench:
    def __init__(self, cfg: Config, use_objects: bool = True, process_width: int = 640, log=print):
        self.cfg = cfg
        self.log = log
        self.engine = LandmarkEngine()
        self.objects = ObjectDetector() if use_objects else None
        self.process_width = process_width
        self.detectors = build_detectors(cfg.detector_params)
        self.calibration = load_calibration(cfg.calibration_path)

    def close(self) -> None:
        self.engine.close()

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w > self.process_width:
            s = self.process_width / w
            img = cv2.resize(img, (self.process_width, int(h * s)), interpolation=cv2.INTER_AREA)
        return img

    def run_item(self, path: Path, expect: list[str], reject: list[str], fire_threshold: float = 0.5,
                 image_seconds: float = 24.0, image_fps: float = 4.0, expect_any: list[str] | None = None,
                 trace: Path | None = None) -> ItemResult:
        ext = path.suffix.lower()
        kind = "image" if ext in IMAGE_EXT else "video"
        extractor = FeatureExtractor(self.calibration)
        tracker = EpisodeTracker({k: d.temporal for k, d in self.detectors.items()})
        peak_raw = {k: 0.0 for k in HABIT_KEYS}
        peak_smooth = {k: 0.0 for k in HABIT_KEYS}
        fired = {k: False for k in HABIT_KEYS}
        feats: list[FrameFeatures] = []
        n = 0
        present = 0
        t0 = time.time()
        last_ms = 0
        expect_any = expect_any or []
        trace_rows: list[dict] = []

        def step(img: np.ndarray, ts: float, ts_ms: int, run_objects: bool) -> None:
            nonlocal n, present
            p = self.engine.process(img, ts_ms, want_hands=True)
            dets = self.objects.detect(img) if (self.objects is not None and run_objects) else None
            f = extractor.extract(img, p, dets, ts)
            n += 1
            present += int(f.present > 0)
            if f.present:
                feats.append(f)
            rules = {k: d.score(f) for k, d in self.detectors.items()}
            for k, v in rules.items():
                peak_raw[k] = max(peak_raw[k], v)
            for kind_, ep in tracker.update(f, rules, None, 0.0):
                if kind_ == "commit":
                    fired[ep.habit] = True
            for k, st in tracker.states.items():
                peak_smooth[k] = max(peak_smooth[k], st.smoothed if (not st.temporal.requires_present or f.present) else 0.0)
            if trace is not None:
                row = {"t": round(ts - t0, 2), "face": int(p.face is not None), "crop": int(p.face_from_crop),
                       "hands": len(p.hands), "pose": int(p.pose is not None),
                       "objs": "|".join(f"{d.name}:{d.conf:.2f}" for d in (dets or []))}
                row.update({k: round(v, 3) for k, v in f.to_dict().items()})
                row.update({"score_" + k: round(v, 3) for k, v in rules.items()})
                row.update({"smooth_" + k: round(st.smoothed, 3) for k, st in tracker.states.items()})
                trace_rows.append(row)

        if kind == "image":
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"cannot read {path}")
            img = self._resize(img)
            # feed the still image as a short clip so temporal logic behaves as it would live
            frames = int(image_seconds * image_fps)
            for i in range(frames):
                ts_ms = int(i * 1000 / image_fps)
                step(img, t0 + ts_ms / 1000, ts_ms, run_objects=(i % 3 == 0))
            seconds = image_seconds
        else:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            stride = max(1, int(round(fps / 8.0)))  # sample ~8 fps like the live pipeline
            i = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    ts_ms = int(i * 1000 / fps)
                    last_ms = ts_ms
                    step(self._resize(img), t0 + ts_ms / 1000, ts_ms, run_objects=((i // stride) % 3 == 0))
                i += 1
            cap.release()
            seconds = last_ms / 1000
        for kind_, ep in tracker.flush(t0 + seconds):
            if kind_ == "commit":
                fired[ep.habit] = True

        if trace is not None and trace_rows:
            import csv
            with trace.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trace_rows[0].keys()))
                w.writeheader()
                w.writerows(trace_rows)
        # pass/fail is judged on what the product would report: committed episodes
        failures = []
        if expect_any and not any(fired.get(k, False) for k in expect_any):
            failures.append("none of " + "/".join(expect_any) + " fired (" + ", ".join(f"{k} peak {peak_smooth.get(k, 0):.2f}" for k in expect_any) + ")")
        for k in expect:
            if k not in HABIT_KEYS:
                failures.append(f"unknown habit {k}")
            elif not fired[k]:
                failures.append(f"missed {k} (peak {peak_smooth[k]:.2f})")
        for k in reject:
            if k in HABIT_KEYS and fired[k]:
                failures.append(f"false {k} (peak {peak_smooth[k]:.2f})")
        med = {}
        if feats:
            arr = np.stack([f.to_vector() for f in feats])
            from .features import FEATURE_NAMES
            for j, name in enumerate(FEATURE_NAMES):
                med[name] = float(np.median(arr[:, j]))
        return ItemResult(path.name, kind, n, seconds, present / max(n, 1), peak_raw, peak_smooth, fired,
                          expect, reject, not failures, failures, expect_any, med,
                          note="" if present else "no person detected")


def load_manifest(bench_dir: Path) -> list[dict]:
    mf = bench_dir / "manifest.json"
    if mf.exists():
        items = json.loads(mf.read_text()).get("items", [])
    else:
        items = []
    known = {it["file"] for it in items}
    # unlabeled media in the folder still gets run, with no expectations
    for p in sorted(bench_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT and p.name not in known:
            items.append({"file": p.name, "expect": [], "reject": []})
    return items


def run_bench(cfg: Config, bench_dir: Path, only: str | None = None, use_objects: bool = True,
              log=print, trace: bool = False) -> list[ItemResult]:
    items = load_manifest(bench_dir)
    if only:
        items = [it for it in items if only in it["file"]]
    b = Bench(cfg, use_objects=use_objects, log=log)
    results = []
    try:
        for it in items:
            p = bench_dir / it["file"]
            if not p.exists():
                log(f"skip {it['file']}: missing")
                continue
            t = time.time()
            tr = (bench_dir / f"trace_{p.stem}.csv") if trace else None
            r = b.run_item(p, it.get("expect", []), it.get("reject", []), expect_any=it.get("expect_any"), trace=tr)
            r.optional = bool(it.get("optional"))
            if r.optional:
                r.note = (r.note + " " + str(it.get("note", ""))).strip()
            results.append(r)
            tag = "PASS" if r.passed else ("soft" if r.optional else "FAIL")
            log(f"{tag} {r.file:<34} {r.frames:>4} frames {time.time() - t:5.1f}s "
                f"present {r.present_frac:.0%}" + (f"  {'; '.join(r.failures)}" if r.failures else ""))
    finally:
        b.close()
    return results


def format_table(results: list[ItemResult], keys: list[str] | None = None) -> str:
    keys = keys or [k for k in HABIT_KEYS if k != "absent"]
    head = f"{'file':<30}" + "".join(f"{HABITS[k].icon or k[:3]:>6}" for k in keys)
    lines = [head, "-" * len(head)]
    for r in results:
        cells = []
        for k in keys:
            v = r.peak_smooth[k]
            mark = "*" if k in r.expect else ("!" if k in r.reject else ("?" if k in (r.expect_any or []) else " "))
            cells.append(f"{'#' if r.fired.get(k) else ' '}{v:4.2f}{mark}")
        lines.append(f"{r.file[:30]:<30}" + "".join(f"{c:>6}" for c in cells))
    lines.append("  # episode fired   * expected   ? any of these expected   ! should stay low   (number = peak smoothed score)")
    return "\n".join(lines)


def save_results(results: list[ItemResult], out: Path) -> None:
    out.write_text(json.dumps([r.__dict__ for r in results], indent=1))
