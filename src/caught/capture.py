"""Camera capture on a background thread with adaptive throttling.

The camera is opened only when a CameraSource is started, and released as soon
as it is stopped. Frames live in memory only; nothing is written to disk here.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray      # BGR, downscaled to process_width
    ts: float              # wall-clock seconds
    ts_ms: int             # monotonic milliseconds (MediaPipe VIDEO mode needs this)
    full_shape: tuple[int, int]


class FrameSource:
    """Interface: read() returns the next Frame or None when exhausted."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> Frame | None: ...
    def set_target_fps(self, fps: float) -> None: ...


class CameraSource(FrameSource):
    def __init__(self, index: int = 0, process_width: int = 640):
        self.index = index
        self.process_width = process_width
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._target_fps = 8.0
        self._t0 = time.monotonic()
        self._last_ms = -1
        self.error: str | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.error = f"could not open camera {self.index}"
            raise RuntimeError(self.error)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_target_fps(self, fps: float) -> None:
        self._target_fps = max(0.2, float(fps))

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            t_start = time.monotonic()
            ok, img = self._cap.read()
            if not ok or img is None:
                self.error = "camera read failed"
                time.sleep(0.2)
                continue
            self.error = None
            h, w = img.shape[:2]
            if w > self.process_width:
                scale = self.process_width / w
                img = cv2.resize(img, (self.process_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            ms = int((time.monotonic() - self._t0) * 1000)
            if ms <= self._last_ms:
                ms = self._last_ms + 1
            self._last_ms = ms
            with self._lock:
                self._latest = Frame(img, time.time(), ms, (h, w))
            # throttle: sleep off the remainder of the frame budget
            budget = 1.0 / self._target_fps
            elapsed = time.monotonic() - t_start
            if budget > elapsed:
                time.sleep(budget - elapsed)

    def read(self) -> Frame | None:
        with self._lock:
            f = self._latest
            self._latest = None
        return f


class VideoFileSource(FrameSource):
    """Replay a recorded video through the pipeline (for development without a camera)."""

    def __init__(self, path: str, process_width: int = 640, realtime: bool = False):
        self.path = path
        self.process_width = process_width
        self.realtime = realtime
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._i = 0
        self._t0 = time.time()

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def set_target_fps(self, fps: float) -> None:
        pass

    def read(self) -> Frame | None:
        assert self._cap is not None
        ok, img = self._cap.read()
        if not ok:
            return None
        self._i += 1
        h, w = img.shape[:2]
        if w > self.process_width:
            scale = self.process_width / w
            img = cv2.resize(img, (self.process_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        ms = int(self._i * 1000 / self._fps)
        if self.realtime:
            # pace by the wall clock: sleep only for whatever the previous step left over
            due = self._t0 + ms / 1000
            now = time.time()
            if due > now:
                time.sleep(due - now)
            ts = time.time()
        else:
            ts = self._t0 + ms / 1000
        return Frame(img, ts, ms, (h, w))


def list_cameras(max_index: int = 4) -> list[dict]:
    """Probe camera indices. Opens each briefly; only used by `caught cameras`."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, img = cap.read()
            if ok and img is not None:
                found.append({"index": i, "width": int(img.shape[1]), "height": int(img.shape[0])})
        cap.release()
    return found
