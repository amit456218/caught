"""Live scoring with the personal model over a ring buffer of recent features."""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..config import HABIT_KEYS
from ..features import FEATURE_DIM, FEATURE_NAMES
from .net import HabitNet


class ModelScorer:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.net: HabitNet | None = None
        self.mean = np.zeros(FEATURE_DIM, np.float32)
        self.std = np.ones(FEATURE_DIM, np.float32)
        self.window = 16
        self.hz = 2.0
        self.buf: deque[np.ndarray] = deque(maxlen=64)
        self.last_push = 0.0
        self.last_scores: dict[str, float] | None = None
        self.info: dict = {}
        self._lock = threading.Lock()
        self.reload()

    @property
    def ready(self) -> bool:
        return self.net is not None

    def reload(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            ck = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            return False
        if ck.get("features") != FEATURE_NAMES or ck.get("habits") != HABIT_KEYS:
            self.info = {"error": "checkpoint feature layout mismatch; retrain"}
            return False
        net = HabitNet(FEATURE_DIM, len(HABIT_KEYS))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        with self._lock:
            self.net = net
            self.mean = ck["mean"].astype(np.float32)
            self.std = ck["std"].astype(np.float32)
            self.window = int(ck["window"])
            self.hz = float(ck["hz"])
            self.buf = deque(maxlen=self.window * 2)
            self.info = {k: ck[k] for k in ("trained_at", "n_windows", "n_human", "metrics")}
        return True

    def push(self, ts: float, vec: np.ndarray) -> dict[str, float] | None:
        """Feed a feature vector; scores are recomputed at the model's cadence."""
        if self.net is None:
            return None
        if ts - self.last_push < 1.0 / self.hz:
            return self.last_scores
        self.last_push = ts
        self.buf.append(vec.astype(np.float32))
        if len(self.buf) < self.window:
            return None
        x = np.stack(list(self.buf)[-self.window:])
        x = (x - self.mean) / self.std
        with self._lock, torch.no_grad():
            logits = self.net(torch.from_numpy(x[None]).float())[0]
        p = torch.sigmoid(logits).numpy()
        self.last_scores = {k: float(p[i]) for i, k in enumerate(HABIT_KEYS)}
        return self.last_scores

    def reset(self) -> None:
        self.buf.clear()
        self.last_scores = None
