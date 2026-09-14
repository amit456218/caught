"""Train the personal habit model on stored features + labels."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..config import HABIT_KEYS
from ..features import FEATURE_DIM, FEATURE_NAMES
from ..store import Store
from .dataset import HZ, WINDOW, Windows, build_windows
from .net import HabitNet


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _metrics(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    out = {}
    p = 1 / (1 + np.exp(-logits))
    for i, k in enumerate(HABIT_KEYS):
        m = w[:, i] > 0
        if m.sum() < 10 or y[m, i].sum() < 2 or (1 - y[m, i]).sum() < 2:
            continue
        yi, pi = y[m, i], p[m, i]
        pred = pi >= 0.5
        tp = float((pred & (yi == 1)).sum())
        fp = float((pred & (yi == 0)).sum())
        fn = float((~pred & (yi == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        # rank-based AUC
        order = np.argsort(pi)
        ranks = np.empty(len(pi))
        ranks[order] = np.arange(1, len(pi) + 1)
        n_pos, n_neg = yi.sum(), (1 - yi).sum()
        auc = float((ranks[yi == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        out[k] = {"precision": round(prec, 3), "recall": round(rec, 3), "auc": round(auc, 3), "n": int(m.sum())}
    return out


def train(store: Store, model_path: Path, days: float = 21.0, epochs: int = 12, max_seconds: float = 240.0,
          log=print) -> dict:
    t0 = time.time()
    now = time.time()
    win = build_windows(store, now - days * 86400, now)
    if win is None or len(win.x) < 200:
        msg = {"ok": False, "reason": "not enough feature history yet", "n_windows": 0 if win is None else len(win.x)}
        log(f"train: {msg['reason']}")
        return msg
    log(f"train: {len(win.x)} windows, {win.n_human} human-labeled episodes")

    # time-ordered split so evaluation is on the most recent data
    n = len(win.x)
    cut = int(n * 0.85)
    order = np.argsort(win.t)
    tr, va = order[:cut], order[cut:]

    mean = win.x[tr].reshape(-1, FEATURE_DIM).mean(0)
    std = win.x[tr].reshape(-1, FEATURE_DIM).std(0) + 1e-4
    xn = (win.x - mean) / std

    # positive up-weighting per habit
    pos_w = np.ones(len(HABIT_KEYS), np.float32)
    for i in range(len(HABIT_KEYS)):
        m = win.w[tr, i] > 0
        pos = (win.y[tr, i][m] * win.w[tr, i][m]).sum()
        neg = ((1 - win.y[tr, i][m]) * win.w[tr, i][m]).sum()
        pos_w[i] = float(np.clip(neg / max(pos, 1.0), 1.0, 20.0))

    dev = _device()
    net = HabitNet(FEATURE_DIM, len(HABIT_KEYS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    X = torch.tensor(xn, dtype=torch.float32)
    Y = torch.tensor(win.y, dtype=torch.float32)
    Wt = torch.tensor(win.w, dtype=torch.float32)
    PW = torch.tensor(pos_w, device=dev)
    bs = 256
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            xb, yb, wb = X[idx].to(dev), Y[idx].to(dev), Wt[idx].to(dev)
            logits = net(xb)
            loss_el = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none",
                                                                           pos_weight=PW)
            loss = (loss_el * wb).sum() / wb.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tot += loss.item() * len(idx)
        log(f"train: epoch {ep + 1}/{epochs} loss {tot / max(len(perm), 1):.4f}")
        if time.time() - t0 > max_seconds:
            log("train: time budget reached")
            break

    net.eval()
    with torch.no_grad():
        lv = net(X[va].to(dev)).cpu().numpy() if len(va) else np.zeros((0, len(HABIT_KEYS)))
    metrics = _metrics(lv, win.y[va], win.w[va]) if len(va) else {}
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "mean": mean, "std": std, "habits": HABIT_KEYS, "features": FEATURE_NAMES,
        "window": WINDOW, "hz": HZ, "trained_at": time.time(), "n_windows": int(n),
        "n_human": int(win.n_human), "metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, model_path)
    result = {"ok": True, "n_windows": int(n), "n_human": int(win.n_human), "metrics": metrics,
              "seconds": round(time.time() - t0, 1), "device": str(dev)}
    store.record_model_run(result)
    store.log_event("train", result)
    log(f"train: saved {model_path} ({result['seconds']}s on {dev})")
    return result
