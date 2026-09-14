"""Object detection for phones, cups, bottles, food, and books via YOLO11n."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ensure_models, models_dir

COCO = {
    39: "bottle", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    64: "mouse", 65: "remote", 67: "cell phone", 73: "book", 76: "scissors", 77: "teddy bear",
}
FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "bowl",
        "fork", "knife", "spoon"}
DRINK = {"bottle", "cup"}
CLASS_IDS = sorted(COCO.keys())


@dataclass
class Detection:
    name: str
    conf: float
    x1: float  # normalized 0..1
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def contains(self, x: float, y: float, pad: float = 0.0) -> bool:
        return (self.x1 - pad) <= x <= (self.x2 + pad) and (self.y1 - pad) <= y <= (self.y2 + pad)


class ObjectDetector:
    def __init__(self, device: str | None = None, conf: float = 0.25, imgsz: int = 416):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(str(ensure_models(log=lambda *_: None) / "yolo11n.pt"))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.conf = conf
        self.imgsz = imgsz
        # warm up
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), verbose=False, device=self.device,
                           imgsz=imgsz, classes=CLASS_IDS, conf=conf)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        res = self.model.predict(bgr, verbose=False, device=self.device, imgsz=self.imgsz,
                                 classes=CLASS_IDS, conf=self.conf, half=False)
        out: list[Detection] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
            name = COCO.get(int(c))
            if not name:
                continue
            out.append(Detection(name, float(p), float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)))
        return out
