"""Multi-scale sliding-window car detector.

A patch classifier (HOG -> scaler -> calibrated linear model) is scanned over an
image pyramid. Scores are calibrated car probabilities, so one threshold works
across scales. Overlapping hits are merged with greedy non-maximum suppression.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from cardet.config import DetectConfig
from cardet.features import HogExtractor


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(float(self.score), 4)
        return d


@dataclass
class DetectionResult:
    boxes: list[Box]
    image_size: tuple[int, int]  # (width, height) of the analysed image
    resize_factor: float = 1.0  # analysed size / original size
    latency_ms: float = 0.0
    n_windows: int = 0
    meta: dict = field(default_factory=dict)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS. boxes: [N, 4] as (x1, y1, x2, y2). Returns kept indices, best first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T.astype(float)
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= iou_thr]
    return keep


class CarDetector:
    """Serializable bundle: feature extractor + classifier + detection defaults."""

    def __init__(
        self,
        extractor: HogExtractor,
        classifier,
        config: DetectConfig,
        metadata: dict | None = None,
    ):
        self.extractor = extractor
        self.classifier = classifier
        self.config = config
        self.metadata = metadata or {}

    # -- patch level ---------------------------------------------------------
    def predict_proba_features(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.empty(0)
        return self.classifier.predict_proba(feats)[:, 1]

    def classify_patch(self, img: np.ndarray) -> float:
        """Probability that a single (cropped) image is a car."""
        return float(self.predict_proba_features(self.extractor.patch(img)[None])[0])

    # -- image level ---------------------------------------------------------
    def _scan_scale(self, img: np.ndarray, s: float, stride: int):
        win_w, win_h = self.extractor.window
        H, W = img.shape
        sw, sh = int(round(W / s)), int(round(H / s))
        if sw < win_w or sh < win_h:
            return None
        interp = cv2.INTER_AREA if s > 1 else cv2.INTER_LINEAR
        scaled = img if (sw, sh) == (W, H) else cv2.resize(img, (sw, sh), interpolation=interp)
        feats, tl = self.extractor.windows(scaled, stride)
        if len(feats) == 0:
            return None
        x1 = tl[:, 0] * W / sw
        y1 = tl[:, 1] * H / sh
        boxes = np.stack([x1, y1, x1 + win_w * W / sw, y1 + win_h * H / sh], 1)
        return boxes, self.predict_proba_features(feats)

    def score_windows(self, img: np.ndarray, scales, stride: int):
        """Return candidate boxes [N, 4] (x1, y1, x2, y2) in image coords and scores.

        Pyramid levels run in parallel threads (OpenCV and BLAS release the GIL).
        """
        with ThreadPoolExecutor(max_workers=min(len(scales), os.cpu_count() or 1)) as pool:
            parts = [r for r in pool.map(lambda s: self._scan_scale(img, s, stride), scales) if r]
        if not parts:
            return np.empty((0, 4)), np.empty(0)
        return np.concatenate([b for b, _ in parts]), np.concatenate([p for _, p in parts])

    def detect(
        self,
        img: np.ndarray,
        threshold: float | None = None,
        scales=None,
        stride: int | None = None,
        nms_iou: float | None = None,
        max_side: int | None = None,
    ) -> DetectionResult:
        cfg = self.config
        threshold = cfg.threshold if threshold is None else threshold
        scales = cfg.scales if scales is None else scales
        stride = cfg.stride if stride is None else stride
        nms_iou = cfg.nms_iou if nms_iou is None else nms_iou
        max_side = cfg.max_side if max_side is None else max_side

        t0 = time.perf_counter()
        factor = 1.0
        if max_side and max(img.shape) > max_side:
            factor = max_side / max(img.shape)
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

        boxes, scores = self.score_windows(img, scales, stride)
        n_windows = len(scores)
        mask = scores >= threshold
        boxes, scores = boxes[mask], scores[mask]
        keep = nms(boxes, scores, nms_iou)
        out = [
            Box(
                int(round(b[0])),
                int(round(b[1])),
                int(round(b[2] - b[0])),
                int(round(b[3] - b[1])),
                float(scores[i]),
            )
            for i in keep
            for b in [boxes[i]]
        ]
        return DetectionResult(
            boxes=out,
            image_size=(img.shape[1], img.shape[0]),
            resize_factor=factor,
            latency_ms=(time.perf_counter() - t0) * 1000,
            n_windows=n_windows,
        )
