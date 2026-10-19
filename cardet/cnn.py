"""CNN car detector: SSDlite (MobileNetV3) fine-tuned for cars seen from any angle.

The COCO-pretrained head is kept (91 classes) and only the 'car' class is used,
so fine-tuning refines existing knowledge instead of starting a new head.
`CnnDetector.detect` has the same interface and output as the HOG detector.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from cardet.detector import Box, DetectionResult

COCO_CAR = 3


def build_model(pretrained: bool = True) -> torch.nn.Module:
    if pretrained:
        return ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    # Architecture only; weights come from our checkpoint (no download at serve time).
    return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """BGR or gray uint8 -> 3-channel *grayscale* float tensor CHW in [0, 1].

    The fine-tuning data (this Stanford Cars release and UIUC) is entirely
    grayscale, so colour inputs are converted too: training and serving then see
    the same distribution, and colour photos are not out-of-distribution.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)


class CnnDetector:
    kind = "cnn"

    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 0.5,
        max_side: int = 960,
        metadata: dict | None = None,
    ):
        self.model = model.eval()
        # Keep candidates down to 0.05 so the UI slider can filter client-side.
        self.model.score_thresh = 0.05
        self.model.detections_per_img = 100
        self.threshold = threshold
        self.max_side = max_side
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Path | str) -> CnnDetector:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(pretrained=False)
        model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
        return cls(model, ckpt["threshold"], ckpt.get("max_side", 960), ckpt.get("metadata", {}))

    def save(self, path: Path | str) -> None:
        # fp16 storage halves the file (~7 MB); weights are cast back to fp32 on load.
        state = {
            k: (v.half() if v.is_floating_point() else v)
            for k, v in self.model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state,
                "threshold": self.threshold,
                "max_side": self.max_side,
                "metadata": self.metadata,
            },
            path,
        )

    @property
    def config(self):  # same shape as the HOG detector's config for the API
        return type("Cfg", (), {"threshold": self.threshold, "max_side": self.max_side})()

    @torch.inference_mode()
    def predict(self, imgs: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        outs = self.model([to_tensor(im) for im in imgs])
        res = []
        for o in outs:
            keep = o["labels"] == COCO_CAR
            res.append((o["boxes"][keep].numpy(), o["scores"][keep].numpy()))
        return res

    def detect(
        self,
        img: np.ndarray,
        threshold: float | None = None,
        max_side: int | None = None,
        **_ignored,
    ) -> DetectionResult:
        threshold = self.threshold if threshold is None else threshold
        max_side = self.max_side if max_side is None else max_side
        t0 = time.perf_counter()
        factor = 1.0
        if max_side and max(img.shape[:2]) > max_side:
            factor = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        boxes, scores = self.predict([img])[0]
        order = np.argsort(-scores)
        out = [
            Box(
                int(round(b[0])),
                int(round(b[1])),
                int(round(b[2] - b[0])),
                int(round(b[3] - b[1])),
                float(s),
            )
            for b, s in zip(boxes[order], scores[order], strict=True)
            if s >= threshold
        ]
        return DetectionResult(
            boxes=out,
            image_size=(img.shape[1], img.shape[0]),
            resize_factor=factor,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
