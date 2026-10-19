"""BentoML inference service for the car detectors.

python -m cardet.bento_export          # copy both models into the BentoML model store
bentoml serve service:CarDetectorService --port 3000
bentoml build && bentoml containerize car_detector_service:latest

curl -F image=@app/static/samples/angle-4-front.jpg localhost:3000/detect
curl -F image=@CarData/TestImages/test-1.pgm -F model=hog localhost:3000/detect
"""

from __future__ import annotations

import os
from typing import Literal

import bentoml
import joblib
import numpy as np
from PIL.Image import Image
from pydantic import Field

from cardet.bento_export import BENTO_MODEL_NAME, BUNDLE_FILE, CNN_FILE
from cardet.features import to_gray


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 30, "max_concurrency": 16},
    http={"cors": {"enabled": True, "access_control_allow_origins": ["*"]}},
)
class CarDetectorService:
    model = bentoml.models.BentoModel(f"{BENTO_MODEL_NAME}:latest")

    def __init__(self) -> None:
        self.detectors = {"hog": joblib.load(self.model.path_of(BUNDLE_FILE))}
        cnn_path = self.model.path_of(CNN_FILE)
        if os.path.exists(cnn_path):
            from cardet.cnn import CnnDetector

            self.detectors["cnn"] = CnnDetector.load(cnn_path)

    @bentoml.api
    def detect(
        self,
        image: Image,
        model: Literal["cnn", "hog"] = "cnn",
        threshold: float | None = Field(default=None, ge=0.05, le=0.99),
    ) -> dict:
        """Return car bounding boxes (x, y, w, h, score) for one image."""
        det = self.detectors.get(model) or self.detectors["hog"]
        name = model if model in self.detectors else "hog"
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # PIL RGB -> OpenCV BGR
        img = to_gray(arr) if name == "hog" else np.ascontiguousarray(arr)
        res = det.detect(img, threshold=threshold)
        return {
            "model": name,
            "count": len(res.boxes),
            "detections": [b.to_dict() for b in res.boxes],
            "image_width": res.image_size[0],
            "image_height": res.image_size[1],
            "resize_factor": round(res.resize_factor, 4),
            "latency_ms": round(res.latency_ms, 1),
            "bento_model": str(self.model.tag),
        }

    @bentoml.api
    def model_info(self) -> dict:
        return {"tag": str(self.model.tag), **{n: d.metadata for n, d in self.detectors.items()}}
