"""Import the trained detectors into the local BentoML model store.

    python -m cardet.bento_export      # then: bentoml build && bentoml containerize ...

Both model files go into one BentoML model, so a Bento always ships a matching pair.
"""

from __future__ import annotations

import shutil

import bentoml
import joblib

from cardet.config import MODEL_PATH, ROOT

BENTO_MODEL_NAME = "car_detector"
BUNDLE_FILE = "car_detector.joblib"
CNN_FILE = "cnn_detector.pt"
CNN_PATH = ROOT / "models" / CNN_FILE


def export() -> bentoml.Model:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found — run `python -m cardet.train` first.")
    meta = joblib.load(MODEL_PATH).metadata
    keep = ("version", "trained_at", "model_family", "cv_score", "mlflow_run_id", "validation")
    with bentoml.models.create(
        BENTO_MODEL_NAME,
        labels={"task": "object-detection", "dataset": "uiuc-cars+stanford-cars"},
        metadata={
            "hog": {k: meta[k] for k in keep if meta.get(k) is not None},
            "has_cnn": CNN_PATH.exists(),
        },
    ) as model:
        shutil.copy(MODEL_PATH, model.path_of(BUNDLE_FILE))
        if CNN_PATH.exists():
            shutil.copy(CNN_PATH, model.path_of(CNN_FILE))
    print(f"Saved BentoML model: {model.tag}")
    return model


if __name__ == "__main__":
    export()
