"""
EXPERIMENT 11 — Module 2: Prediction Logic
===========================================
Loads the PyCaret model saved by train.py.
Supports two input modes:
  - predict_image_path(path)   → from a file path (used by main.py)
  - predict_image_bytes(bytes) → from raw upload bytes (used by API endpoint)

HOG parameters MUST match train.py exactly — same resize, same hog() args.
PyCaret 3.x returns column "prediction_label"; older returns "Label".
This file handles both safely.
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

MODEL_PATH = "app/model/car_model"   # PyCaret saves as car_model.pkl

_model = None   # module-level cache — loaded once, reused for every request


# ── Model Loader ───────────────────────────────────────────────────────────────
def get_model():
    """
    Lazy-load the PyCaret model once and cache it.
    Raises a clear error if train.py has not been run yet.
    """
    global _model
    if _model is None:
        pkl = MODEL_PATH + ".pkl"
        if not os.path.exists(pkl):
            raise FileNotFoundError(
                f"Model not found at '{pkl}'.\n"
                f"Run: python app/train.py  — then try again."
            )

        from pycaret.classification import load_model

        _model = load_model(MODEL_PATH, verbose=False)
        print(f"[predict] Model loaded from {pkl}")

    return _model


# ── Feature Extractor — MUST match train.py ───────────────────────────────────
def extract_hog(img_gray: np.ndarray) -> np.ndarray:
    """
    Resize grayscale image to (64, 32) and compute HOG features.
    Returns shape: (1, 756) — ready for predict_model()
    """
    img_resized = cv2.resize(img_gray, (64, 32))

    features = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )

    return features.reshape(1, -1)


# ── Core Prediction ───────────────────────────────────────────────────────────
def _run_prediction(img_gray: np.ndarray) -> dict:
    """
    Internal: extract HOG → run PyCaret predict_model → parse result.
    """

    from pycaret.classification import predict_model

    model = get_model()

    features = extract_hog(img_gray)
    data = pd.DataFrame(features)

    pred_df = predict_model(model, data=data, verbose=False)

    # ── PyCaret version-safe columns ──────────────────────────────────────────
    if "prediction_label" in pred_df.columns:
        label_col = "prediction_label"
        score_col = "prediction_score"

    elif "Label" in pred_df.columns:
        label_col = "Label"
        score_col = "Score"

    else:
        label_col = pred_df.columns[-2]
        score_col = pred_df.columns[-1]

    # ── Safe label parsing ────────────────────────────────────────────────────
    raw_label = pred_df[label_col].iloc[0]

    try:
        label = int(raw_label)

    except Exception:
        label = 1 if str(raw_label).lower() in [
            "1",
            "car",
            "true",
            "positive"
        ] else 0

    # ── Safe confidence parsing ───────────────────────────────────────────────
    if score_col in pred_df.columns:
        score = float(pred_df[score_col].iloc[0])
    else:
        score = 0.5

    # PyCaret sometimes returns probability of opposite class.
    confidence = max(score, 1 - score)

    return {
        "label": label,
        "result": "Car Detected" if label == 1 else "No Car",
        "confidence": round(confidence * 100, 2)
    }


# ── Public API ────────────────────────────────────────────────────────────────
def predict_image_path(img_path: str) -> dict:
    """
    Predict from a file path.
    Used by: app/main.py (after saving uploaded file to disk)
    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image at path: {img_path}")

    return _run_prediction(img)


def predict_image_bytes(img_bytes: bytes) -> dict:
    """
    Predict from raw bytes (e.g., from an HTTP upload without saving to disk).
    Used by: app/main.py /api/predict endpoint
    """

    nparr = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Cannot decode image bytes — unsupported format?")

    return _run_prediction(img)


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "app/uploads/test-1.pgm"
    )

    print(f"Testing prediction on: {path}")

    result = predict_image_path(path)

    print(f"Result     : {result['result']}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']}%")