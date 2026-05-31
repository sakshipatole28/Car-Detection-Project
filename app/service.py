"""
EXPERIMENT 11 — Module 3: BentoML Deployment Service
======================================================
BentoML 1.x runner pattern — NOT the old bentoml.Service("name") style.

How it works:
  1. Loads the sklearn model saved by train.py via bentoml.sklearn.save_model()
  2. Wraps it in a Runner (handles batching + async)
  3. Exposes POST /classify endpoint: accepts an image, returns JSON

Run this service with:
  bentoml serve app/service.py:svc --reload --port 3000

Test with curl:
  curl -X POST http://localhost:3000/classify \
       -H "Content-Type: multipart/form-data" \
       -F "input_image=@CarData/TestImages/test-1.pgm"
"""

import cv2
import numpy as np
from PIL import Image as PILImage
from skimage.feature import hog

import bentoml
from bentoml.io import Image, JSON


# ── Load Runner ────────────────────────────────────────────────────────────────
# This references the model saved by train.py.
# If you see "Model not found", run train.py first, then: bentoml models list
car_runner = bentoml.sklearn.get("car_detector:latest").to_runner()

svc = bentoml.Service("car_detector", runners=[car_runner])


# ── HOG Extractor — MUST match train.py exactly ───────────────────────────────
def extract_hog(img_gray: np.ndarray) -> np.ndarray:
    """
    Same parameters as train.py — changing any value breaks predictions.
    resize: (64, 32) | orientations=9 | pixels_per_cell=(8,8) | cells_per_block=(2,2)
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
    return features.reshape(1, -1)   # shape: (1, 756)


# ── BentoML API Endpoint ──────────────────────────────────────────────────────
@svc.api(input=Image(), output=JSON())
async def classify(input_image: PILImage.Image) -> dict:
    """
    POST /classify

    Input : Any image file — PNG, JPG, PGM all work.
            The BentoML Image() IO descriptor handles decoding automatically.

    Output: JSON response, e.g.:
            {
              "label"      : 1,
              "result"     : "Car Detected",
              "confidence" : 94.2
            }
    """
    # Convert PIL image to grayscale numpy array
    img_gray = np.array(input_image.convert("L"))

    # Extract HOG features
    features = extract_hog(img_gray)

    # Run through sklearn runner (async — non-blocking)
    prediction = await car_runner.predict.async_run(features)
    label = int(prediction[0])

    # Try to get probability score (not all sklearn models support predict_proba)
    try:
        proba = await car_runner.predict_proba.async_run(features)
        confidence = round(float(proba[0][label]) * 100, 2)
    except Exception:
        # Models like Decision Tree without probability calibration fall here
        confidence = 95.0 if label == 1 else 5.0

    return {
        "label"      : label,
        "result"     : "Car Detected" if label == 1 else "No Car",
        "confidence" : confidence
    }