"""
EXPERIMENT 11 — Module 1: AutoML Training
==========================================
AutoML tool  : PyCaret (classification)
Feature type : HOG (Histogram of Oriented Gradients) — 756 features per image
               WHY: Raw 64×64=4096 features → PyCaret hangs for hours.
                    HOG gives compact, meaningful shape features in ~756 numbers.
Models tried : lr (Logistic Regression), dt (Decision Tree), rf (Random Forest)
               WHY: Only fast sklearn models — no boosting, no SVM, no neural nets.
MLflow       : Logs params + metrics to sqlite:///mlflow.db (local, no server needed)
BentoML      : Saves sklearn pipeline with bentoml.sklearn.save_model for service.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import mlflow
import bentoml

from skimage.feature import hog
from pycaret.classification import (
    setup, compare_models, finalize_model,
    pull, save_model
)

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DIR = "CarData/TrainImages"
MODEL_DIR  = "app/model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── HOG Feature Extractor ──────────────────────────────────────────────────────
def extract_hog(img_path: str) -> np.ndarray:
    """
    Resize image to 64×32, compute HOG features.
    Output shape: (756,)  ← fast for PyCaret to process
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")
    img = cv2.resize(img, (64, 32))          # width=64, height=32
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    return features                          # shape: (756,)


# ── Load Dataset ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  Loading UIUC Car Detection Dataset")
print("=" * 55)

X, y = [], []
files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pgm"))

for i, fname in enumerate(files):
    path = os.path.join(TRAIN_DIR, fname)
    try:
        feats = extract_hog(path)
        X.append(feats)
        y.append(1 if fname.startswith("pos") else 0)
    except Exception as e:
        print(f"  [SKIP] {fname}: {e}")

    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(files)} images...")

total   = len(X)
n_cars  = sum(y)
n_other = total - n_cars
print(f"\nDataset Summary:")
print(f"  Total samples : {total}")
print(f"  Car images    : {n_cars}")
print(f"  Non-car images: {n_other}")
print(f"  Feature size  : {len(X[0])} (HOG)")

df = pd.DataFrame(X)
df["target"] = y


# ── MLflow Experiment ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Starting MLflow Experiment")
print("=" * 55)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CarDetection")


# ── PyCaret AutoML ─────────────────────────────────────────────────────────────
print("\nRunning PyCaret AutoML (lr, dt, rf — 3-fold CV)...")
print("This takes ~2-4 minutes. Please wait...\n")

with mlflow.start_run(run_name="pycaret_automl") as run:

    # Log input parameters
    mlflow.log_param("dataset",      TRAIN_DIR)
    mlflow.log_param("n_samples",    total)
    mlflow.log_param("n_cars",       n_cars)
    mlflow.log_param("n_non_cars",   n_other)
    mlflow.log_param("feature_type", "HOG")
    mlflow.log_param("feature_size", len(X[0]))
    mlflow.log_param("models",       "lr,dt,rf")
    mlflow.log_param("cv_folds",     3)
    mlflow.log_param("session_id",   42)

    # PyCaret setup
    clf = setup(
        data        = df,
        target      = "target",
        session_id  = 42,       # for reproducibility
        fold        = 3,        # 3-fold CV — fast
        verbose     = False,
        html        = False     # no HTML output in terminal
    )

    # AutoML: compare 3 models, pick best by Accuracy
    best_model = compare_models(
        include = ["lr", "dt", "rf"],
        sort    = "Accuracy",
        verbose = False
    )

    # Pull results table
    results  = pull()
    best_row = results.iloc[0]

    # Safe column-name extraction (PyCaret versions differ slightly)
    def _get(col_options, default=0.0):
        for c in col_options:
            if c in results.columns:
                return float(best_row[c])
        return default

    acc  = _get(["Accuracy"])
    prec = _get(["Prec.",   "Precision"])
    rec  = _get(["Recall"])
    f1   = _get(["F1",      "F1 Score"])
    auc  = _get(["AUC"])
    kappa= _get(["Kappa"])

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("auc",       auc)
    mlflow.log_metric("kappa",     kappa)

    print(f"Best model  : {type(best_model).__name__}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC       : {auc:.4f}")

    # Finalize model on full training data (no held-out set)
    final_model = finalize_model(best_model)

    # ── Save via PyCaret (.pkl — used by predict.py) ───────────────────────────
    pycaret_path = os.path.join(MODEL_DIR, "car_model")
    save_model(final_model, pycaret_path)
    print(f"\nPyCaret model → {pycaret_path}.pkl")

    # ── Save via BentoML (used by service.py) ─────────────────────────────────
    bento_tag = bentoml.sklearn.save_model(
        name     = "car_detector",
        model    = final_model,
        metadata = {
            "accuracy"      : acc,
            "f1_score"      : f1,
            "feature_type"  : "HOG",
            "feature_size"  : len(X[0]),
            "mlflow_run_id" : run.info.run_id,
            "framework"     : "pycaret+sklearn",
            "dataset"       : "UIUC Car Detection"
        }
    )
    print(f"BentoML model → {bento_tag}")

    # Log BentoML tag back into MLflow
    mlflow.log_param("bento_model_tag", str(bento_tag))
    mlflow.log_param("mlflow_run_id",   run.info.run_id)

    print(f"\nMLflow run ID : {run.info.run_id}")


print("\n" + "=" * 55)
print("  ✅  TRAINING COMPLETE")
print("=" * 55)
print()
print("Next steps:")
print("  1. View MLflow UI  : mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("  2. List BentoML    : bentoml models list")
print("  3. Start API       : uvicorn app.main:app --reload --port 8000")
print("  4. Start BentoML   : bentoml serve app/service.py:svc --port 3000")
print()