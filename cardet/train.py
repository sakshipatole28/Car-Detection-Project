"""Train the car / background patch classifier.

    python -m cardet.train            (or: dvc repro train)

Steps
  1. Load the 1,050 UIUC training patches (550 car, 500 background).
  2. Stratified train/validation split *before* augmentation, so a flipped copy
     of a validation image can never leak into training.
  3. Augment the training split with horizontal flips.
  4. Model selection: grid search over several model families with
     StratifiedGroupKFold (an image and its flip always share a fold).
  5. Hard-negative mining: scan rescaled background images with the selected
     model and add confident false positives as extra negatives.
  6. Probability calibration (sigmoid) so detector scores are probabilities.
  7. Hold-out validation report, then optional refit on train+val.
  8. Everything is logged to MLflow and the model is registered; the detector
     bundle is written to models/car_detector.joblib.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from datetime import UTC, datetime

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cardet import __version__
from cardet.config import MODEL_PATH, REPORTS_DIR, Params, load_params
from cardet.data import load_train_patches
from cardet.detector import CarDetector
from cardet.features import HogExtractor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("cardet.train")

ESTIMATORS = {
    "logreg": lambda seed: LogisticRegression(max_iter=5000, random_state=seed),
    "linear_svm": lambda seed: LinearSVC(dual="auto", max_iter=20000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(n_jobs=-1, random_state=seed),
}


def make_pipeline(family: str, seed: int) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", ESTIMATORS[family](seed))])


def augment(images, labels, groups, hflip: bool):
    if not hflip:
        return list(images), np.asarray(labels), np.asarray(groups)
    flipped = [cv2.flip(im, 1) for im in images]
    return (
        list(images) + flipped,
        np.concatenate([labels, labels]),
        np.concatenate([groups, groups]),
    )


def select_model(X, y, groups, p: Params):
    """Grid-search each candidate family with grouped CV; return the winner."""
    seed = p["data"]["seed"]
    metric = p["train"]["selection_metric"]
    cv = list(
        StratifiedGroupKFold(p["train"]["cv_folds"], shuffle=True, random_state=seed).split(
            X, y, groups
        )
    )
    results = []
    for family, grid in p["train"]["candidates"].items():
        search = GridSearchCV(
            make_pipeline(family, seed),
            {f"clf__{k}": v for k, v in grid.items()},
            scoring=metric,
            cv=cv,
            n_jobs=-1,
            refit=False,
        )
        search.fit(X, y)
        cvr = pd.DataFrame(search.cv_results_)
        best = cvr.loc[search.best_index_]
        results.append(
            {
                "family": family,
                "params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
                "cv_mean": float(best["mean_test_score"]),
                "cv_std": float(best["std_test_score"]),
            }
        )
        log.info(
            "  %-14s %s=%.4f ± %.4f  %s",
            family,
            metric,
            results[-1]["cv_mean"],
            results[-1]["cv_std"],
            results[-1]["params"],
        )
    results.sort(key=lambda r: r["cv_mean"], reverse=True)
    return results, cv


def build_classifier(family, params, seed, cv):
    est = make_pipeline(family, seed).set_params(**{f"clf__{k}": v for k, v in params.items()})
    return CalibratedClassifierCV(est, method="sigmoid", cv=cv)


def mine_hard_negatives(clf, extractor, neg_images, p: Params):
    """Scan upscaled background patches; return windows the model thinks are cars."""
    cfg = p["hard_negative_mining"]
    feats, scores = [], []
    for img in neg_images:
        for s in cfg["scales"]:
            big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            f, _ = extractor.windows(big, cfg["stride"])
            if len(f):
                feats.append(f)
                scores.append(clf.predict_proba(f)[:, 1])
    feats, scores = np.concatenate(feats), np.concatenate(scores)
    hard = np.argsort(-scores)[: cfg["max_new"]]
    hard = hard[scores[hard] >= 0.5]
    log.info("  scanned %d background windows, %d false positives kept", len(scores), len(hard))
    return feats[hard]


def patch_metrics(y_true, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "val_accuracy": accuracy_score(y_true, pred),
        "val_precision": precision_score(y_true, pred),
        "val_recall": recall_score(y_true, pred),
        "val_f1": f1_score(y_true, pred),
        "val_roc_auc": roc_auc_score(y_true, prob),
        "val_average_precision": average_precision_score(y_true, prob),
    }


def save_validation_plots(y_true, prob, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (prob >= 0.5).astype(int),
        display_labels=["background", "car"],
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Validation confusion matrix")
    RocCurveDisplay.from_predictions(y_true, prob, ax=axes[1], name="val")
    axes[1].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, prob, ax=axes[2], name="val")
    axes[2].set_title("Precision-recall")
    fig.tight_layout()
    path = out_dir / "validation.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def train(p: Params, use_mlflow: bool = True) -> CarDetector:
    seed = p["data"]["seed"]
    extractor = HogExtractor(p.hog)
    REPORTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    log.info("Loading training patches from %s", p.path("train_dir"))
    data = load_train_patches(p.path("train_dir"))
    idx = np.arange(len(data.labels))
    tr_idx, val_idx = train_test_split(
        idx, test_size=p["data"]["val_size"], stratify=data.labels, random_state=seed
    )
    log.info(
        "  %d patches (%d car / %d background) -> %d train, %d val",
        len(idx),
        data.labels.sum(),
        (data.labels == 0).sum(),
        len(tr_idx),
        len(val_idx),
    )

    tr_imgs, y_tr, g_tr = augment(
        [data.images[i] for i in tr_idx], data.labels[tr_idx], tr_idx, p["augment"]["hflip"]
    )
    X_tr = extractor.patches(tr_imgs)
    X_val = extractor.patches([data.images[i] for i in val_idx])
    y_val = data.labels[val_idx]
    log.info(
        "  HOG features: %d dims, %d training rows after augmentation", X_tr.shape[1], len(X_tr)
    )

    log.info(
        "Model selection (%s, %d-fold grouped CV)",
        p["train"]["selection_metric"],
        p["train"]["cv_folds"],
    )
    leaderboard, cv = select_model(X_tr, y_tr, g_tr, p)
    best = leaderboard[0]
    log.info("Selected: %s %s", best["family"], best["params"])

    clf = build_classifier(best["family"], best["params"], seed, cv).fit(X_tr, y_tr)

    n_mined = 0
    neg_tr = [im for im, y in zip(tr_imgs, y_tr, strict=True) if y == 0]
    for r in range(p["hard_negative_mining"]["rounds"]):
        log.info("Hard-negative mining round %d", r + 1)
        hard = mine_hard_negatives(clf, extractor, neg_tr, p)
        if len(hard) == 0:
            break
        n_mined += len(hard)
        new_groups = g_tr.max() + 1 + np.arange(len(hard))
        X_tr = np.concatenate([X_tr, hard])
        y_tr = np.concatenate([y_tr, np.zeros(len(hard), int)])
        g_tr = np.concatenate([g_tr, new_groups])
        cv = list(
            StratifiedGroupKFold(p["train"]["cv_folds"], shuffle=True, random_state=seed).split(
                X_tr, y_tr, g_tr
            )
        )
        clf = build_classifier(best["family"], best["params"], seed, cv).fit(X_tr, y_tr)

    val_prob = clf.predict_proba(X_val)[:, 1]
    val = {k: round(float(v), 4) for k, v in patch_metrics(y_val, val_prob).items()}
    log.info("Hold-out validation: %s", val)
    plot_path = save_validation_plots(y_val, val_prob, REPORTS_DIR)

    if p["train"]["refit_full"]:
        val_imgs, y_v, g_v = augment(
            [data.images[i] for i in val_idx], y_val, val_idx, p["augment"]["hflip"]
        )
        X_full = np.concatenate([X_tr, extractor.patches(val_imgs)])
        y_full = np.concatenate([y_tr, y_v])
        g_full = np.concatenate([g_tr, g_v])
        cv = list(
            StratifiedGroupKFold(p["train"]["cv_folds"], shuffle=True, random_state=seed).split(
                X_full, y_full, g_full
            )
        )
        clf = build_classifier(best["family"], best["params"], seed, cv).fit(X_full, y_full)
        log.info("Refit on train+val: %d rows", len(X_full))

    metadata = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_family": best["family"],
        "model_params": best["params"],
        "cv_metric": p["train"]["selection_metric"],
        "cv_score": round(best["cv_mean"], 4),
        "leaderboard": leaderboard,
        "validation": val,
        "n_hard_negatives": n_mined,
        "n_features": int(X_val.shape[1]),
        "sklearn_version": sklearn.__version__,
        "opencv_version": cv2.__version__,
    }

    detector = CarDetector(extractor, clf, p.detect, metadata)

    if use_mlflow:
        metadata["mlflow_run_id"] = log_to_mlflow(
            p, detector, X_val, leaderboard, val, n_mined, plot_path
        )

    joblib.dump(detector, MODEL_PATH, compress=3)
    (REPORTS_DIR / "train_metrics.json").write_text(
        json.dumps(
            {
                "cv_score": metadata["cv_score"],
                "model_family": best["family"],
                "n_hard_negatives": n_mined,
                **val,
            },
            indent=2,
        )
    )
    pd.DataFrame(leaderboard).to_csv(REPORTS_DIR / "leaderboard.csv", index=False)
    log.info("Saved detector bundle -> %s", MODEL_PATH.relative_to(MODEL_PATH.parents[1]))
    return detector


def log_to_mlflow(p, detector, X_val, leaderboard, val, n_mined, plot_path) -> str:
    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "car-detection"))
    meta = detector.metadata
    with mlflow.start_run(run_name=f"train-{meta['model_family']}") as run:
        mlflow.set_tags(
            {"stage": "train", "python": platform.python_version(), "cardet_version": __version__}
        )
        mlflow.log_params(
            {
                k: str(v)[:250]
                for k, v in flatten(p.raw).items()
                if not k.startswith("data.") or k.endswith(("val_size", "seed"))
            }
        )
        mlflow.log_params(
            {
                "selected_family": meta["model_family"],
                **{f"selected.{k}": v for k, v in meta["model_params"].items()},
            }
        )
        for row in leaderboard:
            mlflow.log_metric(f"cv_{row['family']}", row["cv_mean"])
        mlflow.log_metrics({"cv_best": meta["cv_score"], "n_hard_negatives": n_mined, **val})
        mlflow.log_artifact(str(plot_path))
        mlflow.log_dict({"leaderboard": leaderboard}, "leaderboard.json")
        info = mlflow.sklearn.log_model(
            detector.classifier,
            name="patch_classifier",
            signature=infer_signature(X_val[:5], detector.classifier.predict_proba(X_val[:5])),
            input_example=X_val[:2],
            serialization_format="cloudpickle",  # skops rejects the calibration classes
            registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "car-detector"),
        )
        log.info("MLflow run %s, model %s", run.info.run_id, info.model_uri)
        return run.info.run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", default=None, help="path to params.yaml")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = load_params(args.params) if args.params else load_params()
    train(p, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
