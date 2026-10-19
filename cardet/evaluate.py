"""Evaluate a detector on the official UIUC test sets (and Stanford Cars for the CNN).

    python -m cardet.evaluate --model hog      (or: dvc repro evaluate)
    python -m cardet.evaluate --model cnn      (or: dvc repro evaluate_cnn)

* Single-scale set (170 images, 200 cars): window fixed at 100x40.
* Multi-scale set (108 images, 139 cars): full image pyramid.

Scoring follows the official Java evaluators (see cardet/metrics.py). Reported:
precision / recall / F1 at the configured threshold, recall at the equal-error
rate (the number quoted in the UIUC literature), and average precision.
Writes reports/metrics.json, PR curves, Java-compatible foundLocations files and
example visualisations; optionally logs to the training MLflow run and enforces
the quality gate from params.yaml (`--gate`). HOG reports go to reports/, CNN
reports to reports/cnn/. The CNN is additionally scored with AP@0.5 on the
held-out Stanford Cars test split (all viewing angles) when the archive exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import joblib
import matplotlib
import numpy as np

from cardet.config import MODEL_PATH, REPORTS_DIR, ROOT, Params, load_params
from cardet.data import parse_locations, test_image_path, write_locations
from cardet.features import read_image
from cardet.metrics import average_precision, equal_error_rate, score_detections, sweep_thresholds
from cardet.visualize import draw_boxes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("cardet.evaluate")
SWEEP = np.round(np.arange(0.05, 1.0, 0.01), 2)


CNN_MODEL_PATH = ROOT / "models" / "cnn_detector.pt"


def load_detector(kind: str):
    if kind == "hog":
        return joblib.load(MODEL_PATH)
    from cardet.cnn import CnnDetector

    return CnnDetector.load(CNN_MODEL_PATH)


def uiuc_location(b, multiscale: bool, kind: str) -> tuple[int, ...]:
    """Box -> UIUC (row, col[, width]) of the top-left of the reference window.

    HOG windows *are* UIUC windows (100x40 at scale 1, 2.5:1 at every scale), so
    their corners are used as-is. CNN boxes are tight around the car, so the
    reference window is centred on the box: 100x40 for the single-scale set,
    box-width x 0.4*width for the multi-scale set.
    """
    if kind == "hog":
        return (b.y, b.x, b.w) if multiscale else (b.y, b.x)
    cx, cy = b.x + b.w / 2, b.y + b.h / 2
    if multiscale:
        return (round(cy - 0.2 * b.w), b.x, b.w)
    return (round(cy - 20), round(cx - 50))


def run_split(det, p: Params, multiscale: bool, kind: str, out_dir):
    key = "scale" if multiscale else "single"
    test_dir = p.path("test_scale_dir" if multiscale else "test_dir")
    truth = parse_locations(p.path("true_locations_scale" if multiscale else "true_locations"))
    scales = (det.config.scales if multiscale else (1.0,)) if kind == "hog" else None

    scored, latencies, raw = {}, [], {}
    for n in sorted(truth):
        img = read_image(test_image_path(test_dir, n))
        # max_side=0: never downscale benchmark images, coordinates must stay exact.
        res = det.detect(img, threshold=float(SWEEP[0]), scales=scales, max_side=0)
        latencies.append(res.latency_ms)
        raw[n] = (img, res.boxes)
        # UIUC format: (row, col) or (row, col, width) of the top-left corner.
        scored[n] = [(uiuc_location(b, multiscale, kind), b.score) for b in res.boxes]

    thr = det.config.threshold
    found = {n: [loc for loc, s in d if s >= thr] for n, d in scored.items()}
    at_thr = score_detections(truth, found, multiscale)
    rows = sweep_thresholds(truth, scored, multiscale, SWEEP)
    best = max(rows, key=lambda r: r["f1"])
    metrics = {
        **at_thr.as_dict(),
        "threshold": thr,
        "eer": equal_error_rate(rows),
        "average_precision": round(average_precision(rows), 4),
        "best_f1": round(best["f1"], 4),
        "best_f1_threshold": best["threshold"],
        "n_images": len(truth),
        "latency_ms_mean": round(float(np.mean(latencies)), 1),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1),
    }
    suffix = "_Scale" if multiscale else ""
    write_locations(out_dir / f"foundLocations{suffix}.txt", found, max(truth) + 1)
    log.info(
        "[%s/%s] P=%.3f R=%.3f F1=%.3f | EER recall=%.3f | AP=%.3f | %.0f ms/img",
        kind,
        key,
        at_thr.precision,
        at_thr.recall,
        at_thr.f1,
        metrics["eer"]["recall"],
        metrics["average_precision"],
        metrics["latency_ms_mean"],
    )
    return key, metrics, rows, raw, thr


def plot_pr(curves: dict, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (rows, thr) in curves.items():
        r = [x["recall"] for x in rows]
        pr = [x["precision"] for x in rows]
        ax.plot(r, pr, label=name)
        op = min(rows, key=lambda x: abs(x["threshold"] - thr))
        ax.scatter([op["recall"]], [op["precision"]], zorder=3)
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8, label="EER line")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
        title="UIUC test sets — detection PR curve (dot = operating point)",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def save_examples(raw: dict, thr: float, out_dir, k: int = 6):
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for n in sorted(raw)[:k]:
        img, boxes = raw[n]
        cv2.imwrite(
            str(out_dir / f"test-{n}.png"), draw_boxes(img, [b for b in boxes if b.score >= thr])
        )


def evaluate_stanford(det, p: Params) -> dict | None:
    """AP@0.5 on the held-out Stanford Cars test split (all viewing angles)."""
    zip_path = ROOT / p["cnn"]["zip"]
    if not zip_path.exists():
        log.warning("Stanford archive %s not found; skipping that test set.", zip_path)
        return None
    from cardet.cnn_data import StanfordZip
    from cardet.train_cnn import evaluate_split

    test = StanfordZip(zip_path, "test")
    # P/R/F1 are reported at the deployed threshold (chosen on validation, not here).
    m = evaluate_split(det, test, p["cnn"]["test_images"] or None, threshold=det.threshold)
    log.info("[cnn/stanford-test] %s", m)
    return m


def evaluate(p: Params, kind: str = "hog", use_mlflow: bool = True) -> dict:
    det = load_detector(kind)
    out_dir = REPORTS_DIR if kind == "hog" else REPORTS_DIR / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    metrics, curves = {}, {}
    for multiscale in (False, True):
        key, m, rows, raw, thr = run_split(det, p, multiscale, kind, out_dir)
        metrics[key] = m
        curves[f"{key}-scale"] = (rows, thr)
        save_examples(raw, thr, out_dir / "examples" / key)
    metrics["validation"] = det.metadata.get("validation", {})
    if kind == "cnn":
        metrics["stanford_test"] = evaluate_stanford(det, p)
    metrics["model"] = {
        k: det.metadata.get(k)
        for k in (
            "model_family",
            "model_params",
            "cv_score",
            "trained_at",
            "n_hard_negatives",
            "iterations",
            "mlflow_run_id",
        )
        if k in det.metadata
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    plot_pr(curves, out_dir / "pr_curve.png")
    log.info(
        "Evaluation finished in %.0fs -> %s",
        time.perf_counter() - t0,
        (out_dir / "metrics.json").relative_to(ROOT),
    )

    run_id = det.metadata.get("mlflow_run_id")
    if use_mlflow and run_id:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        with mlflow.start_run(run_id=run_id):
            for key in ("single", "scale"):
                m = metrics[key]
                mlflow.log_metrics(
                    {
                        f"test_{key}_{k}": m[k]
                        for k in ("precision", "recall", "f1", "average_precision")
                    }
                )
                mlflow.log_metric(f"test_{key}_eer_recall", m["eer"]["recall"])
            if metrics.get("stanford_test"):
                st = metrics["stanford_test"]
                mlflow.log_metrics(
                    {"test_stanford_ap50": st["ap50"], "test_stanford_best_f1": st["best_f1"]}
                )
            mlflow.log_artifact(str(out_dir / "pr_curve.png"))
            mlflow.log_artifact(str(out_dir / "metrics.json"))
    return metrics


def quality_gate(metrics: dict, p: Params, kind: str = "hog") -> list[str]:
    g = p["quality_gate"]
    failures = []
    if kind == "cnn":
        if metrics["single"]["f1"] < g["cnn_min_uiuc_f1"]:
            failures.append(
                f"UIUC single-scale F1 {metrics['single']['f1']} < {g['cnn_min_uiuc_f1']}"
            )
        st = metrics.get("stanford_test")
        if st and st["ap50"] < g["cnn_min_stanford_ap50"]:
            failures.append(f"Stanford test AP50 {st['ap50']} < {g['cnn_min_stanford_ap50']}")
        return failures
    if metrics["single"]["f1"] < g["min_test_f1"]:
        failures.append(f"single-scale F1 {metrics['single']['f1']} < {g['min_test_f1']}")
    auc = metrics["validation"].get("val_roc_auc", 0)
    if auc < g["min_val_roc_auc"]:
        failures.append(f"validation ROC-AUC {auc} < {g['min_val_roc_auc']}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--gate", action="store_true", help="exit 1 if the quality gate fails")
    ap.add_argument("--model", choices=["hog", "cnn"], default="hog")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = load_params()
    metrics = evaluate(p, kind=args.model, use_mlflow=not args.no_mlflow)
    if args.gate:
        failures = quality_gate(metrics, p, args.model)
        if failures:
            log.error("QUALITY GATE FAILED: %s", "; ".join(failures))
            sys.exit(1)
        log.info("Quality gate passed.")


if __name__ == "__main__":
    main()
