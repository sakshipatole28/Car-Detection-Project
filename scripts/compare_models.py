"""Compare HOG, COCO-pretrained SSDlite (no fine-tuning) and the fine-tuned CNN.

    python scripts/compare_models.py [--stanford-images 400]

Writes reports/comparison.json. Every model is scored the same way:
* Stanford Cars *test* split (any viewing angle): AP@0.5 and P/R/F1 at the
  model's own threshold (each threshold was chosen on validation data).
* UIUC test sets: official protocol, via cardet.evaluate.run_split.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cardet.cnn import CnnDetector, build_model  # noqa: E402
from cardet.cnn_data import StanfordZip, resize_max  # noqa: E402
from cardet.config import MODEL_PATH, REPORTS_DIR, load_params  # noqa: E402
from cardet.evaluate import CNN_MODEL_PATH, run_split  # noqa: E402
from cardet.features import to_gray  # noqa: E402
from cardet.train_cnn import detection_ap, pr_at_threshold  # noqa: E402

log = logging.getLogger("compare")


def stanford_scores(det, kind: str, items) -> dict:
    preds, gts = [], []
    for img, gt in items:
        res = det.detect(to_gray(img) if kind == "hog" else img, threshold=0.05)
        boxes = np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in res.boxes], np.float32)
        preds.append((boxes.reshape(-1, 4), np.array([b.score for b in res.boxes])))
        gts.append(gt)
    at = pr_at_threshold(preds, gts, det.config.threshold)
    return {"ap50": detection_ap(preds, gts)["ap50"], **at}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stanford-images", type=int, default=400)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = load_params()

    cnn = CnnDetector.load(CNN_MODEL_PATH)
    coco_thr = cnn.metadata["validation"]["baseline_coco"]["combined"]["best_threshold"]
    models = {
        "hog": ("hog", joblib.load(MODEL_PATH)),
        "ssdlite_coco_pretrained": ("cnn", CnnDetector(build_model(True), threshold=coco_thr)),
        "ssdlite_fine_tuned": ("cnn", cnn),
    }

    test = StanfordZip(ROOT / p["cnn"]["zip"], "test")
    idx = sorted(random.Random(0).sample(range(len(test)), args.stanford_images))
    items = [resize_max(*test.load(i), 640) for i in idx]

    out_dir = REPORTS_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (kind, det) in models.items():
        log.info("== %s", name)
        st = stanford_scores(det, kind, items)
        log.info("   stanford %s", st)
        uiuc = {}
        for multiscale in (False, True):
            key, m, *_ = run_split(det, p, multiscale, kind, out_dir)
            uiuc[key] = {
                k: m[k]
                for k in ("precision", "recall", "f1", "average_precision", "latency_ms_mean")
            }
            uiuc[key]["eer_recall"] = m["eer"]["recall"]
        results[name] = {"stanford_test": st, "uiuc": uiuc}
    results["_note"] = (
        f"Stanford: {args.stanford_images} random test images (seed 0); "
        "UIUC: full official test sets. Thresholds chosen on validation data."
    )
    (REPORTS_DIR / "comparison.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote reports/comparison.json")


if __name__ == "__main__":
    main()
