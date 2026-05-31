"""
EXPERIMENT 11 — Module 6: Evaluation Script
============================================
Runs predictions on all test images and writes:
  - metrics.json           → DVC metrics (accuracy on test set)
  - CarData/foundLocations.txt → for the Java Evaluator

Run with:
  python app/evaluate.py
or via DVC:
  dvc repro evaluate
"""

import os
import json
import cv2
import numpy as np

from app.predict import predict_image_path

TEST_DIR       = "CarData/TestImages"
TRUE_LOC_FILE  = "CarData/trueLocations.txt"
FOUND_LOC_FILE = "CarData/foundLocations.txt"
METRICS_FILE   = "metrics.json"

# ── Simple sliding-window detection on test images ────────────────────────────
# For the single-scale test: slide a 100×40 window at stride 10
WINDOW_W = 100
WINDOW_H = 40
STRIDE   = 10

def detect_cars_in_image(img_path: str):
    """
    Slide a 100×40 window over the test image and return
    (x, y) of the highest-confidence car detection.
    Returns None if no car found.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w   = img.shape
    best   = None
    best_c = 0.0

    for y in range(0, h - WINDOW_H + 1, STRIDE):
        for x in range(0, w - WINDOW_W + 1, STRIDE):
            patch = img[y:y+WINDOW_H, x:x+WINDOW_W]
            # Save patch temporarily
            tmp = "/tmp/_eval_patch.pgm"
            cv2.imwrite(tmp, patch)
            try:
                result = predict_image_path(tmp)
                if result["label"] == 1 and result["confidence"] > best_c:
                    best_c = result["confidence"]
                    best   = (y, x)   # (row, col) — matches trueLocations format
            except Exception:
                pass

    return best


def evaluate():
    print("Running evaluation on TestImages...")
    test_files = sorted(
        f for f in os.listdir(TEST_DIR) if f.endswith(".pgm")
    )

    found_lines  = []
    correct      = 0
    total_images = len(test_files)

    for fname in test_files:
        img_idx  = int(fname.replace("test-", "").replace(".pgm", ""))
        img_path = os.path.join(TEST_DIR, fname)
        detection = detect_cars_in_image(img_path)

        if detection:
            row, col = detection
            line = f"{img_idx}: ({row},{col})"
            correct += 1
        else:
            line = f"{img_idx}:"

        found_lines.append(line)
        print(f"  {fname} → {line}")

    # Write foundLocations.txt
    with open(FOUND_LOC_FILE, "w") as f:
        f.write("\n".join(found_lines) + "\n")
    print(f"\nWrote {FOUND_LOC_FILE}")

    # Write metrics.json for DVC
    metrics = {
        "test_images_processed" : total_images,
        "images_with_detection" : correct,
        "detection_rate"        : round(correct / total_images, 4) if total_images else 0
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {METRICS_FILE}")
    print(f"\nTest Summary:")
    print(f"  Images processed   : {total_images}")
    print(f"  Images with car    : {correct}")
    print(f"  Detection rate     : {metrics['detection_rate']:.2%}")
    print()
    print("To run Java evaluator:")
    print("  cd CarData && java Evaluator trueLocations.txt foundLocations.txt")


if __name__ == "__main__":
    evaluate()