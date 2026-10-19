# Problem Statement: Roadside Car Detection Service

*Take-home project for an AI/ML intern role. Expected effort: 3–5 days.*

## Context

A city transport department wants to count cars automatically, both in footage
from fixed roadside cameras and in photos that field staff upload from phones
(car parks, junctions, incident reports). The cameras record grayscale, side-on
views. The phone photos show cars from **any angle**: front, rear, side or
three-quarter.

A first prototype only handled side views. It failed completely on photos taken
from other angles, so this project must fix that without losing accuracy on the
roadside cameras.

The constraints come from the deployment:

- **CPU only.** Cameras report to small edge boxes with no GPU.
- **Explainable and auditable.** The team must be able to show *why* a box was
  drawn and *how good* the system is, using an agreed, reproducible metric.
- **Operable.** The model has to be retrainable, versioned, testable and
  deployable by someone other than its author.

## Task

Build a system that takes an image and returns a **bounding box for every car**
in it, with a confidence score for each box. It must work on cars seen from any
angle and on grayscale side-view street scenes.

## Data

There are two datasets.

**1. UIUC Image Database for Car Detection** (side views), in `CarData/`:

| Split | Contents |
|---|---|
| `TrainImages/` | 550 car patches (`pos-*.pgm`) and 500 background patches (`neg-*.pgm`), each 100×40 |
| `TestImages/` | 170 scenes, 200 cars, all at roughly training scale |
| `TestImages_Scale/` | 108 scenes, 139 cars, at scales from 0.9× to 2.1× |
| `trueLocations*.txt` | Ground-truth car locations |
| `Evaluator*.java` | The official evaluation programs |

Note that the UIUC training data contains **patches**, not scenes. Turning a
patch classifier into a scene detector is part of the problem.

**2. Stanford Cars** (every viewing angle), in `CarData/cardata.zip`. It is in
YOLO format with box labels:

| Split | Images |
|---|---|
| `train/` | 12,949 |
| `valid/` | 1,618 |
| `test/` | 1,618 |

Each image contains one car, usually large in the frame. This release is
grayscale. The archive is 930 MB, so version it with DVC rather than git.

## Requirements

1. **Preprocessing and features.** Load and normalise the images. Training and
   inference must compute features with exactly the same code, and you must
   show that they do.
2. **Baseline model.** Build a classical baseline on the UIUC patches. Compare
   at least three model families with a sound validation scheme. Augmented
   copies of an image must not end up in both a training fold and a
   validation fold.
3. **Any-angle model.** Fine-tune a pretrained CNN detector that runs on a CPU.
   Measure the pretrained model *before* fine-tuning, so the gain is
   demonstrated rather than assumed.
4. **Detection.** Handle cars of different sizes and remove duplicate boxes.
5. **Evaluation.**
   - On both UIUC test sets, report precision, recall and F1 using the
     **official UIUC criterion**, plus recall at the equal-error rate. Show
     that your numbers match the official evaluator.
   - On the Stanford test split, report AP@0.5.
   - Choose every threshold on validation data, never on test data.
6. **Experiment tracking.** Log parameters, metrics and artefacts with MLflow,
   and register the model.
7. **Reproducibility.** Configure everything from one params file and run the
   pipeline with DVC.
8. **Serving.** Provide:
   - a REST API with input validation that lets the caller choose the model;
   - a responsive web UI that works on a phone;
   - a BentoML service.
9. **Engineering.** Provide:
   - unit and API tests;
   - a Docker image;
   - CI that lints, tests, retrains, enforces a **quality gate** and publishes
     the image.

## Acceptance criteria

| Metric | Target |
|---|---|
| Baseline: F1 on the UIUC single-scale test set | ≥ 0.80 |
| Baseline: patch-level ROC-AUC on the held-out validation set | ≥ 0.97 |
| Any-angle model: AP@0.5 on the Stanford test split | ≥ 0.90 |
| Any-angle model: F1 on the UIUC single-scale test set (no forgetting) | ≥ 0.80 |
| Latency per test image on a laptop CPU | < 2 s |
| `docker run` → `/health` returns 200 and `POST /api/v1/detect` works | required |

## How submissions are assessed

| Area | Weight | What we look for |
|---|---|---|
| ML correctness | 35% | No leakage, a faithful metric, sensible model selection, honest reporting of errors |
| Code quality | 20% | Clear modules, typed configuration, tests that catch real bugs |
| MLOps | 20% | A working MLflow, DVC and CI setup, with a quality gate that actually gates |
| Product | 15% | A usable, responsive UI and a clean API |
| Communication | 10% | A README that states results, decisions and limitations |

## Stretch goals

- Speed up sliding-window inference (shared gradient computation or a cascade).
- Choose the operating threshold on validation scenes rather than the test set.
- Compare against a small CNN, such as MobileNet fine-tuned on the patches,
  and discuss the trade-offs.
- Add drift monitoring on the live request log.

---

A reference solution is in this repository. See [README.md](README.md).
