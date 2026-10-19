"""Fine-tune the CNN detector on Stanford Cars + UIUC synthetic scenes.

    python -m cardet.train_cnn            (or: dvc repro train_cnn)

* Starts from COCO-pretrained SSDlite and keeps its 91-class head ('car' = 3).
* BatchNorm statistics stay frozen: CPU batches are too small to estimate them.
* SGD with warm-up + cosine decay over an iteration *or* wall-clock budget,
  whichever is reached first (the learning rate is annealed either way).
* Validation (before and after fine-tuning): AP@0.5 on a subset of the Stanford
  *validation* split and on synthetic street scenes built from *held-out* UIUC
  patches (same 80/20 split as the HOG model). The operating threshold is the
  best-F1 point on both validation sets combined, so it suits close-up photos
  and small street cars alike. The test sets are never used for any choice.
* The checkpoint is written before MLflow logging, so a tracking failure can
  never lose a trained model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from datetime import UTC, datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cardet import __version__
from cardet.cnn import COCO_CAR, CnnDetector, build_model, to_tensor
from cardet.cnn_data import StanfordZip, UiucSynth, hflip, mosaic, photometric, resize_max
from cardet.config import REPORTS_DIR, ROOT, load_params

log = logging.getLogger("cardet.train_cnn")
CNN_MODEL_PATH = ROOT / "models" / "cnn_detector.pt"


class MixedCars(Dataset):
    """Each item draws one sample from a source picked by the configured mix."""

    def __init__(self, stanford: StanfordZip, uiuc: UiucSynth, cfg: dict, length: int, seed: int):
        self.s, self.u, self.cfg, self.length, self.seed = stanford, uiuc, cfg, length, seed

    def __len__(self):
        return self.length

    def _stanford(self, rng):
        img, boxes = self.s.load(rng.randrange(len(self.s)))
        return resize_max(img, boxes, 640)

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx)
        mix = self.cfg["mix"]
        r = rng.random()
        if r < mix["stanford"]:
            img, boxes = self._stanford(rng)
        elif r < mix["stanford"] + mix["mosaic"]:
            img, boxes = mosaic([self._stanford(rng) for _ in range(4)])
        else:
            img, boxes = self.u.scene(rng, p_empty=self.cfg["uiuc_empty_prob"])
        if rng.random() < 0.5:
            img, boxes = hflip(img, boxes)
        img = photometric(img, rng, self.cfg["gray_prob"])
        target = {
            "boxes": torch.from_numpy(boxes).float().reshape(-1, 4),
            "labels": torch.full((len(boxes),), COCO_CAR, dtype=torch.int64),
        }
        return to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch, strict=True))


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def detection_ap(preds, gts, iou_thr: float = 0.5) -> dict:
    """VOC-style AP@iou_thr for one class + the best-F1 operating point.

    preds: list of (boxes [N,4], scores [N]); gts: list of boxes [M,4].
    """
    scores, tps = [], []
    n_gt = sum(len(g) for g in gts)
    for (boxes, sc), gt in zip(preds, gts, strict=True):
        order = np.argsort(-sc)
        used = np.zeros(len(gt), bool)
        ious = box_iou(boxes[order], gt) if len(gt) and len(boxes) else None
        for k, i in enumerate(order):
            hit = False
            if ious is not None:
                j = int(np.argmax(np.where(used, -1, ious[k])))
                if not used[j] and ious[k, j] >= iou_thr:
                    used[j] = hit = True
            scores.append(sc[i])
            tps.append(hit)
    if not scores:
        return {"ap50": 0.0, "best_f1": 0.0, "best_threshold": 0.5, "n_gt": n_gt}
    order = np.argsort(-np.asarray(scores))
    tp = np.cumsum(np.asarray(tps)[order])
    fp = np.cumsum(~np.asarray(tps)[order])
    rec, prec = tp / max(n_gt, 1), tp / (tp + fp)
    mprec = np.concatenate([[0], prec, [0]])
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    ap = float(np.sum(np.diff(mrec) * mprec[1:]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    b = int(np.argmax(f1))
    return {
        "ap50": round(ap, 4),
        "best_f1": round(float(f1[b]), 4),
        "precision_at_best": round(float(prec[b]), 4),
        "recall_at_best": round(float(rec[b]), 4),
        "best_threshold": round(float(np.asarray(scores)[order][b]), 3),
        "n_gt": n_gt,
    }


def pr_at_threshold(preds, gts, thr: float, iou_thr: float = 0.5) -> dict:
    """Precision / recall / F1 of the boxes scoring >= thr (greedy IoU matching)."""
    tp = fp = 0
    for (boxes, scores), gt in zip(preds, gts, strict=True):
        order = np.argsort(-scores)
        used = np.zeros(len(gt), bool)
        for b in boxes[order][scores[order] >= thr]:
            ious = box_iou(b[None], gt)[0] if len(gt) else np.zeros(0)
            j = int(np.argmax(np.where(used, -1, ious))) if len(gt) else -1
            if j >= 0 and not used[j] and ious[j] >= iou_thr:
                used[j] = True
                tp += 1
            else:
                fp += 1
    n_gt = sum(len(g) for g in gts)
    prec, rec = tp / max(tp + fp, 1), tp / max(n_gt, 1)
    return {
        "threshold": round(float(thr), 3),
        "tp": tp,
        "fp": fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
    }


def evaluate_split(
    det: CnnDetector,
    data: StanfordZip,
    n: int | None,
    seed: int = 0,
    batch: int = 8,
    threshold: float | None = None,
) -> dict:
    """AP@0.5 on a split; with `threshold`, also P/R/F1 at that operating point."""
    idx = list(range(len(data)))
    if n and n < len(idx):
        idx = sorted(random.Random(seed).sample(idx, n))
    preds, gts = [], []
    for s in range(0, len(idx), batch):
        items = [resize_max(*data.load(i), 640) for i in idx[s : s + batch]]
        preds += det.predict([im for im, _ in items])
        gts += [b for _, b in items]
    out = detection_ap(preds, gts)
    if threshold is not None:
        out["at_threshold"] = pr_at_threshold(preds, gts, threshold)
    return out


def predict_scenes(det: CnnDetector, scenes, batch: int = 8):
    preds = []
    for s in range(0, len(scenes), batch):
        preds += det.predict([im for im, _ in scenes[s : s + batch]])
    return preds, [b for _, b in scenes]


def validate(det: CnnDetector, val_set: StanfordZip, scenes, n_stanford: int, seed: int) -> dict:
    """AP on Stanford val, on held-out UIUC scenes, and on both combined."""
    idx = list(range(len(val_set)))
    idx = sorted(random.Random(seed).sample(idx, min(n_stanford, len(idx))))
    st = [resize_max(*val_set.load(i), 640) for i in idx]
    p_st, g_st = predict_scenes(det, st)
    p_ui, g_ui = predict_scenes(det, scenes)
    return {
        "stanford": detection_ap(p_st, g_st),
        "uiuc_scenes": detection_ap(p_ui, g_ui),
        "combined": detection_ap(p_st + p_ui, g_st + g_ui),
    }


def train(
    p, use_mlflow: bool = True, max_iters: int | None = None, max_minutes: float | None = None
) -> CnnDetector:
    cfg = p["cnn"]
    seed = p["data"]["seed"]
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("threads") or os.cpu_count() or 1)
    zip_path = ROOT / cfg["zip"]
    REPORTS_DIR.mkdir(exist_ok=True)

    train_set = StanfordZip(zip_path, "train")
    val_set = StanfordZip(zip_path, "val")
    split = {"val_size": p["data"]["val_size"], "seed": seed}
    uiuc = UiucSynth.from_dir(p.path("train_dir"), part="train", **split)
    uiuc_val = UiucSynth.from_dir(p.path("train_dir"), part="val", **split)
    val_scenes = [
        uiuc_val.scene(random.Random(seed * 7919 + i), p_empty=cfg["uiuc_empty_prob"])
        for i in range(cfg["val_scenes"])
    ]
    log.info(
        "Stanford train=%d val=%d | UIUC patches train pos=%d neg=%d, val scenes=%d",
        len(train_set),
        len(val_set),
        len(uiuc.pos),
        len(uiuc.neg),
        len(val_scenes),
    )

    model = build_model(pretrained=True)
    det = CnnDetector(model)
    log.info("Baseline (COCO weights, no fine-tuning)…")
    baseline = validate(det, val_set, val_scenes, cfg["val_images"], seed)
    log.info("  baseline %s", baseline)

    iters = max_iters or cfg["iters"]
    budget = (max_minutes or cfg["max_minutes"]) * 60
    ds = MixedCars(train_set, uiuc, cfg, length=iters * cfg["batch_size"], seed=seed)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["workers"],
        collate_fn=collate,
        persistent_workers=cfg["workers"] > 0,
    )

    model.train()
    freeze_bn(model)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    warm = cfg["warmup_iters"]

    def lr_at(it: int, elapsed: float) -> float:
        # Cosine decay over whichever budget runs out first (iterations or time),
        # so a time-capped run still ends with an annealed learning rate.
        progress = min(1.0, max(it / iters, elapsed / budget))
        return cfg["lr"] * min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * progress))

    t0 = time.perf_counter()
    history, run_loss, done, lr = [], None, 0, 0.0
    for it, (imgs, targets) in enumerate(loader):
        lr = lr_at(it, time.perf_counter() - t0)
        for g in opt.param_groups:
            g["lr"] = lr
        losses = model(list(imgs), list(targets))
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        done = it + 1
        run_loss = loss.item() if run_loss is None else 0.95 * run_loss + 0.05 * loss.item()
        if done % 25 == 0 or done == iters:
            elapsed = time.perf_counter() - t0
            history.append(
                {
                    "iter": done,
                    "loss": round(run_loss, 4),
                    "lr": lr,
                    "minutes": round(elapsed / 60, 2),
                }
            )
            log.info(
                "iter %4d/%d  loss %.3f  lr %.5f  %.1f min",
                done,
                iters,
                run_loss,
                lr,
                elapsed / 60,
            )
        if time.perf_counter() - t0 > budget:
            log.info("Time budget reached after %d iterations.", done)
            break
    train_minutes = (time.perf_counter() - t0) / 60

    model.eval()
    det = CnnDetector(model)
    log.info("Validating fine-tuned model…")
    val = validate(det, val_set, val_scenes, cfg["val_images"], seed)
    log.info("  fine-tuned %s", val)
    det.threshold = float(np.clip(val["combined"]["best_threshold"], 0.2, 0.9))
    det.metadata = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_family": "ssdlite320_mobilenet_v3_large",
        "iterations": done,
        "train_minutes": round(train_minutes, 1),
        "batch_size": cfg["batch_size"],
        "mix": cfg["mix"],
        "validation": {"baseline_coco": baseline, "fine_tuned": val},
        "threshold_source": "best F1 on Stanford val + held-out UIUC scenes",
    }
    # Save first: a tracking failure must never cost a trained model.
    det.save(CNN_MODEL_PATH)
    if use_mlflow:
        det.metadata["mlflow_run_id"] = log_to_mlflow(p, det, history, baseline, val)
        det.save(CNN_MODEL_PATH)
    (REPORTS_DIR / "cnn_train_metrics.json").write_text(
        json.dumps(
            {
                "baseline_val": baseline,
                "val": val,
                "iterations": done,
                "train_minutes": round(train_minutes, 1),
                "threshold": det.threshold,
            },
            indent=2,
        )
    )
    (REPORTS_DIR / "cnn_loss.json").write_text(json.dumps(history))
    log.info("Saved %s (threshold %.2f)", CNN_MODEL_PATH.relative_to(ROOT), det.threshold)
    return det


def log_to_mlflow(p, det, history, baseline, val) -> str:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "car-detection"))
    with mlflow.start_run(run_name="train-cnn-ssdlite") as run:
        mlflow.set_tags({"stage": "train", "model": "cnn", "cardet_version": __version__})
        mlflow.log_params({f"cnn.{k}": str(v)[:250] for k, v in p["cnn"].items()})
        for h in history:
            mlflow.log_metric("train_loss", h["loss"], step=h["iter"])
            mlflow.log_metric("lr", h["lr"], step=h["iter"])
        for prefix, res in (("val_baseline", baseline), ("val", val)):
            for part, m in res.items():
                mlflow.log_metrics(
                    {f"{prefix}_{part}_{k}": v for k, v in m.items() if isinstance(v, float)}
                )
        mlflow.log_metric("threshold", det.threshold)
        mlflow.pytorch.log_model(
            det.model,
            name="cnn_detector",
            serialization_format="pickle",  # 'pt2' needs a traceable example input
            registered_model_name=os.environ.get("MLFLOW_CNN_MODEL_NAME", "car-detector-cnn"),
        )
        return run.info.run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--iters", type=int, default=None, help="override cnn.iters")
    ap.add_argument("--minutes", type=float, default=None, help="override cnn.max_minutes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(
        load_params(), use_mlflow=not args.no_mlflow, max_iters=args.iters, max_minutes=args.minutes
    )


if __name__ == "__main__":
    main()
