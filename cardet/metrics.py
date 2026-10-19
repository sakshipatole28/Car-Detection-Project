"""Python port of the official UIUC evaluators (CarData/Evaluator*.java).

A detection is correct if it falls inside an ellipse (ellipsoid for the
multi-scale set) centred on a true object with axes = 25% of the object size.
Each true object can be matched once; extra hits count as false positives.
Detections are matched in the order given (pass them sorted by score).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBJ_W, OBJ_H = 100, 40
AXIS = 0.25
HW_RATIO = OBJ_H / OBJ_W


@dataclass
class DetectionScore:
    n_objects: int
    tp: int
    fp: int

    @property
    def recall(self) -> float:
        return self.tp / self.n_objects if self.n_objects else 0.0

    @property
    def precision(self) -> float:
        n = self.tp + self.fp
        return self.tp / n if n else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}n_objects": self.n_objects,
            f"{prefix}tp": self.tp,
            f"{prefix}fp": self.fp,
            f"{prefix}recall": round(self.recall, 4),
            f"{prefix}precision": round(self.precision, 4),
            f"{prefix}f1": round(self.f1, 4),
        }


def _acceptable_single(p, q) -> bool:
    hd, wd = abs(p[0] - q[0]), abs(p[1] - q[1])
    return (hd / (AXIS * OBJ_H)) ** 2 + (wd / (AXIS * OBJ_W)) ** 2 <= 1


def _acceptable_scale(p, q) -> bool:
    # Mirrors Evaluator_Scale.isAcceptable, including Java's integer truncation.
    def centre(t):
        return t[0] + int(HW_RATIO * t[2] / 2), t[1] + t[2] // 2

    (pi, pj), (qi, qj) = centre(p), centre(q)
    qw = q[2]
    return ((pi - qi) / (AXIS * HW_RATIO * qw)) ** 2 + ((pj - qj) / (AXIS * qw)) ** 2 + (
        (p[2] - qw) / (AXIS * qw)
    ) ** 2 <= 1


def score_detections(truth: dict, found: dict, multiscale: bool = False) -> DetectionScore:
    """truth/found map image index -> list of (i, j) or (i, j, w) tuples."""
    accept = _acceptable_scale if multiscale else _acceptable_single
    n_obj = sum(len(v) for v in truth.values())
    tp = fp = 0
    for n, dets in found.items():
        remaining = list(truth.get(n, []))
        for p in dets:
            for k, q in enumerate(remaining):
                if accept(p, q):
                    tp += 1
                    remaining.pop(k)
                    break
            else:
                fp += 1
    return DetectionScore(n_obj, tp, fp)


def sweep_thresholds(truth: dict, scored: dict, multiscale: bool, thresholds) -> list[dict]:
    """Evaluate at many thresholds from one low-threshold detection pass.

    `scored` maps image -> list of (location_tuple, score) after NMS. Greedy NMS
    commutes with thresholding, so filtering here equals re-running detection.
    """
    rows = []
    for t in thresholds:
        found = {n: [loc for loc, s in dets if s >= t] for n, dets in scored.items()}
        s = score_detections(truth, found, multiscale)
        rows.append(
            {
                "threshold": float(t),
                "recall": s.recall,
                "precision": s.precision,
                "f1": s.f1,
                "tp": s.tp,
                "fp": s.fp,
            }
        )
    return rows


def equal_error_rate(rows: list[dict]) -> dict:
    """Operating point where recall ~= precision (the metric reported in the UIUC papers)."""
    best = min(rows, key=lambda r: abs(r["recall"] - r["precision"]))
    return {
        "threshold": best["threshold"],
        "recall": round(best["recall"], 4),
        "precision": round(best["precision"], 4),
    }


def average_precision(rows: list[dict]) -> float:
    """Area under the precision-recall curve (all-points interpolation)."""
    pts = sorted((r["recall"], r["precision"]) for r in rows)
    rec = np.array([0.0] + [p[0] for p in pts] + [pts[-1][0]])
    prec = np.array([1.0] + [p[1] for p in pts] + [0.0])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum(np.diff(rec) * prec[1:]))
