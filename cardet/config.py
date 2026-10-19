"""Typed access to params.yaml so every stage reads the same configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
MODEL_PATH = ROOT / "models" / "car_detector.joblib"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class HogConfig:
    win_width: int = 100
    win_height: int = 40
    block_size: int = 20
    block_stride: int = 10
    cell_size: int = 10
    nbins: int = 9


@dataclass(frozen=True)
class DetectConfig:
    stride: int = 4
    threshold: float = 0.5
    nms_iou: float = 0.2
    scales: tuple[float, ...] = (0.8, 0.92, 1.06, 1.22, 1.4, 1.61, 1.85, 2.13, 2.45)
    max_side: int = 640


@dataclass(frozen=True)
class Params:
    raw: dict[str, Any] = field(repr=False)
    hog: HogConfig
    detect: DetectConfig

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        return ROOT / self.raw["data"][key]


def load_params(path: Path | str = PARAMS_PATH) -> Params:
    with open(path) as f:
        raw = yaml.safe_load(f)
    det = dict(raw["detect"])
    det["scales"] = tuple(det["scales"])
    return Params(raw=raw, hog=HogConfig(**raw["features"]), detect=DetectConfig(**det))
