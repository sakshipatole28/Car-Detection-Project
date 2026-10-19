"""Dataset loading for the UIUC Car Detection dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardet.features import read_image

_LOC = re.compile(r"\((-?\d+),(-?\d+)(?:,(-?\d+))?\)")


@dataclass
class PatchSet:
    images: list[np.ndarray]
    labels: np.ndarray  # 1 = car, 0 = background
    names: list[str]


def load_train_patches(train_dir: Path) -> PatchSet:
    """Load pos-*.pgm (cars) and neg-*.pgm (background) 100x40 training patches."""
    files = sorted(Path(train_dir).glob("*.pgm"))
    if not files:
        raise FileNotFoundError(f"no .pgm files in {train_dir}")
    images, labels, names = [], [], []
    for f in files:
        if f.name.startswith("pos-"):
            labels.append(1)
        elif f.name.startswith("neg-"):
            labels.append(0)
        else:
            continue
        images.append(read_image(f))
        names.append(f.name)
    return PatchSet(images, np.asarray(labels), names)


def parse_locations(path: Path) -> dict[int, list[tuple[int, ...]]]:
    """Parse trueLocations(.txt|_Scale.txt): `n: (i,j) ...` or `n: (i,j,w) ...`.

    Coordinates are (row, col[, width]) of the top-left corner, as in the README.
    """
    out: dict[int, list[tuple[int, ...]]] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        idx, rest = line.split(":", 1)
        out[int(idx)] = [
            tuple(int(v) for v in m.groups() if v is not None) for m in _LOC.finditer(rest)
        ]
    return out


def write_locations(path: Path, found: dict[int, list[tuple[int, ...]]], n_images: int) -> None:
    """Write detections in the format expected by the official Java Evaluator."""
    lines = []
    for n in range(n_images):
        locs = " ".join("(" + ",".join(str(v) for v in loc) + ")" for loc in found.get(n, []))
        lines.append(f"{n}: {locs}".rstrip())
    Path(path).write_text("\n".join(lines) + "\n")


def test_image_path(test_dir: Path, n: int) -> Path:
    return Path(test_dir) / f"test-{n}.pgm"
