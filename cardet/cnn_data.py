"""Training data for the CNN detector (any viewing angle).

Three sources are mixed on the fly, all read without extracting archives:

* Stanford Cars (YOLO format, inside CarData/cardata.zip): ~16k colour photos,
  one car per image, every viewing angle. Used as-is and as 2x2 mosaics so the
  model also sees several small cars per image.
* UIUC synthetic street scenes: 100x40 side-view car patches pasted onto
  canvases tiled from background patches. Some scenes get no car at all, so
  the model also learns that an image can be car-free.

Boxes are (x1, y1, x2, y2) in pixels; images are BGR uint8.
"""

from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cardet.data import load_train_patches

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def yolo_to_xyxy(line: str, w: int, h: int) -> list[float] | None:
    """Convert one YOLO label line ('cls cx cy bw bh', normalised) to pixel xyxy."""
    parts = line.split()
    if len(parts) != 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:])
    x1, y1 = max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h)
    x2, y2 = min(float(w), (cx + bw / 2) * w), min(float(h), (cy + bh / 2) * h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


class StanfordZip:
    """Random access to the YOLO-format Stanford Cars archive."""

    def __init__(self, zip_path: Path | str, split: str):
        self.zip_path = str(zip_path)
        self._zip: zipfile.ZipFile | None = None
        names = zipfile.ZipFile(self.zip_path).namelist()
        sub = f"/{SPLITS[split]}/images/"
        self.images = sorted(n for n in names if sub in n and n.lower().endswith(".jpg"))
        if not self.images:
            raise FileNotFoundError(f"no '{split}' images in {zip_path}")

    @property
    def zip(self) -> zipfile.ZipFile:
        # Opened lazily so each DataLoader worker gets its own handle.
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __len__(self) -> int:
        return len(self.images)

    def __getstate__(self):
        return {**self.__dict__, "_zip": None}

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        name = self.images[i]
        img = cv2.imdecode(np.frombuffer(self.zip.read(name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot decode {name}")
        h, w = img.shape[:2]
        label = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        text = self.zip.read(label).decode() if label in self.zip.NameToInfo else ""
        boxes = [b for b in (yolo_to_xyxy(ln, w, h) for ln in text.splitlines()) if b]
        return img, np.asarray(boxes, np.float32).reshape(-1, 4)


def resize_max(img, boxes, max_side: int):
    s = max_side / max(img.shape[:2])
    if s >= 1:
        return img, boxes
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, boxes * s


def mosaic(samples, size: int = 640):
    """Tile four (img, boxes) samples into a 2x2 grid, letterboxing each cell."""
    canvas = np.full((size, size, 3), 114, np.uint8)
    half = size // 2
    out = []
    for k, (img, boxes) in enumerate(samples):
        h, w = img.shape[:2]
        s = half / max(h, w)
        img = cv2.resize(
            img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
        )
        ox, oy = (
            (k % 2) * half + (half - img.shape[1]) // 2,
            (k // 2) * half + (half - img.shape[0]) // 2,
        )
        canvas[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
        out.append(boxes * s + np.array([ox, oy, ox, oy], np.float32))
    return canvas, np.concatenate(out).reshape(-1, 4)


@dataclass
class UiucSynth:
    """Synthetic grayscale street scenes from UIUC training patches."""

    pos: list[np.ndarray]
    neg: list[np.ndarray]

    @classmethod
    def from_dir(
        cls, train_dir: Path, part: str = "all", val_size: float = 0.2, seed: int = 42
    ) -> UiucSynth:
        """Load UIUC patches; `part` = "train" / "val" uses the same stratified split
        as the HOG model, so validation scenes are built from unseen patches."""
        ps = load_train_patches(train_dir)
        idx = np.arange(len(ps.labels))
        if part != "all":
            from sklearn.model_selection import train_test_split

            tr, va = train_test_split(
                idx, test_size=val_size, stratify=ps.labels, random_state=seed
            )
            idx = tr if part == "train" else va
        return cls(
            [ps.images[i] for i in idx if ps.labels[i] == 1],
            [ps.images[i] for i in idx if ps.labels[i] == 0],
        )

    def scene(self, rng: random.Random, p_empty: float = 0.25):
        cols, rows = rng.randint(2, 4), rng.randint(2, 4)
        bg = np.vstack(
            [np.hstack([rng.choice(self.neg) for _ in range(cols)]) for _ in range(rows)]
        )
        k = rng.uniform(1.0, 2.5)
        bg = cv2.resize(bg, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        H, W = bg.shape
        boxes = []
        if rng.random() >= p_empty:
            for _ in range(rng.randint(1, 3)):
                s = rng.uniform(0.9, 2.2)
                car = rng.choice(self.pos)
                if rng.random() < 0.5:
                    car = cv2.flip(car, 1)
                cw, ch = int(100 * s), int(40 * s)
                if cw >= W or ch >= H:
                    continue
                x, y = rng.randint(0, W - cw), rng.randint(0, H - ch)
                if any(x < b[2] and x + cw > b[0] and y < b[3] and y + ch > b[1] for b in boxes):
                    continue  # keep cars from overlapping
                bg[y : y + ch, x : x + cw] = cv2.resize(
                    car, (cw, ch), interpolation=cv2.INTER_LINEAR
                )
                boxes.append([x, y, x + cw, y + ch])
        return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), np.asarray(boxes, np.float32).reshape(-1, 4)


def photometric(img: np.ndarray, rng: random.Random, p_gray: float) -> np.ndarray:
    """Brightness/contrast jitter and occasional grayscale (UIUC cameras are mono)."""
    a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=a, beta=b)
    if rng.random() < p_gray:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img


def hflip(img, boxes):
    w = img.shape[1]
    img = cv2.flip(img, 1)
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img, boxes
