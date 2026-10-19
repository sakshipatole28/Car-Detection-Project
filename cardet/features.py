"""Image preprocessing and HOG features.

Training patches and inference windows go through exactly the same code path:
`to_gray` normalises any input image, and `HogExtractor` computes features either
for single 100x40 patches or for every sliding window of a larger image. A unit
test asserts both paths give identical vectors, which rules out train/serve skew.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cardet.config import HogConfig


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert a decoded image (gray / BGR / BGRA, 8 or 16 bit) to uint8 grayscale."""
    if img is None or img.size == 0:
        raise ValueError("empty image")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(img)


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded bytes (PGM, PNG, JPEG, BMP, ...) into uint8 grayscale."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode image; supported formats: PGM, PNG, JPEG, BMP")
    return to_gray(img)


def decode_image_color(data: bytes) -> np.ndarray:
    """Decode encoded bytes keeping colour: BGR uint8, or 2-D uint8 for gray files."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode image; supported formats: PGM, PNG, JPEG, BMP")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return np.ascontiguousarray(img)


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return to_gray(img)


class HogExtractor:
    """Thin, picklable wrapper around cv2.HOGDescriptor."""

    def __init__(self, cfg: HogConfig):
        self.cfg = cfg
        self._hog = None

    @property
    def window(self) -> tuple[int, int]:
        return self.cfg.win_width, self.cfg.win_height

    @property
    def hog(self) -> cv2.HOGDescriptor:
        if self._hog is None:
            c = self.cfg
            self._hog = cv2.HOGDescriptor(
                (c.win_width, c.win_height),
                (c.block_size, c.block_size),
                (c.block_stride, c.block_stride),
                (c.cell_size, c.cell_size),
                c.nbins,
            )
        return self._hog

    @property
    def n_features(self) -> int:
        return int(self.hog.getDescriptorSize())

    def __getstate__(self):
        return {"cfg": self.cfg, "_hog": None}

    def patch(self, img: np.ndarray) -> np.ndarray:
        """Features for one patch; resized to the window size if needed."""
        w, h = self.window
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return self.hog.compute(img).ravel()

    def patches(self, imgs) -> np.ndarray:
        return np.stack([self.patch(im) for im in imgs]).astype(np.float32)

    def windows(self, img: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
        """Features for every window of `img` at the given stride.

        Each window is described in isolation, exactly like a training patch.
        (cv2's whole-image mode reuses gradients from pixels outside the window,
        which would differ from how the 100x40 training crops were described.)
        Returns (features [N, D], top-left corners [N, 2] as (x, y)).
        """
        w, h = self.window
        H, W = img.shape
        if h > H or w > W:
            return np.empty((0, self.n_features), np.float32), np.empty((0, 2), int)
        views = sliding_window_view(img, (h, w))[::stride, ::stride]
        ny, nx = views.shape[:2]
        hog = self.hog
        feats = np.empty((ny * nx, self.n_features), np.float32)
        for k in range(ny * nx):
            feats[k] = hog.compute(np.ascontiguousarray(views[k // nx, k % nx])).ravel()
        ys, xs = np.divmod(np.arange(ny * nx), nx)
        return feats, np.stack([xs * stride, ys * stride], axis=1)
