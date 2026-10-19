import numpy as np

from cardet.config import DetectConfig
from cardet.detector import CarDetector, nms

TEMPLATE = np.zeros((40, 100), np.float32)
TEMPLATE[8:32, 20:80] = 1.0  # a bright "car" with a dark margin around it


def draw_object(img, x, y, scale):
    h, w = int(40 * scale), int(100 * scale)
    img[
        y + int(8 * scale) : y + h - int(8 * scale), x + int(20 * scale) : x + w - int(20 * scale)
    ] = 255


class SimilarityClassifier:
    """Stub: car probability = similarity of the window to TEMPLATE."""

    def predict_proba(self, feats):
        p = feats[:, 0]
        return np.stack([1 - p, p], 1)


class StubExtractor:
    """Feature = 1 - mean |window - template| (1.0 only for an exact, size-matched hit)."""

    window = (100, 40)
    n_features = 1

    def windows(self, img, stride):
        H, W = img.shape
        out, tl = [], []
        for y in range(0, H - 40 + 1, stride):
            for x in range(0, W - 100 + 1, stride):
                out.append([self._sim(img[y : y + 40, x : x + 100])])
                tl.append([x, y])
        return np.asarray(out, np.float32).reshape(-1, 1), np.asarray(tl).reshape(-1, 2)

    def patch(self, img):
        return np.array([self._sim(img)], np.float32)

    @staticmethod
    def _sim(win):
        return 1 - np.abs(win.astype(np.float32) / 255 - TEMPLATE).mean()


def make_detector(**cfg):
    ext = StubExtractor()
    return CarDetector(ext, SimilarityClassifier(), DetectConfig(**cfg))


def test_nms_suppresses_overlaps_and_keeps_best_first():
    boxes = np.array([[0, 0, 100, 40], [4, 0, 104, 40], [300, 0, 400, 40]])
    scores = np.array([0.8, 0.9, 0.7])
    assert nms(boxes, scores, 0.3) == [1, 2]


def test_nms_commutes_with_threshold(rng):
    """Filtering after NMS == NMS after filtering (the evaluation sweep relies on it)."""
    for _ in range(25):
        xy = rng.uniform(0, 200, (40, 2))
        boxes = np.hstack([xy, xy + rng.uniform(30, 120, (40, 2))])
        scores = rng.uniform(0, 1, 40)
        for t in (0.2, 0.5, 0.8):
            after = [i for i in nms(boxes, scores, 0.3) if scores[i] >= t]
            idx = np.flatnonzero(scores >= t)
            before = [int(idx[i]) for i in nms(boxes[idx], scores[idx], 0.3)]
            assert sorted(after) == sorted(before)


def test_detects_single_scale_object():
    img = np.zeros((120, 300), np.uint8)
    draw_object(img, 120, 40, 1.0)
    res = make_detector(scales=(1.0,), stride=4, threshold=0.99).detect(img)
    assert len(res.boxes) == 1
    b = res.boxes[0]
    assert (b.x, b.y, b.w, b.h) == (120, 40, 100, 40)


def test_detects_object_at_larger_scale():
    img = np.zeros((200, 400), np.uint8)
    draw_object(img, 100, 50, 2.0)  # 200x80 object
    res = make_detector(scales=(1.0, 1.5, 2.0), stride=2, threshold=0.9).detect(img)
    best = res.boxes[0]
    assert abs(best.w - 200) <= 2 and abs(best.x - 100) <= 4 and abs(best.y - 50) <= 4


def test_large_images_are_downscaled():
    img = np.zeros((800, 1600), np.uint8)
    res = make_detector(scales=(1.0,), stride=16, max_side=640).detect(img)
    assert res.resize_factor == 0.4 and res.image_size == (640, 320)
