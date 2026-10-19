import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cardet.cnn import CnnDetector, build_model, to_tensor  # noqa: E402
from cardet.cnn_data import UiucSynth, hflip, mosaic, yolo_to_xyxy  # noqa: E402
from cardet.detector import Box  # noqa: E402
from cardet.evaluate import uiuc_location  # noqa: E402
from cardet.train_cnn import detection_ap  # noqa: E402


def test_yolo_to_xyxy_converts_and_clips():
    assert yolo_to_xyxy("0 0.5 0.5 0.5 0.5", 200, 100) == [50.0, 25.0, 150.0, 75.0]
    assert yolo_to_xyxy("0 0.1 0.5 0.4 0.2", 100, 100) == pytest.approx([0, 40, 30, 60])  # clipped
    assert yolo_to_xyxy("garbage", 10, 10) is None


def test_hflip_mirrors_boxes():
    img = np.zeros((10, 100, 3), np.uint8)
    _, b = hflip(img, np.array([[10, 0, 30, 5]], np.float32))
    assert b.tolist() == [[70, 0, 90, 5]]


def test_mosaic_places_boxes_in_their_cells():
    imgs = [(np.zeros((100, 200, 3), np.uint8), np.array([[0, 0, 200, 100]], np.float32))] * 4
    canvas, boxes = mosaic(imgs, size=400)
    assert canvas.shape == (400, 400, 3) and len(boxes) == 4
    # each 200x100 image is scaled to 200x100 in a 200x200 cell, centred vertically
    assert boxes[0].tolist() == [0, 50, 200, 150]
    assert boxes[3].tolist() == [200, 250, 400, 350]


def test_uiuc_synthetic_scenes_are_valid():
    rng = random.Random(0)
    synth = UiucSynth([np.full((40, 100), 200, np.uint8)], [np.zeros((40, 100), np.uint8)])
    n_empty = 0
    for _ in range(50):
        img, boxes = synth.scene(rng, p_empty=0.3)
        h, w = img.shape[:2]
        assert img.ndim == 3
        n_empty += len(boxes) == 0
        for x1, y1, x2, y2 in boxes:
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            assert img[int(y1) : int(y2), int(x1) : int(x2)].mean() > 150  # the pasted car
    assert 0 < n_empty < 50


def test_detection_ap():
    gt = [np.array([[0, 0, 10, 10]], np.float32), np.array([[0, 0, 10, 10]], np.float32)]
    perfect = [(np.array([[0, 0, 10, 10]], np.float32), np.array([0.9]))] * 2
    assert detection_ap(perfect, gt)["ap50"] == 1.0
    # one hit, one miss (wrong place) and a duplicate on the hit
    preds = [
        (np.array([[0, 0, 10, 10], [1, 1, 10, 10]], np.float32), np.array([0.9, 0.8])),
        (np.array([[50, 50, 60, 60]], np.float32), np.array([0.7])),
    ]
    m = detection_ap(preds, gt)
    assert m["ap50"] == 0.5 and m["recall_at_best"] == 0.5


def test_to_tensor_is_grayscale_for_colour_input():
    img = np.zeros((4, 5, 3), np.uint8)
    img[..., 2] = 255  # pure red in BGR
    t = to_tensor(img)
    assert t.shape == (3, 4, 5)
    assert torch.allclose(t[0], t[1]) and torch.allclose(t[1], t[2])


def test_cnn_uiuc_location_centres_reference_window():
    b = Box(x=110, y=60, w=80, h=30, score=0.9)  # centre (150, 75)
    assert uiuc_location(b, False, "cnn") == (55, 100)  # 100x40 window around centre
    assert uiuc_location(b, True, "cnn") == (59, 110, 80)  # cy - 0.2*w
    assert uiuc_location(b, False, "hog") == (60, 110)


@pytest.fixture(scope="module")
def untrained_detector():
    torch.manual_seed(0)
    return CnnDetector(build_model(pretrained=False), threshold=0.5, max_side=320)


def test_cnn_detector_output_contract(untrained_detector):
    res = untrained_detector.detect(np.zeros((200, 640), np.uint8), threshold=0.05)
    assert res.resize_factor == 0.5 and res.image_size == (320, 100)
    assert all(isinstance(b, Box) for b in res.boxes)
    assert [b.score for b in res.boxes] == sorted((b.score for b in res.boxes), reverse=True)


def test_cnn_checkpoint_roundtrip(untrained_detector, tmp_path):
    untrained_detector.metadata = {"note": "test"}
    path = tmp_path / "m.pt"
    untrained_detector.save(path)
    clone = CnnDetector.load(path)
    assert clone.threshold == 0.5 and clone.metadata == {"note": "test"}
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    a = untrained_detector.predict([img])[0][1]
    b = clone.predict([img])[0][1]
    assert len(a) == len(b)  # fp16 storage: same detections, scores ~equal
    np.testing.assert_allclose(a, b, atol=2e-2)
