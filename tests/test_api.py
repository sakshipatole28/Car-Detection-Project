"""API tests against the committed model (skipped if it hasn't been trained)."""

import pytest
from fastapi.testclient import TestClient

from cardet.config import MODEL_PATH, ROOT

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
SAMPLE = ROOT / "app" / "static" / "samples" / "single-1.png"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.db"
    mp = pytest.MonkeyPatch()
    mp.setenv("CARDET_DB_URL", f"sqlite:///{db}")
    from app.main import app

    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert "hog" in body["models"] and body["default_model"] in body["models"]


@pytest.mark.parametrize("path", ["/", "/dashboard", "/about", "/docs"])
def test_pages_render(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("model", ["hog", "cnn"])
def test_detect_finds_both_cars(client, model):
    if model not in client.get("/health").json()["models"]:
        pytest.skip(f"{model} model not trained")
    # Ground truth for test-1: (row, col) = (61, 20) and (63, 140) of the 100x40 window
    r = client.post(
        "/api/v1/detect",
        params={"model": model},
        files={"file": ("s.png", SAMPLE.read_bytes(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == model
    assert body["count"] == 2 and body["image_png_base64"] is None
    # compare box centres with the centres of the true 100x40 windows
    found = sorted((d["y"] + d["h"] / 2, d["x"] + d["w"] / 2) for d in body["detections"])
    for (ty, tx), (cy, cx) in zip([(81, 70), (83, 190)], found, strict=True):
        assert abs(ty - cy) <= 10 and abs(tx - cx) <= 25


def test_cnn_detects_cars_from_other_angles(client):
    if "cnn" not in client.get("/health").json()["models"]:
        pytest.skip("cnn model not trained")
    for name in ("angle-2-pickup.jpg", "angle-3-rear.jpg", "angle-4-front.jpg"):
        data = (SAMPLE.parent / name).read_bytes()
        body = client.post(
            "/api/v1/detect", params={"model": "cnn"}, files={"file": (name, data, "image/jpeg")}
        ).json()
        assert body["count"] >= 1, name
        # the main car should take up a large part of these photos
        biggest = max(d["w"] * d["h"] for d in body["detections"])
        assert biggest > 0.2 * body["image_width"] * body["image_height"], name


def test_detect_return_image_includes_low_scores(client):
    r = client.post(
        "/api/v1/detect",
        params={"return_image": True, "threshold": 0.9},
        files={"file": ("s.png", SAMPLE.read_bytes(), "image/png")},
    )
    body = r.json()
    assert body["image_png_base64"]
    assert len(body["detections"]) >= body["count"]


def test_rejects_non_images_and_bad_threshold(client):
    assert client.post("/api/v1/detect", files={"file": ("x.txt", b"hello")}).status_code == 415
    assert client.post("/api/v1/detect", files={"file": ("x.png", b"")}).status_code == 400
    r = client.post(
        "/api/v1/detect", params={"threshold": 2}, files={"file": ("s.png", SAMPLE.read_bytes())}
    )
    assert r.status_code == 422


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setenv("CARDET_MAX_UPLOAD_MB", "0.001")
    r = client.post("/api/v1/detect", files={"file": ("s.png", SAMPLE.read_bytes())})
    assert r.status_code == 413


def test_history_records_requests(client):
    rows = client.get("/api/v1/history").json()
    assert rows and {"n_cars", "latency_ms", "filename"} <= rows[0].keys()
    assert {"model", "filename"} <= rows[0].keys()
    hog = client.get("/api/v1/model", params={"model": "hog"}).json()
    assert hog["metadata"]["n_features"] == 972
