"""FastAPI web app + REST API for the car detectors.

    uvicorn app.main:app --reload            # http://localhost:8000

Two models are served side by side:
  cnn  SSDlite fine-tuned on Stanford Cars + UIUC — cars from any angle (default)
  hog  HOG + calibrated logistic regression — side views only, classic baseline

Environment variables
  CARDET_MODEL_PATH      HOG bundle (default models/car_detector.joblib)
  CARDET_CNN_MODEL_PATH  CNN checkpoint (default models/cnn_detector.pt)
  CARDET_DB_URL          SQLAlchemy URL for history (default sqlite:///var/cardet.db)
  CARDET_MAX_UPLOAD_MB   upload size limit (default 8)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import joblib
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.db import Database
from cardet import __version__
from cardet.config import MODEL_PATH, REPORTS_DIR, ROOT
from cardet.features import decode_image_color, to_gray
from cardet.visualize import encode_png

log = logging.getLogger("cardet.api")
HERE = Path(__file__).parent
SAMPLES_DIR = HERE / "static" / "samples"
CNN_MODEL_PATH = ROOT / "models" / "cnn_detector.pt"
MIN_THRESHOLD = 0.05
ModelName = Literal["cnn", "hog"]
MODEL_LABELS = {"cnn": "Any angle", "hog": "Side view"}


class BoxOut(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float


class DetectResponse(BaseModel):
    filename: str
    model: ModelName
    count: int
    threshold: float
    detections: list[BoxOut]
    image_width: int
    image_height: int
    resize_factor: float
    latency_ms: float
    model_version: str
    image_png_base64: str | None = None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_detectors() -> dict:
    dets = {}
    hog_path = Path(os.environ.get("CARDET_MODEL_PATH", MODEL_PATH))
    if hog_path.exists():
        dets["hog"] = joblib.load(hog_path)
    cnn_path = Path(os.environ.get("CARDET_CNN_MODEL_PATH", CNN_MODEL_PATH))
    if cnn_path.exists():
        from cardet.cnn import CnnDetector

        dets["cnn"] = CnnDetector.load(cnn_path)
    for name in ("cnn", "hog"):
        if name not in dets:
            log.warning("Model '%s' not found — train it to enable it.", name)
    return dets


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.environ.get("CARDET_DB_URL", f"sqlite:///{ROOT / 'var' / 'cardet.db'}")
    if db_url.startswith("sqlite:///"):
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    app.state.db = Database(db_url)
    app.state.detectors = load_detectors()
    app.state.default_model = "cnn" if "cnn" in app.state.detectors else "hog"
    app.state.metrics = {
        "hog": load_json(REPORTS_DIR / "metrics.json"),
        "cnn": load_json(REPORTS_DIR / "cnn" / "metrics.json"),
    }
    yield


app = FastAPI(
    title="Car Detection API",
    version=__version__,
    description="Car detection served by two models: an SSDlite CNN fine-tuned on Stanford Cars "
    "+ UIUC (any viewing angle) and a classic HOG + logistic-regression sliding-window "
    "detector (side views).",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def get_detector(request: Request, model: str | None):
    name = model or request.app.state.default_model
    det = request.app.state.detectors.get(name)
    if det is None:
        raise HTTPException(503, f"Model '{name}' is not loaded. Train it first (see README).")
    return name, det


async def run_detection(
    request: Request,
    file: UploadFile,
    threshold: float | None,
    return_image: bool,
    source: str,
    model: str | None,
) -> DetectResponse:
    name, det = get_detector(request, model)
    limit = int(float(os.environ.get("CARDET_MAX_UPLOAD_MB", 8)) * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File too large (max {limit // 1024 // 1024} MB).")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        img = decode_image_color(data)
    except ValueError as e:
        raise HTTPException(415, str(e)) from e
    if name == "hog":
        img = to_gray(img)  # the HOG model was trained on grayscale only

    thr = det.config.threshold if threshold is None else threshold
    # Detect at the lowest threshold once; the UI slider then filters client-side.
    floor = min(thr, MIN_THRESHOLD) if return_image else thr
    res = await run_in_threadpool(det.detect, img, floor)
    kept = [b for b in res.boxes if b.score >= thr]

    analysed = img
    if res.resize_factor != 1.0:
        analysed = cv2.resize(img, res.image_size, interpolation=cv2.INTER_AREA)

    request.app.state.db.add(
        filename=(file.filename or "upload")[:255],
        source=source,
        model=name,
        n_cars=len(kept),
        max_score=max((b.score for b in kept), default=0.0),
        threshold=thr,
        width=img.shape[1],
        height=img.shape[0],
        latency_ms=res.latency_ms,
    )

    boxes = res.boxes if return_image else kept
    return DetectResponse(
        filename=file.filename or "upload",
        model=name,
        count=len(kept),
        threshold=thr,
        detections=[BoxOut(**b.to_dict()) for b in boxes],
        image_width=res.image_size[0],
        image_height=res.image_size[1],
        resize_factor=round(res.resize_factor, 4),
        latency_ms=round(res.latency_ms, 1),
        model_version=det.metadata.get("trained_at", "unknown"),
        image_png_base64=base64.b64encode(encode_png(analysed)).decode() if return_image else None,
    )


# ── Pages ──────────────────────────────────────────────────────────────────────
def samples() -> list[str]:
    files = [p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg"}]
    # Colour photos (any angle) first, then the UIUC street scenes.
    return sorted(files, key=lambda n: (not n.startswith("angle"), n))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    dets = request.app.state.detectors
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "page": "detect",
            "samples": samples(),
            "models": [
                {"name": n, "label": MODEL_LABELS[n], "threshold": dets[n].config.threshold}
                for n in ("cnn", "hog")
                if n in dets
            ],
            "default_model": request.app.state.default_model,
            "model_ready": bool(dets),
        },
    )


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    dets = request.app.state.detectors
    db: Database = request.app.state.db
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "page": "dashboard",
            "stats": db.stats(),
            "rows": db.recent(50),
            "metrics": request.app.state.metrics.get("hog", {}),
            "cnn_metrics": request.app.state.metrics.get("cnn", {}),
            "meta": dets["hog"].metadata if "hog" in dets else {},
            "cnn_meta": dets["cnn"].metadata if "cnn" in dets else {},
        },
    )


@app.get("/about", response_class=HTMLResponse, include_in_schema=False)
async def about(request: Request):
    return templates.TemplateResponse(request, "about.html", {"page": "about"})


# ── API ────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/detect", response_model=DetectResponse, tags=["detection"])
async def detect(
    request: Request,
    file: UploadFile = File(..., description="PGM, PNG, JPEG or BMP image"),
    model: ModelName | None = Query(
        None, description="'cnn' (any angle, default) or 'hog' (side views, classic)"
    ),
    threshold: float | None = Query(
        None, ge=MIN_THRESHOLD, le=0.99, description="Min car probability (default from model)"
    ),
    return_image: bool = Query(
        False,
        description="Include the analysed image as base64 PNG "
        "and all boxes above 0.05 for client-side filtering",
    ),
    source: str = Query("api", pattern="^(api|ui)$", include_in_schema=False),
):
    """Detect cars. Returns bounding boxes in analysed-image pixels."""
    return await run_detection(request, file, threshold, return_image, source, model)


@app.get("/api/v1/history", tags=["monitoring"])
async def history(request: Request, limit: int = Query(50, ge=1, le=500)):
    return [r.to_dict() for r in request.app.state.db.recent(limit)]


@app.get("/api/v1/model", tags=["monitoring"])
async def model_info(request: Request, model: ModelName | None = None):
    name, det = get_detector(request, model)
    m = request.app.state.metrics.get(name, {})
    return {
        "model": name,
        "metadata": det.metadata,
        "threshold": det.config.threshold,
        "test_metrics": {k: m.get(k) for k in ("single", "scale", "stanford_test") if k in m},
    }


@app.get("/health", tags=["monitoring"])
async def health(request: Request):
    dets = request.app.state.detectors
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": bool(dets),
        "models": sorted(dets),
        "default_model": request.app.state.default_model,
    }
