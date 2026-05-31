"""
EXPERIMENT 11 — Module 4: FastAPI Web Application
==================================================
Routes:
  GET  /              → Upload form (index.html)
  POST /predict       → Upload image → predict → result.html
  POST /api/predict   → JSON API (for curl / BentoML testing)
  GET  /dashboard     → Prediction history (dashboard.html)
  GET  /health        → Health check JSON

Start with:
  uvicorn app.main:app --reload --port 8000

Then open: http://localhost:8000
"""

import os
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.predict import predict_image_path, predict_image_bytes
from app.db import SessionLocal, Prediction, init_db

# ── App Instance ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Car Detection API",
    description = "UIUC Car Detection — AutoML + BentoML + MLflow + DVC",
    version     = "1.0.0"
)

# ── Static Files & Templates ───────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ── Upload Directory ───────────────────────────────────────────────────────────
UPLOAD_DIR = "app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    """Create SQLite tables on first launch."""
    init_db()
    print("[FastAPI] Database initialized.")
    print("[FastAPI] App started — http://localhost:8000")


# ── DB Session Dependency ──────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page — image upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_upload(
    request : Request,
    file    : UploadFile = File(...),
    db      : Session    = Depends(get_db)
):
    """
    Receive uploaded image, run prediction, save result to DB, show result page.
    Accepts: .pgm, .png, .jpg, .jpeg
    """
    # ── Save file ──────────────────────────────────────────────────────────────
    ext       = os.path.splitext(file.filename)[-1].lower() or ".pgm"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ── Predict ────────────────────────────────────────────────────────────────
    try:
        result = predict_image_path(save_path)
    except FileNotFoundError as e:
        # Model not trained yet
        return templates.TemplateResponse("result.html", {
            "request"  : request,
            "error"    : str(e),
            "filename" : file.filename
        })
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request"  : request,
            "error"    : f"Prediction failed: {e}",
            "filename" : file.filename
        })

    # ── Store in DB ────────────────────────────────────────────────────────────
    entry = Prediction(
        filename   = file.filename,
        result     = result["result"],
        confidence = result["confidence"]
    )
    db.add(entry)
    db.commit()

    return templates.TemplateResponse("result.html", {
        "request"    : request,
        "filename"   : file.filename,
        "result"     : result["result"],
        "confidence" : result["confidence"],
        "label"      : result["label"]
    })


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    JSON API endpoint — no HTML, pure JSON response.
    Used for: curl testing, BentoML integration, assignment JSON output screenshots.

    Example:
      curl -X POST http://localhost:8000/api/predict \
           -F "file=@CarData/TestImages/test-1.pgm"

    Returns:
      {"label": 1, "result": "Car Detected", "confidence": 93.5}
    """
    img_bytes = await file.read()
    try:
        result = predict_image_bytes(img_bytes)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse(result)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Dashboard showing last 50 predictions and summary stats."""
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(50)
        .all()
    )
    total    = db.query(Prediction).count()
    cars     = db.query(Prediction).filter(Prediction.result == "Car Detected").count()
    non_cars = total - cars

    return templates.TemplateResponse("dashboard.html", {
        "request"     : request,
        "predictions" : predictions,
        "total"       : total,
        "cars"        : cars,
        "non_cars"    : non_cars
    })


@app.get("/health")
async def health():
    """Health check — returns 200 OK when service is running."""
    return {
        "status"  : "ok",
        "service" : "car-detection-fastapi",
        "version" : "1.0.0"
    }