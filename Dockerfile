# FastAPI web app + REST API (CNN + HOG models).   docker build -t car-detector . && docker run -p 8000:8000 car-detector
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARDET_DB_URL=sqlite:////data/cardet.db

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 10001 app && mkdir /data && chown app /data
COPY --chown=app cardet/ cardet/
COPY --chown=app app/ app/
COPY --chown=app models/car_detector.joblib models/cnn_detector.pt models/
COPY --chown=app reports/metrics.json reports/
COPY --chown=app reports/cnn/metrics.json reports/cnn/
COPY --chown=app params.yaml .

USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --proxy-headers"]
