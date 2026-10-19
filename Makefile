PY ?= .venv/bin/python
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

.PHONY: setup train evaluate train-cnn evaluate-cnn pipeline test lint format serve mlflow bento bento-serve docker docker-run clean

setup:            ## create .venv and install all dependencies
	python3.11 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements-dev.txt

train:            ## train + log to MLflow + register model
	$(PY) -m cardet.train

evaluate:         ## UIUC-protocol evaluation + quality gate
	$(PY) -m cardet.evaluate --gate

train-cnn:        ## fine-tune the any-angle CNN (needs CarData/cardata.zip; ~35 min on a laptop CPU)
	$(PY) -m cardet.train_cnn

evaluate-cnn:     ## CNN on UIUC (official protocol) + Stanford test split + quality gate
	$(PY) -m cardet.evaluate --model cnn --gate

pipeline:         ## full reproducible pipeline (only re-runs what changed)
	$(PY) -m dvc repro

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

serve:            ## FastAPI app on http://localhost:8000
	$(PY) -m uvicorn app.main:app --reload --port 8000

mlflow:           ## MLflow UI on http://localhost:5000
	$(PY) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

bento:            ## import model into BentoML store and build a Bento
	$(PY) -m cardet.bento_export && .venv/bin/bentoml build

bento-serve:      ## BentoML service on http://localhost:3000
	.venv/bin/bentoml serve service:CarDetectorService --port 3000

docker:
	docker build -t car-detector:local .

docker-run: docker
	docker run --rm -p 8000:8000 car-detector:local

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ var/
