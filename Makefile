PYTHON ?= python3
IMAGE ?= nyc-taxi-demand-api

.PHONY: help install data data-real features train predict drift repro repro-real api mlflow lint test security validate clean docker-build docker-run compose-up compose-down k8s-apply k8s-apply-base

help:
	@echo "Available commands:"
	@echo "  make install       Install pinned runtime and development dependencies"
	@echo "  make repro         Reproduce the complete DVC pipeline (auto/sample data)"
	@echo "  make repro-real    Require raw NYC files and reproduce the DVC pipeline"
	@echo "  make api           Run FastAPI and the Web UI locally"
	@echo "  make mlflow        Run the local MLflow server"
	@echo "  make validate      Run formatting, lint, security and tests"
	@echo "  make docker-build  Build a runnable API image (bootstraps a sample model)"
	@echo "  make compose-up    Start API, MLflow, Prometheus and Grafana"
	@echo "  make k8s-apply     Apply the local Minikube Kustomize overlay"
	@echo "  make k8s-apply-base Apply the GHCR/Argo CD base manifests"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev-lock.txt

data:
	$(PYTHON) src/data/make_dataset.py

data-real:
	NYC_TAXI_DATA_MODE=real $(PYTHON) src/data/make_dataset.py

features:
	$(PYTHON) src/features/build_features.py

train:
	$(PYTHON) src/models/train_model.py

predict:
	$(PYTHON) src/models/predict_model.py

drift:
	$(PYTHON) src/monitoring/calculate_drift.py

repro:
	$(PYTHON) -m dvc repro

repro-real:
	NYC_TAXI_DATA_MODE=real $(PYTHON) -m dvc repro --force

api:
	$(PYTHON) -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

mlflow:
	$(PYTHON) -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

lint:
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m flake8 --config=.flake8 src tests

test:
	$(PYTHON) -m pytest -q

security:
	$(PYTHON) -m bandit -q -r src

validate: lint security test

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage pytest-of-root

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

compose-up:
	docker compose -f infra/docker-compose.yml up --build

compose-down:
	docker compose -f infra/docker-compose.yml down

k8s-apply:
	kubectl apply -k infra/k8s/overlays/local

k8s-apply-base:
	kubectl apply -k infra/k8s
