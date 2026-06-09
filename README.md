# NYC Taxi Demand MLOps

Production-like учебный MLOps-проект для прогноза почасового спроса на NYC Yellow Taxi.

Проект включает полный локальный цикл: подготовку данных, DVC pipeline, обучение baseline-модели, MLflow tracking и model registry, FastAPI inference service, Web UI, Docker/Docker Compose, Prometheus metrics, Grafana stack, drift reporting и Kubernetes/minikube-манифесты.

## Project status

| Area                                 | Status                                                   |
| ------------------------------------ | -------------------------------------------------------- |
| Dataset and baseline model           | Implemented                                              |
| Cookiecutter-style project structure | Implemented and cleaned up                               |
| Git flow and Conventional Commits    | Used during development                                  |
| DVC data and pipeline versioning     | Implemented                                              |
| MLflow experiment tracking           | Implemented                                              |
| MLflow Model Registry                | Implemented in training workflow                         |
| FastAPI inference service            | Implemented                                              |
| OpenAPI documentation                | Available at `/docs`                                     |
| Web UI                               | Implemented                                              |
| Docker image                         | Implemented                                              |
| Docker Compose MLOps stack           | Implemented                                              |
| Prometheus metrics                   | Implemented                                              |
| Grafana stack                        | Included; dashboards can be extended                     |
| Drift reporting                      | Implemented with PSI and MAE-ratio logic                 |
| Kubernetes/minikube manifests        | Implemented for local deployment                         |
| CI                                   | Implemented with lint/test/build checks                  |
| CD                                   | Partially prepared with Kubernetes and Argo CD manifests |

## What the project does

The model predicts expected taxi trip demand for a selected pickup zone and hour.

The prediction target is:

```text
predicted_trip_count = expected number of taxi trips from a pickup zone during a specific hour
```

The current model forecasts demand by pickup zone. It does not model full origin-destination routes.

## Project structure

```text
.
├── .github
│   └── workflows
│       └── ci.yml                  # GitHub Actions CI checks
├── data
│   ├── raw                         # Raw data tracked by DVC
│   ├── interim                     # Intermediate generated datasets
│   ├── processed                   # Model-ready generated datasets
│   └── external                    # External reference data
├── infra
│   ├── argocd
│   │   └── application.yaml        # Argo CD application manifest
│   ├── docker-compose.yml          # Local API/MLflow/Prometheus/Grafana stack
│   ├── grafana                     # Grafana provisioning files
│   ├── k8s                         # Kubernetes/minikube manifests
│   └── prometheus
│       └── prometheus.yml          # Prometheus scrape configuration
├── k8s                             # Additional local minikube API manifests
├── models                          # Trained model artifacts
├── notebooks                       # Research notebooks
├── reports                         # Metrics, predictions and drift reports
├── src
│   ├── api                         # FastAPI service and API routes
│   ├── data                        # Dataset creation scripts
│   ├── features                    # Feature engineering scripts
│   ├── models                      # Training and prediction scripts
│   ├── monitoring                  # Drift calculation logic
│   └── web                         # Web UI assets
├── tests                           # API and project tests
├── .dockerignore
├── .dvcignore
├── .flake8
├── .gitignore
├── Dockerfile
├── Makefile
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── requirements-lock.txt
```

## Setup

The project was developed with Python 3.10.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## DVC pipeline

The project uses DVC to version the data pipeline.

If the DVC remote is available, restore tracked data and artifacts:

```bash
dvc pull
```

Run the full pipeline:

```bash
python -m dvc repro
```

Main pipeline stages:

```text
data -> features -> train -> predict -> drift_report
```

The pipeline produces model artifacts, prediction outputs, metrics and drift reports.

## Model training and MLflow

Training logs metrics, parameters and model artifacts to MLflow.

Typical local run:

```bash
python src/models/train_model.py
```

If MLflow server is running locally:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python src/models/train_model.py
```

The training script registers the model as:

```text
nyc-taxi-demand-regressor
```

Generated artifacts include:

```text
models/baseline_demand_model.joblib
reports/baseline_metrics.json
reports/model_registry.json
```

## API and Web UI

Start the FastAPI service locally:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
Web UI:      http://127.0.0.1:8000/ui
OpenAPI:    http://127.0.0.1:8000/docs
Health:     http://127.0.0.1:8000/health
Metrics:    http://127.0.0.1:8000/metrics
Drift HTML: http://127.0.0.1:8000/reports/drift
```

Main endpoints:

```text
GET  /health
POST /predict
GET  /api/predictions
GET  /api/drift
GET  /api/experiments
POST /api/retrain
GET  /metrics
```

The Web UI includes:

* inference form;
* latest predictions table;
* model status;
* prediction history;
* drift alerts;
* links to MLflow, Prometheus and Grafana;
* retrain request button.

## Docker Compose stack

Docker Compose starts the local MLOps stack:

* FastAPI application;
* MLflow server;
* Prometheus;
* Grafana.

Run:

```bash
docker compose -f infra/docker-compose.yml up --build
```

Open:

```text
API/UI:     http://127.0.0.1:8000/ui
MLflow:     http://127.0.0.1:5000
Prometheus: http://127.0.0.1:9090/targets
Grafana:    http://127.0.0.1:3000
```

Default Grafana credentials:

```text
admin / admin
```

Stop the stack:

```bash
docker compose -f infra/docker-compose.yml down
```

## Monitoring

The API exposes Prometheus metrics at:

```text
/metrics
```

Prometheus scrapes:

```text
api:8000/metrics
prometheus:9090/metrics
```

The current monitoring stack validates that:

* the API container is reachable;
* `/metrics` is available;
* Prometheus can scrape the API;
* Grafana is available for dashboarding.

Grafana is included as part of the monitoring stack. Dashboards can be extended with additional panels for prediction count, latency, anomaly flags, retrain requests and drift status.

## Drift reporting

Drift logic is implemented in:

```text
src/monitoring/calculate_drift.py
```

The report includes:

* data drift using PSI over feature columns;
* target drift using PSI over `trip_count`;
* concept drift using MAE-ratio between reference and current windows.

Generated reports:

```text
reports/drift_report.json
reports/drift_report.html
```

The HTML report is available through the API:

```text
http://127.0.0.1:8000/reports/drift
```

## Kubernetes / Minikube

Local Kubernetes deployment is prepared with minikube manifests.

Start minikube:

```bash
minikube start --driver=docker
```

Build image for minikube:

```bash
eval $(minikube docker-env)
docker build -t nyc-taxi-demand-api:latest .
```

Apply manifests:

```bash
kubectl apply -f infra/k8s
```

Check resources:

```bash
kubectl get all
```

The Kubernetes setup is intended for local validation and demonstration.

## Argo CD

An Argo CD application manifest is included at:

```text
infra/argocd/application.yaml
```

It prepares the project for GitOps-style deployment.

Before using it in a real cluster, update:

* `repoURL`;
* target branch/revision;
* Kubernetes namespace;
* image name and tag;
* cluster/server configuration.

This part is prepared as a CD foundation, not as a fully automated production deployment.

## CI

GitHub Actions CI validates pull requests to `main`.

The CI workflow includes:

* dependency installation;
* formatting/lint checks;
* tests;
* Docker image build;
* Kubernetes manifest validation where applicable.

Local validation commands:

```bash
python -m black --check src tests
python -m flake8 --config=.flake8 src tests
python -m pytest -q
```

## Development workflow

The project uses GitHub Flow:

1. Keep `main` stable.
2. Create a branch for each task:

   * `feature/<short-name>`
   * `fix/<short-name>`
   * `docs/<short-name>`
   * `chore/<short-name>`
3. Use Conventional Commits:

   * `feat:`
   * `fix:`
   * `docs:`
   * `test:`
   * `chore:`
   * `refactor:`
4. Open a pull request into `main`.
5. Merge only after local checks and/or CI pass.

## Useful commands

Run tests:

```bash
python -m pytest -q
```

Run formatter check:

```bash
python -m black --check src tests
```

Run linter:

```bash
python -m flake8 --config=.flake8 src tests
```

Run DVC pipeline:

```bash
python -m dvc repro
```

Run Docker Compose stack:

```bash
docker compose -f infra/docker-compose.yml up --build
```

Stop Docker Compose stack:

```bash
docker compose -f infra/docker-compose.yml down
```

## Current limitations

This is an educational production-like MLOps project, not a full production system.

Current limitations:

* Grafana dashboards are included as a monitoring foundation and can be extended further.
* CD is prepared through Kubernetes and Argo CD manifests, but no external production cluster is configured.
* The retrain endpoint records or triggers retrain workflow behavior depending on local setup; it is not a full production retraining orchestrator.
* DVC remote configuration may need to be adjusted for another machine or team environment.
* The model predicts pickup-zone demand, not full origin-destination route demand.

## Final validation checklist

The project has been locally validated with:

```text
black check: passed
flake8 check: passed
pytest: passed
Docker Compose stack: started successfully
API/UI: available
MLflow: available
Prometheus: API target UP
Grafana: available
MLflow model registry: model registered
```

## License

This project is created for educational purposes.
