# NYC Taxi Demand MLOps

MLOps project for forecasting hourly demand for NYC Yellow Taxi trips.

The project uses NYC Yellow Taxi trip data, weather data, and taxi zone information to build an hourly `zone × hour` demand dataset. The target variable is the number of taxi trips that started in a given zone during a given hour.

## Project status

Implemented:

- Cookiecutter Data Science project structure
- GitHub Flow with Conventional Commits
- DVC data versioning
- Reproducible DVC pipeline
- Baseline demand forecasting model
- MLflow experiment tracking
- GitHub Actions CI checks
- FastAPI prediction service
- Docker and Docker Compose deployment
- Local Kubernetes deployment with minikube

Planned:

- Monitoring and drift detection
- Drift reports
- Optional MLflow Model Registry
- Extended CI/CD if an external deployment target is added

## Project structure

```text
.
├── .github
│   └── workflows
│       └── ci.yml          <- GitHub Actions CI checks
├── data
│   ├── raw                 <- Raw data tracked by DVC
│   ├── interim             <- Intermediate generated datasets
│   ├── processed           <- Model-ready generated datasets
│   └── external            <- External reference data
├── k8s
│   ├── deployment.yaml     <- Kubernetes deployment for minikube
│   └── service.yaml        <- Kubernetes service for API access
├── models                  <- Trained model artifacts
├── notebooks               <- Research notebooks
├── reports                 <- Metrics, predictions and generated reports
├── src
│   ├── api                 <- FastAPI prediction service
│   ├── data                <- Dataset creation scripts
│   ├── features            <- Feature engineering scripts
│   └── models              <- Training and prediction scripts
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml                <- DVC pipeline definition
├── dvc.lock                <- Reproducible DVC pipeline state
├── requirements.txt
└── requirements-lock.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data and DVC

Raw data and generated artifacts are not stored directly in Git. They are tracked with DVC.

The current DVC remote is local:

```text
C:\dvc-storage
```

To reproduce the pipeline:

```bash
dvc repro
```

To check pipeline status:

```bash
dvc status
```

To push DVC artifacts to the configured remote:

```bash
dvc push
```

Pipeline stages:

```text
data/raw
→ data/interim/hourly_demand.parquet
→ data/processed/model_features.parquet
→ models/baseline_demand_model.joblib
→ reports/baseline_metrics.json
→ reports/baseline_predictions.parquet
```

## Model

The baseline model predicts hourly taxi demand for each pickup zone.

Main features:

- pickup zone
- hour, day of week, month
- weather features
- lag features for previous demand
- rolling demand statistics

Latest metrics are saved to:

```text
reports/baseline_metrics.json
```

The model is also compared with naive lag baselines such as previous hour, previous day, and previous week demand.

## MLflow

Training logs experiment parameters, metrics, and artifacts to MLflow.

Run training:

```bash
python src/models/train_model.py
```

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Experiment name:

```text
nyc-taxi-demand-baseline
```

## API service

The project includes a FastAPI service for model inference.

Available endpoints:

- `GET /health`
- `POST /predict`

Run locally:

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "pu_location_id": 161,
  "temperature_2m": 20,
  "relative_humidity_2m": 60,
  "precipitation": 0,
  "weather_code": 0,
  "wind_speed_10m": 10,
  "hour": 18,
  "day_of_week": 2,
  "day_of_month": 15,
  "month": 6,
  "is_weekend": 0,
  "lag_1h": 120,
  "lag_24h": 110,
  "lag_168h": 100,
  "rolling_mean_24h": 105
}
```

Example response:

```json
{
  "predicted_trip_count": 121.8
}
```

## Docker

Build the API image:

```bash
docker build -t nyc-taxi-demand-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 nyc-taxi-demand-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker Compose

Run the API service with Docker Compose:

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Kubernetes with minikube

The project includes basic Kubernetes manifests for local deployment with minikube.

Start minikube:

```bash
minikube start --driver=docker
```

Load the local Docker image into minikube:

```bash
minikube image load nyc-taxi-demand-api:latest
```

Apply manifests:

```bash
minikube kubectl -- apply -f k8s/
```

Check resources:

```bash
minikube kubectl -- get pods
minikube kubectl -- get services
```

Open the service:

```bash
minikube service nyc-taxi-demand-api --url
```

Append `/docs` to the printed URL.

Stop minikube when finished:

```bash
minikube stop
```

## CI

The project uses GitHub Actions for basic CI checks.

Current checks:

- Black formatting check
- Flake8 linting
- environment test

Workflow file:

```text
.github/workflows/ci.yml
```

## Development workflow

The project follows GitHub Flow:

1. Create a branch for a specific task.
2. Commit changes using Conventional Commits.
3. Open a Pull Request to `main`.
4. Wait for CI checks.
5. Merge into `main`.

Example commit messages:

```text
feat(api): add FastAPI prediction service
feat(docker): add FastAPI service container
feat(k8s): add minikube deployment
docs(readme): add deployment instructions
```

## Next steps

- Add monitoring and drift detection
- Generate drift reports
- Add optional MLflow Model Registry
- Extend CI/CD if a deployment target is added
- Prepare final project documentation and presentation

- ## Monitoring and Drift Detection

В проект добавлен базовый monitoring pipeline с использованием библиотеки Evidently.

Реализованы два типа drift detection:

### 1. Data Drift Detection

Проверяется изменение распределения входных признаков между:
- reference data (`X_train`)
- current data (`X_test`)

Сохраняемый артефакт:
- `data_drift_report.html`

### 2. Prediction Drift Detection

Проверяется изменение распределения предсказаний модели между:
- reference predictions
- current predictions

Сохраняемый артефакт:
- `prediction_drift_report.html`

Все monitoring artifacts автоматически сохраняются в папку `artifacts/`.

Monitoring реализован с помощью библиотеки Evidently.

## Project Architecture

Проект реализует упрощённый MLOps pipeline:

1. Загрузка и подготовка данных
2. Feature engineering
3. Обучение baseline-модели
4. Логирование экспериментов в MLflow
5. Сохранение model artifacts
6. FastAPI inference service
7. Docker / docker-compose deployment
8. Kubernetes / minikube deployment
9. Monitoring и drift detection через Evidently

Pipeline проекта включает:
- data layer
- training layer
- serving layer
- monitoring layer

- ## Future Improvements

Возможные дальнейшие улучшения проекта:

- MLflow Model Registry
- Prometheus + Grafana monitoring
- Автоматический retraining pipeline
- Полноценный CI/CD deployment
- ArgoCD integration
- Online inference monitoring
