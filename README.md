# NYC Taxi Demand MLOps

MLOps project for forecasting hourly demand for NYC Yellow Taxi trips.

The project includes data versioning with DVC, a reproducible baseline pipeline, and experiment tracking with MLflow.

## Project status

Implemented:

- Cookiecutter Data Science project structure
- GitHub Flow with Conventional Commits
- DVC raw data tracking
- Local DVC remote storage
- Reproducible DVC pipeline
- Baseline demand forecasting model
- MLflow experiment tracking

Planned:

- FastAPI prediction service
- Docker / docker-compose
- Kubernetes / minikube orchestration
- CI/CD
- Monitoring and drift detection
- Web interface

## Project structure

```text
├── data
│   ├── raw          <- Raw data tracked by DVC
│   ├── interim      <- Intermediate generated datasets
│   ├── processed    <- Model-ready generated datasets
│   └── external     <- External reference data
├── models           <- Trained model artifacts
├── notebooks        <- Research notebooks
├── reports          <- Metrics, predictions and generated reports
├── src
│   ├── data         <- Dataset creation scripts
│   ├── features     <- Feature engineering scripts
│   └── models       <- Training and prediction scripts
├── dvc.yaml         <- DVC pipeline definition
├── dvc.lock         <- Reproducible DVC pipeline state
├── requirements.txt
└── requirements-lock.txt