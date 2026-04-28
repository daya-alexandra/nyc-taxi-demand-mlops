"""Train baseline model for hourly NYC taxi demand forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
MLFLOW_TRACKING_DIR = PROJECT_DIR / "mlruns"

INPUT_FILE_NAME = "model_features.parquet"
MODEL_FILE_NAME = "baseline_demand_model.joblib"
METRICS_FILE_NAME = "baseline_metrics.json"

EXPERIMENT_NAME = "nyc-taxi-demand-baseline"

PICKUP_HOUR_COL = "pickup_hour"
TARGET_COL = "trip_count"
TEST_SIZE = 0.2

MODEL_PARAMS = {
    "loss": "poisson",
    "learning_rate": 0.05,
    "max_iter": 200,
    "random_state": 42,
}

FEATURE_COLUMNS = [
    "PULocationID",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
]


def split_train_test(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time: earlier dates for train, later dates for test."""
    data = data.sort_values(PICKUP_HOUR_COL).copy()

    unique_hours = sorted(data[PICKUP_HOUR_COL].unique())
    split_index = int(len(unique_hours) * (1 - test_size))
    split_hour = unique_hours[split_index]

    train_data = data[data[PICKUP_HOUR_COL] < split_hour].copy()
    test_data = data[data[PICKUP_HOUR_COL] >= split_hour].copy()

    return train_data, test_data


def train_model(train_data: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Train baseline regression model."""
    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COL]

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train)

    return model


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate regression metrics."""
    predictions = np.clip(predictions, a_min=0, a_max=None)

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_model(
    model: HistGradientBoostingRegressor,
    test_data: pd.DataFrame,
) -> dict[str, dict[str, float] | int]:
    """Evaluate model and naive lag baselines on test data."""
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COL]

    model_predictions = model.predict(x_test)

    metrics = {
        "model": calculate_metrics(y_test, model_predictions),
        "naive_lag_1h": calculate_metrics(y_test, test_data["lag_1h"]),
        "naive_lag_24h": calculate_metrics(y_test, test_data["lag_24h"]),
        "naive_lag_168h": calculate_metrics(y_test, test_data["lag_168h"]),
        "test_rows": int(len(test_data)),
    }

    return metrics


def flatten_metrics(metrics: dict[str, dict[str, float] | int]) -> dict[str, float]:
    """Flatten nested metrics for MLflow logging."""
    flat_metrics: dict[str, float] = {}

    for metric_group, metric_values in metrics.items():
        if isinstance(metric_values, dict):
            for metric_name, metric_value in metric_values.items():
                flat_metrics[f"{metric_group}_{metric_name}"] = float(metric_value)
        else:
            flat_metrics[metric_group] = float(metric_values)

    return flat_metrics


def log_to_mlflow(
    model: HistGradientBoostingRegressor,
    metrics: dict[str, dict[str, float] | int],
    model_path: Path,
    metrics_path: Path,
    train_rows: int,
    test_rows: int,
) -> None:
    """Log experiment parameters, metrics and artifacts to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline_hist_gradient_boosting"):
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_params(MODEL_PARAMS)

        mlflow.log_metrics(flatten_metrics(metrics))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(metrics_path), artifact_path="artifacts")


def main() -> None:
    """Train baseline model and save model artifact with metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = PROCESSED_DATA_DIR / INPUT_FILE_NAME
    model_path = MODELS_DIR / MODEL_FILE_NAME
    metrics_path = REPORTS_DIR / METRICS_FILE_NAME

    data = pd.read_parquet(input_path)
    data[PICKUP_HOUR_COL] = pd.to_datetime(data[PICKUP_HOUR_COL])

    train_data, test_data = split_train_test(data)

    model = train_model(train_data)
    metrics = evaluate_model(model, test_data)

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COL,
    }

    joblib.dump(model_package, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    log_to_mlflow(
        model=model,
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        train_rows=len(train_data),
        test_rows=len(test_data),
    )

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    print(f"Metrics: {metrics}")
    print(f"MLflow tracking directory: {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
