"""FastAPI service for NYC taxi demand prediction."""

from __future__ import annotations

import json
import os

# The service executes one fixed local DVC command without a shell.
import subprocess  # nosec B404
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "baseline_demand_model.joblib"
REPORTS_DIR = PROJECT_DIR / "reports"
WEB_DIR = PROJECT_DIR / "src" / "web"
METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "baseline_predictions.parquet"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_REPORT_HTML_PATH = REPORTS_DIR / "drift_report.html"
DATA_PROFILE_PATH = REPORTS_DIR / "data_profile.json"
RETRAIN_REQUESTS_PATH = REPORTS_DIR / "retrain_requests.jsonl"
RETRAIN_STATUS_PATH = REPORTS_DIR / "retrain_status.json"
RETRAIN_LOG_PATH = REPORTS_DIR / "retrain.log"
MAX_PREDICTION_HISTORY = 30
RETRAIN_TIMEOUT_SECONDS = int(os.getenv("RETRAIN_TIMEOUT_SECONDS", "1800"))
RETRAIN_ENABLED = os.getenv("RETRAIN_ENABLED", "true").lower() in {"1", "true", "yes"}

app = FastAPI(title="NYC Taxi Demand MLOps API")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

prediction_history: deque[dict[str, object]] = deque(maxlen=MAX_PREDICTION_HISTORY)
retrain_lock = threading.Lock()

PREDICTION_REQUESTS = Counter("taxi_prediction_requests", "Total prediction requests")
RETRAIN_REQUESTS = Counter("taxi_retrain_requests", "Total manual retrain requests")
PREDICTION_LATENCY = Histogram(
    "taxi_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
MODEL_AVAILABLE = Gauge("taxi_model_available", "Whether the model artifact is available")
LAST_PREDICTION_VALUE = Gauge("taxi_last_prediction_value", "Last predicted trip count")
LAST_ANOMALY_FLAGS = Gauge("taxi_last_anomaly_flags", "Number of anomaly flags for last prediction")
PREDICTED_DEMAND = Counter(
    "taxi_predicted_demand",
    "Cumulative predicted trips returned by the service",
)
ANOMALY_FLAGS = Counter(
    "taxi_anomaly_flags",
    "Anomaly flags emitted by prediction requests",
    ["flag"],
)
MODEL_MAE = Gauge("taxi_model_mae", "Latest offline model MAE")
MODEL_RMSE = Gauge("taxi_model_rmse", "Latest offline model RMSE")
MODEL_R2 = Gauge("taxi_model_r2", "Latest offline model R squared")
DATA_DRIFT_STATUS = Gauge(
    "taxi_data_drift_status",
    "Data drift status encoded as 0=ok, 1=warning, 2=critical, -1=pending",
)
DATA_DRIFT_MAX_PSI = Gauge("taxi_data_drift_max_psi", "Maximum feature PSI")
DATA_DRIFT_FEATURES = Gauge(
    "taxi_data_drift_features",
    "Number of drifted feature columns",
)
TARGET_DRIFT_PSI = Gauge("taxi_target_drift_psi", "Target drift PSI")
CONCEPT_DRIFT_MAE_RATIO = Gauge(
    "taxi_concept_drift_mae_ratio",
    "Concept drift current/reference MAE ratio",
)
DRIFT_ACTIVE_ALERTS = Gauge(
    "taxi_drift_active_alerts",
    "Number of active drift alerts",
)

DRIFT_STATUS_VALUES = {
    "ok": 0,
    "warning": 1,
    "critical": 2,
    "pending": -1,
    "not_generated": -1,
}


class PredictionRequest(BaseModel):
    """Input features for taxi demand prediction."""

    pu_location_id: int = Field(..., ge=1)
    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float = Field(..., ge=0)
    weather_code: float
    wind_speed_10m: float
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    day_of_month: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    lag_1h: float
    lag_24h: float
    lag_168h: float
    rolling_mean_24h: float

    def to_feature_dict(self) -> dict[str, float | int]:
        """Convert API input to model feature names."""
        return {
            "PULocationID": self.pu_location_id,
            "temperature_2m": self.temperature_2m,
            "relative_humidity_2m": self.relative_humidity_2m,
            "precipitation": self.precipitation,
            "weather_code": self.weather_code,
            "wind_speed_10m": self.wind_speed_10m,
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "day_of_month": self.day_of_month,
            "month": self.month,
            "is_weekend": self.is_weekend,
            "lag_1h": self.lag_1h,
            "lag_24h": self.lag_24h,
            "lag_168h": self.lag_168h,
            "rolling_mean_24h": self.rolling_mean_24h,
        }


class PredictionResponse(BaseModel):
    """Prediction response."""

    predicted_trip_count: float
    anomaly_flags: list[str] = Field(default_factory=list)
    created_at: str | None = None


class RetrainResponse(BaseModel):
    """Manual retraining request response."""

    status: str
    message: str
    command: str
    request_id: str


class RetrainStatusResponse(BaseModel):
    """State of the most recent manual retraining job."""

    request_id: str | None = None
    status: str = "not_requested"
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    command: str | None = None
    message: str | None = None


@lru_cache
def load_model_package() -> dict:
    """Load trained model package from disk."""
    return joblib.load(MODEL_PATH)


def model_is_ready() -> bool:
    """Check that the model can be deserialized and has the expected keys."""
    if not MODEL_PATH.exists():
        return False

    try:
        package = load_model_package()
    except Exception:
        return False

    return (
        isinstance(package, dict)
        and hasattr(package.get("model"), "predict")
        and isinstance(package.get("feature_columns"), list)
    )


@lru_cache
def load_baseline_metrics() -> dict:
    """Load baseline metrics for the experiments page."""
    if not METRICS_PATH.exists():
        return {}

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache
def load_data_profile() -> dict:
    """Load provenance for the dataset used in the latest DVC run."""
    if not DATA_PROFILE_PATH.exists():
        return {"source": "unknown", "is_synthetic": None}

    with DATA_PROFILE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_drift_report() -> dict[str, object]:
    """Load the latest drift report for API and UI."""
    if not DRIFT_REPORT_PATH.exists():
        return {
            "generated_at": None,
            "summary": {
                "status": "not_generated",
                "active_alerts": 0,
                "last_report": None,
            },
            "items": [
                {
                    "type": "data_drift",
                    "severity": "pending",
                    "title": "Data drift",
                    "message": "Run the DVC drift_report stage to generate drift metrics.",
                    "metrics": {},
                }
            ],
            "feature_drift": [],
        }

    with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def drift_item(report: dict[str, object], item_type: str) -> dict[str, object]:
    """Return one drift item by its type."""
    items = report.get("items", [])
    if not isinstance(items, list):
        return {}

    for item in items:
        if isinstance(item, dict) and item.get("type") == item_type:
            return item
    return {}


def metric_values(item: dict[str, object]) -> dict[str, object]:
    """Return a safely typed metrics mapping from a report item."""
    values = item.get("metrics", {})
    return values if isinstance(values, dict) else {}


def update_prometheus_report_metrics() -> None:
    """Export offline model-quality and drift values as Prometheus gauges."""
    offline_metrics = load_baseline_metrics().get("model", {})
    if isinstance(offline_metrics, dict):
        if offline_metrics.get("mae") is not None:
            MODEL_MAE.set(float(offline_metrics["mae"]))
        if offline_metrics.get("rmse") is not None:
            MODEL_RMSE.set(float(offline_metrics["rmse"]))
        if offline_metrics.get("r2") is not None:
            MODEL_R2.set(float(offline_metrics["r2"]))

    report = load_drift_report()
    summary = report.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    data_drift = drift_item(report, "data_drift")
    target_drift = drift_item(report, "target_drift")
    concept_drift = drift_item(report, "concept_drift")

    data_metrics = metric_values(data_drift)
    target_metrics = metric_values(target_drift)
    concept_metrics = metric_values(concept_drift)

    DATA_DRIFT_STATUS.set(
        DRIFT_STATUS_VALUES.get(str(data_drift.get("severity", "not_generated")), -1)
    )
    DATA_DRIFT_MAX_PSI.set(float(data_metrics.get("max_psi", 0) or 0))
    DATA_DRIFT_FEATURES.set(float(data_metrics.get("drifted_features", 0) or 0))
    TARGET_DRIFT_PSI.set(float(target_metrics.get("psi", 0) or 0))
    CONCEPT_DRIFT_MAE_RATIO.set(float(concept_metrics.get("mae_ratio", 0) or 0))
    DRIFT_ACTIVE_ALERTS.set(float(summary.get("active_alerts", 0) or 0))


def load_retrain_status() -> dict[str, object]:
    """Read the persisted state of the most recent retraining job."""
    if not RETRAIN_STATUS_PATH.exists():
        return {"request_id": None, "status": "not_requested"}
    try:
        with RETRAIN_STATUS_PATH.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"request_id": None, "status": "unknown"}
    return value if isinstance(value, dict) else {"request_id": None, "status": "unknown"}


def write_retrain_event(event: dict[str, object]) -> None:
    """Persist retraining status and append it to the audit log."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = RETRAIN_STATUS_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(event, indent=2), encoding="utf-8")
    temporary_path.replace(RETRAIN_STATUS_PATH)
    with RETRAIN_REQUESTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def run_retrain_pipeline(request_id: str, created_at: str) -> None:
    """Run the DVC pipeline in the background and hot-reload its model."""
    command = [sys.executable, "-m", "dvc", "repro", "--force"]
    command_environment = os.environ.copy()
    command_environment.setdefault("DVC_NO_ANALYTICS", "1")
    event: dict[str, object] = {
        "request_id": request_id,
        "status": "running",
        "created_at": created_at,
        "started_at": utc_now(),
        "finished_at": None,
        "command": "python -m dvc repro --force",
        "message": "Retraining pipeline is running.",
    }
    write_retrain_event(event)

    try:
        with RETRAIN_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{event['started_at']}] request_id={request_id}\n")
            result = subprocess.run(  # nosec B603
                command,
                cwd=PROJECT_DIR,
                env=command_environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=RETRAIN_TIMEOUT_SECONDS,
                check=False,
            )

        if result.returncode != 0:
            raise RuntimeError(f"DVC pipeline exited with code {result.returncode}")
        load_model_package.cache_clear()
        if not model_is_ready():
            raise RuntimeError(f"Retrained model is missing or invalid at {MODEL_PATH}")
        load_baseline_metrics.cache_clear()
        load_data_profile.cache_clear()
        event.update(
            {
                "status": "succeeded",
                "finished_at": utc_now(),
                "message": "Retraining completed and the API model cache was refreshed.",
            }
        )
    except subprocess.TimeoutExpired:
        event.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "message": f"Retraining timed out after {RETRAIN_TIMEOUT_SECONDS} seconds.",
            }
        )
    except Exception as error:
        event.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "message": str(error),
            }
        )
    finally:
        write_retrain_event(event)
        retrain_lock.release()


def utc_now() -> str:
    """Return current UTC timestamp for UI records."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_anomaly_flags(request: PredictionRequest, prediction: float) -> list[str]:
    """Flag simple inference anomalies for the UI."""
    flags: list[str] = []

    rolling_mean = max(request.rolling_mean_24h, 1)
    if prediction > rolling_mean * 1.8 and prediction - rolling_mean > 25:
        flags.append("demand_spike")

    if request.precipitation >= 5 or request.wind_speed_10m >= 35:
        flags.append("weather_risk")

    if min(request.lag_1h, request.lag_24h, request.lag_168h) <= 0:
        flags.append("missing_lag_signal")

    return flags


def add_prediction_record(
    request: PredictionRequest,
    prediction: float,
    flags: list[str],
    created_at: str,
) -> None:
    """Store latest API predictions for the web UI."""
    prediction_history.append(
        {
            "id": str(uuid4()),
            "created_at": created_at,
            "source": "online",
            "pu_location_id": request.pu_location_id,
            "hour": request.hour,
            "day_of_week": request.day_of_week,
            "predicted_trip_count": round(prediction, 2),
            "actual_trip_count": None,
            "absolute_error": None,
            "anomaly_flags": flags,
        }
    )


def report_prediction_records(limit: int = 20) -> list[dict[str, object]]:
    """Load recent batch predictions when online history is empty."""
    if not PREDICTIONS_PATH.exists():
        return []

    try:
        data = pd.read_parquet(PREDICTIONS_PATH).tail(limit).copy()
    except Exception:
        return []

    records: list[dict[str, object]] = []
    for row in data.to_dict(orient="records"):
        predicted = float(row["predicted_trip_count"])
        actual = float(row["trip_count"])
        absolute_error = float(row["absolute_error"])
        flags: list[str] = []

        if absolute_error >= max(20, predicted * 0.5):
            flags.append("high_error")
        if predicted >= 100:
            flags.append("high_demand")

        records.append(
            {
                "id": str(uuid4()),
                "created_at": str(row["pickup_hour"]),
                "source": "batch",
                "pu_location_id": int(row["PULocationID"]),
                "hour": pd.Timestamp(row["pickup_hour"]).hour,
                "day_of_week": pd.Timestamp(row["pickup_hour"]).dayofweek,
                "predicted_trip_count": round(predicted, 2),
                "actual_trip_count": round(actual, 2),
                "absolute_error": round(absolute_error, 2),
                "anomaly_flags": flags,
            }
        )

    return list(reversed(records))


@app.get("/health")
def health_check() -> dict[str, object]:
    """Return a diagnostic health summary without hiding a missing model."""
    model_available = model_is_ready()
    MODEL_AVAILABLE.set(1 if model_available else 0)
    return {
        "status": "ready" if model_available else "degraded",
        "model_file_exists": model_available,
        "ui_enabled": WEB_DIR.exists(),
        "drift_report_exists": DRIFT_REPORT_PATH.exists(),
        "retrain": load_retrain_status(),
    }


@app.get("/health/live")
def liveness_check() -> dict[str, str]:
    """Confirm that the HTTP process is alive."""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness_check() -> dict[str, str]:
    """Confirm that a model is present and inference can be served."""
    if not model_is_ready():
        MODEL_AVAILABLE.set(0)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is missing. Run `dvc repro` before serving the API.",
        )
    MODEL_AVAILABLE.set(1)
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict hourly taxi demand."""
    start_time = perf_counter()
    if not MODEL_PATH.exists():
        MODEL_AVAILABLE.set(0)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable. Run the DVC pipeline first.",
        )

    try:
        model_package = load_model_package()
    except Exception as error:
        MODEL_AVAILABLE.set(0)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact could not be loaded.",
        ) from error
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    features = pd.DataFrame([request.to_feature_dict()])
    features = features[feature_columns]

    prediction = model.predict(features)[0]
    prediction = float(np.clip(prediction, a_min=0, a_max=None))
    flags = detect_anomaly_flags(request, prediction)
    created_at = utc_now()

    add_prediction_record(
        request=request,
        prediction=prediction,
        flags=flags,
        created_at=created_at,
    )

    PREDICTION_REQUESTS.inc()
    PREDICTION_LATENCY.observe(perf_counter() - start_time)
    LAST_PREDICTION_VALUE.set(prediction)
    LAST_ANOMALY_FLAGS.set(len(flags))
    PREDICTED_DEMAND.inc(prediction)
    for flag in flags:
        ANOMALY_FLAGS.labels(flag=flag).inc()

    return PredictionResponse(
        predicted_trip_count=prediction,
        anomaly_flags=flags,
        created_at=created_at,
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui/monitoring", response_class=HTMLResponse)
@app.get("/ui/experiments", response_class=HTMLResponse)
def web_ui() -> HTMLResponse:
    """Serve the web UI shell."""
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/styles.css")
@app.get("/ui/styles.css")
def web_styles() -> FileResponse:
    """Serve UI styles for direct and nested UI routes."""
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
@app.get("/ui/app.js")
def web_script() -> FileResponse:
    """Serve UI script for direct and nested UI routes."""
    return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")


@app.get("/reports/drift")
def drift_html_report() -> FileResponse:
    """Serve generated drift HTML report."""
    if not DRIFT_REPORT_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Drift report has not been generated yet.",
        )
    return FileResponse(DRIFT_REPORT_HTML_PATH, media_type="text/html")


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics."""
    MODEL_AVAILABLE.set(1 if model_is_ready() else 0)
    update_prometheus_report_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/predictions")
def latest_predictions() -> dict[str, list[dict[str, object]]]:
    """Return recent online or batch predictions for the UI table."""
    if prediction_history:
        return {"items": list(reversed(prediction_history))}

    return {"items": report_prediction_records()}


@app.get("/api/drift")
def drift_notifications() -> dict[str, object]:
    """Return current drift notifications for the monitoring screen."""
    return load_drift_report()


@app.get("/api/data")
def data_profile() -> dict:
    """Return provenance of the dataset used for the latest training run."""
    return load_data_profile()


@app.get("/api/experiments")
def experiments() -> dict[str, list[dict[str, object]]]:
    """Return baseline experiment metrics for the UI."""
    metrics = load_baseline_metrics()
    model_metrics = metrics.get("model", {})
    registry_path = REPORTS_DIR / "model_registry.json"
    registry = {}
    profile = load_data_profile()

    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as file:
            registry = json.load(file)

    return {
        "items": [
            {
                "name": "baseline_hist_gradient_boosting",
                "experiment": "nyc-taxi-demand-baseline",
                "model": "HistGradientBoostingRegressor",
                "registry_status": registry.get("status", "not_registered"),
                "registered_model": registry.get("name", "nyc-taxi-demand-baseline"),
                "model_version": registry.get("version"),
                "dataset_source": profile.get("source", "unknown"),
                "is_synthetic": profile.get("is_synthetic"),
                "mae": model_metrics.get("mae"),
                "rmse": model_metrics.get("rmse"),
                "r2": model_metrics.get("r2"),
                "test_rows": metrics.get("test_rows"),
            }
        ]
    }


@app.post(
    "/api/retrain",
    response_model=RetrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_retrain(background_tasks: BackgroundTasks) -> RetrainResponse:
    """Queue one local DVC retraining run and reload the model on success."""
    if not RETRAIN_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retraining is disabled for this deployment.",
        )
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A retraining job is already running.",
        )

    request_id = str(uuid4())
    created_at = utc_now()
    event = {
        "request_id": request_id,
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "status": "queued",
        "command": "python -m dvc repro --force",
        "message": "Retraining request was queued.",
    }

    try:
        write_retrain_event(event)
        background_tasks.add_task(run_retrain_pipeline, request_id, created_at)
    except Exception:
        retrain_lock.release()
        raise

    RETRAIN_REQUESTS.inc()

    return RetrainResponse(
        status="queued",
        message="Запрос принят: DVC pipeline запущен в фоне.",
        command="python -m dvc repro --force",
        request_id=request_id,
    )


@app.get("/api/retrain/status", response_model=RetrainStatusResponse)
def retrain_status() -> RetrainStatusResponse:
    """Return the latest persisted retraining status."""
    return RetrainStatusResponse(**load_retrain_status())
