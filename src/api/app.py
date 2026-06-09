"""FastAPI service for NYC taxi demand prediction."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
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
RETRAIN_REQUESTS_PATH = REPORTS_DIR / "retrain_requests.jsonl"
MAX_PREDICTION_HISTORY = 30

app = FastAPI(title="NYC Taxi Demand MLOps API")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

prediction_history: deque[dict[str, object]] = deque(maxlen=MAX_PREDICTION_HISTORY)

PREDICTION_REQUESTS = Counter("taxi_prediction_requests", "Total prediction requests")
RETRAIN_REQUESTS = Counter("taxi_retrain_requests", "Total manual retrain requests")
PREDICTION_LATENCY = Histogram(
    "taxi_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
MODEL_AVAILABLE = Gauge(
    "taxi_model_available", "Whether the model artifact is available"
)
LAST_PREDICTION_VALUE = Gauge("taxi_last_prediction_value", "Last predicted trip count")
LAST_ANOMALY_FLAGS = Gauge(
    "taxi_last_anomaly_flags", "Number of anomaly flags for last prediction"
)


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


@lru_cache
def load_model_package() -> dict:
    """Load trained model package from disk."""
    return joblib.load(MODEL_PATH)


@lru_cache
def load_baseline_metrics() -> dict:
    """Load baseline metrics for the experiments page."""
    if not METRICS_PATH.exists():
        return {}

    with METRICS_PATH.open("r", encoding="utf-8") as file:
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
def health_check() -> dict[str, str | bool]:
    """Check API health."""
    MODEL_AVAILABLE.set(1 if MODEL_PATH.exists() else 0)
    return {
        "status": "ok",
        "model_file_exists": MODEL_PATH.exists(),
        "ui_enabled": WEB_DIR.exists(),
        "drift_report_exists": DRIFT_REPORT_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict hourly taxi demand."""
    start_time = perf_counter()
    model_package = load_model_package()
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
    return FileResponse(DRIFT_REPORT_HTML_PATH, media_type="text/html")


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics."""
    MODEL_AVAILABLE.set(1 if MODEL_PATH.exists() else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/predictions")
def latest_predictions() -> dict[str, list[dict[str, object]]]:
    """Return recent online or batch predictions for the UI table."""
    if prediction_history:
        return {"items": list(reversed(prediction_history))}

    return {"items": report_prediction_records()}


@app.get("/api/drift")
def drift_notifications() -> dict[str, object]:
    """Return drift notification placeholders for the monitoring screen."""
    return load_drift_report()


@app.get("/api/experiments")
def experiments() -> dict[str, list[dict[str, object]]]:
    """Return baseline experiment metrics for the UI."""
    metrics = load_baseline_metrics()
    model_metrics = metrics.get("model", {})
    registry_path = REPORTS_DIR / "model_registry.json"
    registry = {}

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
                "mae": model_metrics.get("mae"),
                "rmse": model_metrics.get("rmse"),
                "r2": model_metrics.get("r2"),
                "test_rows": metrics.get("test_rows"),
            }
        ]
    }


@app.post("/api/retrain", response_model=RetrainResponse)
def request_retrain() -> RetrainResponse:
    """Register a manual retraining request for the UI."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    request_id = str(uuid4())
    event = {
        "request_id": request_id,
        "created_at": utc_now(),
        "status": "manual_review_required",
        "command": "dvc repro",
    }

    with RETRAIN_REQUESTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

    RETRAIN_REQUESTS.inc()

    return RetrainResponse(
        status="manual_review_required",
        message=(
            "Запрос принят. Для текущей версии retrain "
            "запускается вручную через DVC."
        ),
        command="dvc repro",
        request_id=request_id,
    )
