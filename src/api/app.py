"""FastAPI service for NYC taxi demand prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "baseline_demand_model.joblib"

app = FastAPI(title="NYC Taxi Demand MLOps API")


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


@lru_cache
def load_model_package() -> dict:
    """Load trained model package from disk."""
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    """Check API health."""
    return {
        "status": "ok",
        "model_file_exists": MODEL_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict hourly taxi demand."""
    model_package = load_model_package()
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    features = pd.DataFrame([request.to_feature_dict()])
    features = features[feature_columns]

    prediction = model.predict(features)[0]
    prediction = float(np.clip(prediction, a_min=0, a_max=None))

    return PredictionResponse(predicted_trip_count=prediction)
