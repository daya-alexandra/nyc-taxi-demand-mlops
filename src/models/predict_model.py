"""Generate predictions using trained baseline demand model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

INPUT_FILE_NAME = "model_features.parquet"
MODEL_FILE_NAME = "baseline_demand_model.joblib"
OUTPUT_FILE_NAME = "baseline_predictions.parquet"

PICKUP_HOUR_COL = "pickup_hour"
PICKUP_ZONE_COL = "PULocationID"
TARGET_COL = "trip_count"
PREDICTION_COL = "predicted_trip_count"


def make_predictions(
    data: pd.DataFrame,
    model_package: dict,
) -> pd.DataFrame:
    """Generate model predictions and return prediction table."""
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    predictions = model.predict(data[feature_columns])
    predictions = np.clip(predictions, a_min=0, a_max=None)

    result = data[[PICKUP_HOUR_COL, PICKUP_ZONE_COL, TARGET_COL]].copy()
    result[PREDICTION_COL] = predictions
    result["absolute_error"] = (result[TARGET_COL] - result[PREDICTION_COL]).abs()

    return result


def main() -> None:
    """Load trained model and save predictions."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = PROCESSED_DATA_DIR / INPUT_FILE_NAME
    model_path = MODELS_DIR / MODEL_FILE_NAME
    output_path = REPORTS_DIR / OUTPUT_FILE_NAME

    data = pd.read_parquet(data_path)
    model_package = joblib.load(model_path)

    predictions = make_predictions(data, model_package)
    predictions.to_parquet(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Predictions shape: {predictions.shape}")
    print(
        f"Date range: {predictions[PICKUP_HOUR_COL].min()} — "
        f"{predictions[PICKUP_HOUR_COL].max()}"
    )


if __name__ == "__main__":
    main()
