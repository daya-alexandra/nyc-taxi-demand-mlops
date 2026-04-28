"""Build model features from hourly taxi demand dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE_NAME = "hourly_demand.parquet"
OUTPUT_FILE_NAME = "model_features.parquet"

PICKUP_HOUR_COL = "pickup_hour"
PICKUP_ZONE_COL = "PULocationID"
TARGET_COL = "trip_count"


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features."""
    data = data.copy()

    data["hour"] = data[PICKUP_HOUR_COL].dt.hour
    data["day_of_week"] = data[PICKUP_HOUR_COL].dt.dayofweek
    data["day_of_month"] = data[PICKUP_HOUR_COL].dt.day
    data["month"] = data[PICKUP_HOUR_COL].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    return data


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add demand lag features by taxi zone."""
    data = data.sort_values([PICKUP_ZONE_COL, PICKUP_HOUR_COL]).copy()

    grouped = data.groupby(PICKUP_ZONE_COL)[TARGET_COL]

    data["lag_1h"] = grouped.shift(1)
    data["lag_24h"] = grouped.shift(24)
    data["lag_168h"] = grouped.shift(168)

    data["rolling_mean_24h"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=24).mean()
    )

    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build final feature table for model training."""
    data = data.copy()
    data[PICKUP_HOUR_COL] = pd.to_datetime(data[PICKUP_HOUR_COL])

    weather_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
    ]

    for column in weather_columns:
        if column in data.columns:
            data[column] = data[column].ffill().bfill()

    data = add_time_features(data)
    data = add_lag_features(data)

    data = data.dropna().reset_index(drop=True)

    return data


def main() -> None:
    """Create and save processed feature dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    input_path = INTERIM_DATA_DIR / INPUT_FILE_NAME
    output_path = PROCESSED_DATA_DIR / OUTPUT_FILE_NAME

    data = pd.read_parquet(input_path)
    features = build_features(data)

    features.to_parquet(output_path, index=False)

    print(f"Saved features to: {output_path}")
    print(f"Features shape: {features.shape}")
    print(
        f"Date range: {features[PICKUP_HOUR_COL].min()} — {features[PICKUP_HOUR_COL].max()}"
    )


if __name__ == "__main__":
    main()
