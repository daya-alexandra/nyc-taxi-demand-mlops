"""Create hourly taxi demand dataset from raw NYC taxi and weather data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"
REPORTS_DIR = PROJECT_DIR / "reports"

EXPECTED_TRIP_FILE_NAMES = [f"yellow_tripdata_2024-{month:02d}.parquet" for month in range(3, 7)]
WEATHER_FILE_NAME = "open-meteo-40.74N74.04W51m.csv"
ZONES_FILE_NAME = "NYC_Taxi_Zones_20260326.geojson"

PICKUP_DATETIME_COL = "tpep_pickup_datetime"
PICKUP_ZONE_COL = "PULocationID"
PICKUP_HOUR_COL = "pickup_hour"
TARGET_COL = "trip_count"

START_DATE = "2024-03-01"
END_DATE = "2024-07-01"
DATA_MODE_ENV = "NYC_TAXI_DATA_MODE"
VALID_DATA_MODES = {"auto", "real", "sample"}
SYNTHETIC_ZONE_IDS = [
    4,
    13,
    48,
    68,
    79,
    100,
    107,
    132,
    138,
    141,
    142,
    148,
    161,
    162,
    170,
    186,
    230,
    234,
    236,
    237,
]


def load_taxi_trips(raw_data_dir: Path) -> pd.DataFrame:
    """Load taxi trip files and keep only pickup datetime and pickup zone."""
    trip_files = [raw_data_dir / name for name in EXPECTED_TRIP_FILE_NAMES]

    missing_files = [path.name for path in trip_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing taxi trip files in {raw_data_dir}: {', '.join(missing_files)}"
        )

    frames: list[pd.DataFrame] = []

    for trip_file in trip_files:
        frame = pd.read_parquet(
            trip_file,
            columns=[PICKUP_DATETIME_COL, PICKUP_ZONE_COL],
        )
        frames.append(frame)

    trips = pd.concat(frames, ignore_index=True)

    trips = trips.dropna(subset=[PICKUP_DATETIME_COL, PICKUP_ZONE_COL]).copy()
    trips[PICKUP_DATETIME_COL] = pd.to_datetime(trips[PICKUP_DATETIME_COL])
    trips = trips[
        (trips[PICKUP_DATETIME_COL] >= START_DATE) & (trips[PICKUP_DATETIME_COL] < END_DATE)
    ].copy()
    trips[PICKUP_ZONE_COL] = trips[PICKUP_ZONE_COL].astype(int)
    trips[PICKUP_HOUR_COL] = trips[PICKUP_DATETIME_COL].dt.floor("h")

    return trips[[PICKUP_HOUR_COL, PICKUP_ZONE_COL]]


def load_weather(raw_data_dir: Path) -> pd.DataFrame:
    """Load weather data and prepare weather features by hour."""
    weather_path = raw_data_dir / WEATHER_FILE_NAME

    if not weather_path.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_path}")

    weather = pd.read_csv(weather_path, skiprows=3)

    column_prefixes = {
        "time": PICKUP_HOUR_COL,
        "temperature_2m": "temperature_2m",
        "relative_humidity_2m": "relative_humidity_2m",
        "precipitation": "precipitation",
        "weather_code": "weather_code",
        "wind_speed_10m": "wind_speed_10m",
    }
    rename_columns = {}
    for column in weather.columns:
        for prefix, destination in column_prefixes.items():
            if column == prefix or column.startswith(f"{prefix} "):
                rename_columns[column] = destination
                break
    weather = weather.rename(columns=rename_columns)

    weather[PICKUP_HOUR_COL] = pd.to_datetime(weather[PICKUP_HOUR_COL])
    weather = weather[
        (weather[PICKUP_HOUR_COL] >= START_DATE) & (weather[PICKUP_HOUR_COL] < END_DATE)
    ].copy()

    weather_columns = [
        PICKUP_HOUR_COL,
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
    ]

    missing_columns = [column for column in weather_columns if column not in weather]
    if missing_columns:
        raise ValueError("Open-Meteo CSV is missing columns: " + ", ".join(missing_columns))

    return weather[weather_columns].copy()


def load_zone_ids(raw_data_dir: Path, fallback_zones: list[int]) -> list[int]:
    """Load taxi zone IDs from GeoJSON or use observed zones as fallback."""
    zones_path = raw_data_dir / ZONES_FILE_NAME

    if not zones_path.exists():
        return fallback_zones

    with zones_path.open("r", encoding="utf-8") as file:
        zones_geojson = json.load(file)

    zone_ids: list[int] = []

    for feature in zones_geojson.get("features", []):
        properties = feature.get("properties", {})

        for key in ["LocationID", "location_id", "locationid", "OBJECTID"]:
            if key in properties:
                zone_ids.append(int(properties[key]))
                break

    if not zone_ids:
        return fallback_zones

    return sorted(set(zone_ids))


def build_hourly_demand_dataset(
    trips: pd.DataFrame,
    weather: pd.DataFrame,
    zone_ids: list[int],
) -> pd.DataFrame:
    """Build full zone-hour grid and merge taxi demand with weather."""
    aggregated_trips = (
        trips.groupby([PICKUP_HOUR_COL, PICKUP_ZONE_COL]).size().reset_index(name=TARGET_COL)
    )

    all_hours = pd.date_range(
        start=trips[PICKUP_HOUR_COL].min(),
        end=trips[PICKUP_HOUR_COL].max(),
        freq="h",
    )

    full_grid = pd.MultiIndex.from_product(
        [all_hours, zone_ids],
        names=[PICKUP_HOUR_COL, PICKUP_ZONE_COL],
    ).to_frame(index=False)

    dataset = full_grid.merge(
        aggregated_trips,
        on=[PICKUP_HOUR_COL, PICKUP_ZONE_COL],
        how="left",
    )

    dataset[TARGET_COL] = dataset[TARGET_COL].fillna(0).astype(int)

    dataset = dataset.merge(weather, on=PICKUP_HOUR_COL, how="left")

    return dataset.sort_values([PICKUP_HOUR_COL, PICKUP_ZONE_COL]).reset_index(drop=True)


def build_sample_hourly_demand_dataset() -> pd.DataFrame:
    """Build a deterministic NYC-like sample for CI and a clean clone."""
    rng = np.random.default_rng(42)
    hours = pd.date_range(start=START_DATE, end=END_DATE, freq="h", inclusive="left")
    dataset = pd.MultiIndex.from_product(
        [hours, SYNTHETIC_ZONE_IDS],
        names=[PICKUP_HOUR_COL, PICKUP_ZONE_COL],
    ).to_frame(index=False)

    hour = dataset[PICKUP_HOUR_COL].dt.hour
    day_of_week = dataset[PICKUP_HOUR_COL].dt.dayofweek
    day_of_year = dataset[PICKUP_HOUR_COL].dt.dayofyear
    zone_index = dataset[PICKUP_ZONE_COL].map(
        {zone_id: index for index, zone_id in enumerate(SYNTHETIC_ZONE_IDS)}
    )

    commute_peak = np.where(hour.isin([7, 8, 9, 17, 18, 19]), 18, 0)
    nightlife_peak = np.where(hour.isin([21, 22, 23, 0, 1]), 9, 0)
    weekend_boost = np.where(day_of_week >= 5, 7, 0)
    zone_boost = (zone_index % 7) * 3
    seasonal = 5 * np.sin(day_of_year / 14)

    temperature = 15 + 10 * np.sin((day_of_year - 80) / 28) + rng.normal(0, 1.5, len(dataset))
    humidity = 58 + 18 * np.sin(day_of_year / 9) + rng.normal(0, 5, len(dataset))
    precipitation = np.clip(
        rng.gamma(shape=0.45, scale=1.2, size=len(dataset)) - 0.4,
        0,
        None,
    )
    wind_speed = np.clip(12 + rng.normal(0, 4, len(dataset)), 1, None)
    weather_code = np.where(
        precipitation > 2,
        61,
        np.where(precipitation > 0.2, 51, 0),
    )

    demand_signal = (
        12
        + commute_peak
        + nightlife_peak
        + weekend_boost
        + zone_boost
        + seasonal
        - precipitation * 1.2
    )

    dataset[TARGET_COL] = rng.poisson(np.clip(demand_signal, 1, None)).astype(int)
    dataset["temperature_2m"] = np.round(temperature, 2)
    dataset["relative_humidity_2m"] = np.round(np.clip(humidity, 20, 100), 2)
    dataset["precipitation"] = np.round(precipitation, 2)
    dataset["weather_code"] = weather_code.astype(float)
    dataset["wind_speed_10m"] = np.round(wind_speed, 2)

    return dataset


def resolve_data_mode(raw_data_dir: Path) -> tuple[str, list[Path]]:
    """Choose real data when complete, otherwise use the deterministic sample."""
    requested_mode = os.getenv(DATA_MODE_ENV, "auto").strip().lower()
    if requested_mode not in VALID_DATA_MODES:
        choices = ", ".join(sorted(VALID_DATA_MODES))
        raise ValueError(f"{DATA_MODE_ENV} must be one of: {choices}")

    trip_files = [raw_data_dir / name for name in EXPECTED_TRIP_FILE_NAMES]
    real_data_available = (
        all(path.exists() for path in trip_files) and (raw_data_dir / WEATHER_FILE_NAME).exists()
    )

    if requested_mode == "real" and not real_data_available:
        raise FileNotFoundError(
            "Real-data mode requires NYC TLC parquet files and the Open-Meteo CSV "
            f"inside {raw_data_dir}"
        )

    if requested_mode == "sample":
        return "sample", []
    if real_data_available:
        raw_files = [*trip_files, raw_data_dir / WEATHER_FILE_NAME]
        zones_path = raw_data_dir / ZONES_FILE_NAME
        if zones_path.exists():
            raw_files.append(zones_path)
        return "real", raw_files
    return "sample", []


def write_data_profile(dataset: pd.DataFrame, source: str, raw_files: list[Path]) -> None:
    """Persist dataset provenance used by DVC, MLflow and the Web UI."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    profile = {
        "source": source,
        "is_synthetic": source == "deterministic_nyc_like_sample",
        "rows": int(len(dataset)),
        "zones": int(dataset[PICKUP_ZONE_COL].nunique()),
        "start": str(dataset[PICKUP_HOUR_COL].min()),
        "end": str(dataset[PICKUP_HOUR_COL].max()),
        "raw_files": [path.name for path in raw_files],
    }
    profile_path = REPORTS_DIR / "data_profile.json"
    with profile_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)


def main() -> None:
    """Create and save interim hourly demand dataset."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    mode, raw_files = resolve_data_mode(RAW_DATA_DIR)

    if mode == "real":
        trips = load_taxi_trips(RAW_DATA_DIR)
        weather = load_weather(RAW_DATA_DIR)
        observed_zones = sorted(trips[PICKUP_ZONE_COL].unique())
        zone_ids = load_zone_ids(RAW_DATA_DIR, fallback_zones=observed_zones)
        dataset = build_hourly_demand_dataset(
            trips=trips,
            weather=weather,
            zone_ids=zone_ids,
        )
        source = "nyc_tlc_and_open_meteo"
    else:
        dataset = build_sample_hourly_demand_dataset()
        source = "deterministic_nyc_like_sample"

    output_path = INTERIM_DATA_DIR / "hourly_demand.parquet"
    dataset.to_parquet(output_path, index=False)
    write_data_profile(dataset=dataset, source=source, raw_files=raw_files)

    print(f"Saved dataset to: {output_path}")
    print(f"Dataset source: {source}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Date range: {dataset[PICKUP_HOUR_COL].min()} — {dataset[PICKUP_HOUR_COL].max()}")


if __name__ == "__main__":
    main()
