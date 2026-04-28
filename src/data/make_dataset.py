"""Create hourly taxi demand dataset from raw NYC taxi and weather data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"

TRIP_FILE_PATTERN = "yellow_tripdata_2024-*.parquet"
WEATHER_FILE_NAME = "open-meteo-40.74N74.04W51m.csv"
ZONES_FILE_NAME = "NYC_Taxi_Zones_20260326.geojson"

PICKUP_DATETIME_COL = "tpep_pickup_datetime"
PICKUP_ZONE_COL = "PULocationID"
PICKUP_HOUR_COL = "pickup_hour"
TARGET_COL = "trip_count"

START_DATE = "2024-03-01"
END_DATE = "2024-07-01"


def load_taxi_trips(raw_data_dir: Path) -> pd.DataFrame:
    """Load taxi trip files and keep only pickup datetime and pickup zone."""
    trip_files = sorted(raw_data_dir.glob(TRIP_FILE_PATTERN))

    if not trip_files:
        raise FileNotFoundError(
            f"No taxi trip files found in {raw_data_dir} "
            f"with pattern {TRIP_FILE_PATTERN}"
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
        (trips[PICKUP_DATETIME_COL] >= START_DATE)
        & (trips[PICKUP_DATETIME_COL] < END_DATE)
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

    weather = weather.rename(
        columns={
            "time": PICKUP_HOUR_COL,
            "temperature_2m (°C)": "temperature_2m",
            "relative_humidity_2m (%)": "relative_humidity_2m",
            "precipitation (mm)": "precipitation",
            "weather_code (wmo code)": "weather_code",
            "wind_speed_10m (km/h)": "wind_speed_10m",
        }
    )

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
        trips.groupby([PICKUP_HOUR_COL, PICKUP_ZONE_COL])
        .size()
        .reset_index(name=TARGET_COL)
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

    return dataset.sort_values([PICKUP_HOUR_COL, PICKUP_ZONE_COL]).reset_index(
        drop=True
    )


def main() -> None:
    """Create and save interim hourly demand dataset."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    trips = load_taxi_trips(RAW_DATA_DIR)
    weather = load_weather(RAW_DATA_DIR)

    observed_zones = sorted(trips[PICKUP_ZONE_COL].unique())
    zone_ids = load_zone_ids(RAW_DATA_DIR, fallback_zones=observed_zones)

    dataset = build_hourly_demand_dataset(
        trips=trips,
        weather=weather,
        zone_ids=zone_ids,
    )

    output_path = INTERIM_DATA_DIR / "hourly_demand.parquet"
    dataset.to_parquet(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"Dataset shape: {dataset.shape}")
    print(
        f"Date range: {dataset[PICKUP_HOUR_COL].min()} — {dataset[PICKUP_HOUR_COL].max()}"
    )


if __name__ == "__main__":
    main()
