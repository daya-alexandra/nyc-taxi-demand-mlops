from __future__ import annotations

import pandas as pd
import pytest

from src.data import make_dataset
from src.features.build_features import build_features


def test_sample_dataset_is_deterministic_and_complete() -> None:
    first = make_dataset.build_sample_hourly_demand_dataset()
    second = make_dataset.build_sample_hourly_demand_dataset()

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 58_560
    assert first["PULocationID"].nunique() == 20
    assert first["trip_count"].min() >= 0
    assert first["pickup_hour"].min() == pd.Timestamp("2024-03-01 00:00:00")
    assert first["pickup_hour"].max() == pd.Timestamp("2024-06-30 23:00:00")


def test_data_mode_requires_all_real_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(make_dataset.DATA_MODE_ENV, "real")

    with pytest.raises(FileNotFoundError, match="Real-data mode"):
        make_dataset.resolve_data_mode(tmp_path)

    for filename in make_dataset.EXPECTED_TRIP_FILE_NAMES:
        (tmp_path / filename).touch()
    (tmp_path / make_dataset.WEATHER_FILE_NAME).touch()

    mode, raw_files = make_dataset.resolve_data_mode(tmp_path)

    assert mode == "real"
    assert len(raw_files) == 5


def test_auto_mode_falls_back_to_declared_sample(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(make_dataset.DATA_MODE_ENV, "auto")

    mode, raw_files = make_dataset.resolve_data_mode(tmp_path)

    assert mode == "sample"
    assert raw_files == []


def test_weather_loader_accepts_units_in_column_names(tmp_path) -> None:
    weather_path = tmp_path / make_dataset.WEATHER_FILE_NAME
    weather_path.write_text(
        "latitude,40.74\nlongitude,-74.04\nelevation,51\n"
        "time,temperature_2m (°C),relative_humidity_2m (%),"
        "precipitation (mm),weather_code (wmo code),wind_speed_10m (km/h)\n"
        "2024-03-01T00:00,5.0,70,0.0,0,12.0\n",
        encoding="utf-8",
    )

    weather = make_dataset.load_weather(tmp_path)

    assert list(weather.columns) == [
        "pickup_hour",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
    ]


def test_lags_are_calculated_independently_by_zone() -> None:
    hours = pd.date_range("2024-01-01", periods=170, freq="h")
    rows = []
    for zone_id, offset in [(1, 0), (2, 1_000)]:
        for index, pickup_hour in enumerate(hours):
            rows.append(
                {
                    "pickup_hour": pickup_hour,
                    "PULocationID": zone_id,
                    "trip_count": index + offset,
                    "temperature_2m": 10.0,
                    "relative_humidity_2m": 50.0,
                    "precipitation": 0.0,
                    "weather_code": 0.0,
                    "wind_speed_10m": 5.0,
                }
            )

    features = build_features(pd.DataFrame(rows))
    zone_one = features[features["PULocationID"] == 1].iloc[0]
    zone_two = features[features["PULocationID"] == 2].iloc[0]

    assert zone_one["lag_168h"] == 0
    assert zone_two["lag_168h"] == 1_000
    assert zone_one["pickup_hour"] == pd.Timestamp("2024-01-08 00:00:00")
