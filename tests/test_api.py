from __future__ import annotations

import json

from src.api import app as api_module

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


def request_payload() -> dict[str, int | float]:
    return {
        "pu_location_id": 161,
        "temperature_2m": 20.0,
        "relative_humidity_2m": 60.0,
        "precipitation": 0.0,
        "weather_code": 0.0,
        "wind_speed_10m": 10.0,
        "hour": 18,
        "day_of_week": 2,
        "day_of_month": 15,
        "month": 6,
        "is_weekend": 0,
        "lag_1h": 120.0,
        "lag_24h": 110.0,
        "lag_168h": 100.0,
        "rolling_mean_24h": 105.0,
    }


class DummyModel:
    def predict(self, features):
        assert list(features.columns) == FEATURE_COLUMNS
        return [12.5]


def test_health_check_returns_status() -> None:
    response = api_module.health_check()

    assert response["status"] == "ok"
    assert "model_file_exists" in response


def test_predict_maps_request_to_model_features(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "load_model_package",
        lambda: {"model": DummyModel(), "feature_columns": FEATURE_COLUMNS},
    )

    request = api_module.PredictionRequest(**request_payload())
    response = api_module.predict(request)

    assert response.predicted_trip_count == 12.5


def test_drift_endpoint_reads_report(monkeypatch, tmp_path) -> None:
    report_path = tmp_path / "drift_report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {"status": "ok", "active_alerts": 0},
                "items": [],
                "feature_drift": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "DRIFT_REPORT_PATH", report_path)

    response = api_module.drift_notifications()

    assert response["summary"]["status"] == "ok"


def test_retrain_request_is_logged(monkeypatch, tmp_path) -> None:
    requests_path = tmp_path / "retrain_requests.jsonl"
    monkeypatch.setattr(api_module, "RETRAIN_REQUESTS_PATH", requests_path)

    response = api_module.request_retrain()

    assert response.command == "dvc repro"
    assert requests_path.exists()
