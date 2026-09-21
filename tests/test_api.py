from __future__ import annotations

import json

from fastapi.testclient import TestClient

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


def clear_api_caches() -> None:
    api_module.load_model_package.cache_clear()
    api_module.load_baseline_metrics.cache_clear()
    api_module.load_data_profile.cache_clear()


def test_health_and_readiness_report_missing_model(monkeypatch, tmp_path) -> None:
    clear_api_caches()
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.joblib")
    client = TestClient(api_module.app)

    health = client.get("/health")
    readiness = client.get("/health/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["model_file_exists"] is False
    assert readiness.status_code == 503


def test_readiness_rejects_corrupt_model(monkeypatch, tmp_path) -> None:
    clear_api_caches()
    model_path = tmp_path / "model.joblib"
    model_path.write_text("not a model", encoding="utf-8")
    monkeypatch.setattr(api_module, "MODEL_PATH", model_path)
    client = TestClient(api_module.app)

    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/health/ready").status_code == 503


def test_ui_and_openapi_are_available() -> None:
    client = TestClient(api_module.app)

    assert client.get("/ui").status_code == 200
    assert "NYC Taxi Demand MLOps" in client.get("/ui").text
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == ("NYC Taxi Demand MLOps API")


def test_predict_maps_http_request_to_model_features(monkeypatch, tmp_path) -> None:
    clear_api_caches()
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(api_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(
        api_module,
        "load_model_package",
        lambda: {"model": DummyModel(), "feature_columns": FEATURE_COLUMNS},
    )
    client = TestClient(api_module.app)

    response = client.post("/predict", json=request_payload())

    assert response.status_code == 200
    assert response.json()["predicted_trip_count"] == 12.5
    assert response.json()["created_at"].endswith("+00:00")


def test_predict_returns_503_without_model(monkeypatch, tmp_path) -> None:
    clear_api_caches()
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.joblib")
    client = TestClient(api_module.app)

    response = client.post("/predict", json=request_payload())

    assert response.status_code == 503
    assert "DVC pipeline" in response.json()["detail"]


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
    client = TestClient(api_module.app)

    response = client.get("/api/drift")

    assert response.status_code == 200
    assert response.json()["summary"]["status"] == "ok"


def test_prometheus_endpoint_exports_quality_and_drift(monkeypatch, tmp_path) -> None:
    metrics_path = tmp_path / "baseline_metrics.json"
    drift_path = tmp_path / "drift_report.json"
    metrics_path.write_text(
        json.dumps({"model": {"mae": 1.2, "rmse": 2.3, "r2": 0.8}}),
        encoding="utf-8",
    )
    drift_path.write_text(
        json.dumps(
            {
                "summary": {"status": "warning", "active_alerts": 1},
                "items": [
                    {
                        "type": "data_drift",
                        "severity": "warning",
                        "metrics": {"max_psi": 0.15, "drifted_features": 1},
                    },
                    {
                        "type": "target_drift",
                        "severity": "ok",
                        "metrics": {"psi": 0.02},
                    },
                    {
                        "type": "concept_drift",
                        "severity": "ok",
                        "metrics": {"mae_ratio": 1.1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(api_module, "DRIFT_REPORT_PATH", drift_path)
    api_module.load_baseline_metrics.cache_clear()
    client = TestClient(api_module.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "taxi_model_mae 1.2" in response.text
    assert "taxi_data_drift_max_psi 0.15" in response.text
    assert "taxi_drift_active_alerts 1.0" in response.text


def test_retrain_request_runs_background_task(monkeypatch, tmp_path) -> None:
    if api_module.retrain_lock.locked():
        api_module.retrain_lock.release()

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(api_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(api_module, "RETRAIN_STATUS_PATH", reports_dir / "status.json")
    monkeypatch.setattr(api_module, "RETRAIN_REQUESTS_PATH", reports_dir / "events.jsonl")
    monkeypatch.setattr(api_module, "RETRAIN_ENABLED", True)

    def fake_pipeline(request_id: str, created_at: str) -> None:
        api_module.write_retrain_event(
            {
                "request_id": request_id,
                "status": "succeeded",
                "created_at": created_at,
                "message": "Test retraining completed.",
            }
        )
        api_module.retrain_lock.release()

    monkeypatch.setattr(api_module, "run_retrain_pipeline", fake_pipeline)
    client = TestClient(api_module.app)

    response = client.post("/api/retrain")
    retrain_status = client.get("/api/retrain/status")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert retrain_status.json()["status"] == "succeeded"
    assert (reports_dir / "events.jsonl").exists()
