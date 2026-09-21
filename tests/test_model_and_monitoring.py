from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.predict_model import make_predictions
from src.models.train_model import calculate_metrics, display_tracking_uri, split_train_test
from src.monitoring.calculate_drift import (
    ABSOLUTE_ERROR_COL,
    PICKUP_HOUR_COL,
    calculate_concept_drift,
    numeric_psi,
    split_reference_current,
)


class ConstantModel:
    def predict(self, features):
        return np.full(len(features), -2.0)


def test_time_split_has_no_timestamp_leakage() -> None:
    data = pd.DataFrame(
        {
            "pickup_hour": np.repeat(pd.date_range("2024-01-01", periods=10, freq="h"), 2),
            "PULocationID": [1, 2] * 10,
        }
    )

    train, test = split_train_test(data, test_size=0.2)

    assert train["pickup_hour"].max() < test["pickup_hour"].min()
    assert test["pickup_hour"].nunique() == 2


def test_metrics_match_known_values() -> None:
    metrics = calculate_metrics(pd.Series([0.0, 2.0]), np.array([0.0, 4.0]))

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == np.sqrt(2)
    assert metrics["r2"] == -1.0


def test_tracking_uri_is_safe_for_report() -> None:
    assert display_tracking_uri("sqlite:////private/path/mlflow.db") == "sqlite:///mlflow.db"
    assert (
        display_tracking_uri("https://alice:secret@mlflow.example:5000/api")
        == "https://mlflow.example:5000/api"
    )


def test_batch_predictions_are_clipped_at_zero() -> None:
    data = pd.DataFrame(
        {
            "pickup_hour": [pd.Timestamp("2024-01-01")],
            "PULocationID": [1],
            "trip_count": [3],
            "feature": [10],
        }
    )

    result = make_predictions(
        data,
        {"model": ConstantModel(), "feature_columns": ["feature"]},
    )

    assert result["predicted_trip_count"].iloc[0] == 0
    assert result["absolute_error"].iloc[0] == 3


def test_numeric_psi_detects_shift() -> None:
    reference = pd.Series(np.arange(100, dtype=float))

    identical = numeric_psi(reference, reference.copy())
    shifted = numeric_psi(reference, reference + 1_000)

    assert identical == 0.0
    assert shifted > 0.2


def test_monitoring_windows_do_not_split_one_hour() -> None:
    data = pd.DataFrame(
        {
            PICKUP_HOUR_COL: np.repeat(pd.date_range("2024-01-01", periods=10, freq="h"), 3),
            "value": range(30),
        }
    )

    reference, current = split_reference_current(data, window_fraction=0.2)

    assert len(reference) == len(current) == 6
    assert set(reference[PICKUP_HOUR_COL]).isdisjoint(set(current[PICKUP_HOUR_COL]))


def test_concept_drift_compares_adjacent_recent_error_windows() -> None:
    hours = pd.date_range("2024-01-01", periods=20, freq="h")
    predictions = pd.DataFrame(
        {
            PICKUP_HOUR_COL: hours,
            ABSOLUTE_ERROR_COL: [1.0] * 18 + [4.0, 4.0],
        }
    )

    report = calculate_concept_drift(predictions)

    assert report["severity"] == "critical"
    assert report["metrics"]["mae_ratio"] == 4.0
