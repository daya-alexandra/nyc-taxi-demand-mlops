"""Calculate data, target and concept drift reports for taxi demand model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_features.parquet"
PREDICTIONS_PATH = PROJECT_DIR / "reports" / "baseline_predictions.parquet"
REPORTS_DIR = PROJECT_DIR / "reports"
DRIFT_JSON_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"

PICKUP_HOUR_COL = "pickup_hour"
TARGET_COL = "trip_count"
ABSOLUTE_ERROR_COL = "absolute_error"

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

PSI_WARNING_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.2
CONCEPT_WARNING_RATIO = 1.25
CONCEPT_CRITICAL_RATIO = 1.5


def utc_now() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def split_reference_current(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered table into reference and current windows."""
    data = data.sort_values(PICKUP_HOUR_COL).copy()
    split_index = int(len(data) * 0.8)
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def severity_from_psi(psi: float) -> str:
    """Convert PSI value into drift severity."""
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "ok"


def psi_from_distributions(
    reference: Iterable[float], current: Iterable[float]
) -> float:
    """Calculate population stability index from aligned distributions."""
    epsilon = 1e-6
    reference_values = np.asarray(list(reference), dtype=float) + epsilon
    current_values = np.asarray(list(current), dtype=float) + epsilon

    reference_values = reference_values / reference_values.sum()
    current_values = current_values / current_values.sum()

    psi = np.sum(
        (current_values - reference_values) * np.log(current_values / reference_values)
    )
    return float(psi)


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI for numeric columns."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())

    if len(edges) < 3:
        categories = sorted(set(reference.astype(str)).union(set(current.astype(str))))
        reference_counts = (
            reference.astype(str)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
        )
        current_counts = (
            current.astype(str)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
        )
        return psi_from_distributions(reference_counts, current_counts)

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = pd.cut(reference, bins=edges).value_counts(
        normalize=True, sort=False
    )
    current_counts = pd.cut(current, bins=edges).value_counts(
        normalize=True, sort=False
    )

    return psi_from_distributions(reference_counts, current_counts)


def calculate_data_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> list[dict[str, object]]:
    """Calculate PSI drift for every model feature."""
    feature_reports: list[dict[str, object]] = []

    for column in FEATURE_COLUMNS:
        psi = numeric_psi(reference[column], current[column])
        feature_reports.append(
            {
                "feature": column,
                "psi": round(psi, 6),
                "severity": severity_from_psi(psi),
                "reference_mean": round(float(reference[column].mean()), 6),
                "current_mean": round(float(current[column].mean()), 6),
            }
        )

    return sorted(feature_reports, key=lambda item: item["psi"], reverse=True)


def calculate_target_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, object]:
    """Calculate target drift on trip_count."""
    psi = numeric_psi(reference[TARGET_COL], current[TARGET_COL])
    return {
        "type": "target_drift",
        "severity": severity_from_psi(psi),
        "title": "Target drift",
        "message": f"PSI по target trip_count = {psi:.3f}.",
        "metrics": {
            "psi": round(psi, 6),
            "reference_mean": round(float(reference[TARGET_COL].mean()), 6),
            "current_mean": round(float(current[TARGET_COL].mean()), 6),
        },
    }


def calculate_concept_drift(predictions: pd.DataFrame) -> dict[str, object]:
    """Estimate concept drift through model error growth."""
    reference, current = split_reference_current(predictions)
    reference_mae = float(reference[ABSOLUTE_ERROR_COL].mean())
    current_mae = float(current[ABSOLUTE_ERROR_COL].mean())
    ratio = current_mae / max(reference_mae, 1e-6)

    if ratio >= CONCEPT_CRITICAL_RATIO:
        severity = "critical"
    elif ratio >= CONCEPT_WARNING_RATIO:
        severity = "warning"
    else:
        severity = "ok"

    return {
        "type": "concept_drift",
        "severity": severity,
        "title": "Concept drift",
        "message": f"MAE current/reference ratio = {ratio:.2f}.",
        "metrics": {
            "reference_mae": round(reference_mae, 6),
            "current_mae": round(current_mae, 6),
            "mae_ratio": round(ratio, 6),
        },
    }


def build_report(
    features: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, object]:
    """Build combined drift report."""
    reference_features, current_features = split_reference_current(features)
    feature_drift = calculate_data_drift(reference_features, current_features)
    drifted_features = [item for item in feature_drift if item["severity"] != "ok"]
    max_feature = (
        feature_drift[0]
        if feature_drift
        else {
            "feature": None,
            "psi": 0,
            "severity": "ok",
        }
    )

    if max_feature["severity"] == "critical":
        data_severity = "critical"
    elif drifted_features:
        data_severity = "warning"
    else:
        data_severity = "ok"

    data_drift_item = {
        "type": "data_drift",
        "severity": data_severity,
        "title": "Data drift",
        "message": (
            f"{len(drifted_features)} features above PSI threshold. "
            f"Max PSI: {max_feature['feature']} = {max_feature['psi']}."
        ),
        "metrics": {
            "drifted_features": len(drifted_features),
            "max_psi": max_feature["psi"],
            "max_psi_feature": max_feature["feature"],
        },
    }

    target_drift_item = calculate_target_drift(reference_features, current_features)
    concept_drift_item = calculate_concept_drift(predictions)
    items = [data_drift_item, target_drift_item, concept_drift_item]

    active_alerts = sum(item["severity"] in {"warning", "critical"} for item in items)
    status = "ok"
    if any(item["severity"] == "critical" for item in items):
        status = "critical"
    elif active_alerts:
        status = "warning"

    return {
        "generated_at": utc_now(),
        "summary": {
            "status": status,
            "active_alerts": active_alerts,
            "last_report": "reports/drift_report.html",
            "data_drift_features": len(drifted_features),
            "target_drift_severity": target_drift_item["severity"],
            "concept_drift_severity": concept_drift_item["severity"],
        },
        "items": items,
        "feature_drift": feature_drift,
    }


def write_html_report(report: dict[str, object]) -> None:
    """Write a compact standalone HTML drift report."""
    rows = "\n".join(
        "<tr>"
        f"<td>{item['feature']}</td>"
        f"<td>{item['psi']}</td>"
        f"<td>{item['severity']}</td>"
        f"<td>{item['reference_mean']}</td>"
        f"<td>{item['current_mean']}</td>"
        "</tr>"
        for item in report["feature_drift"]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NYC Taxi Drift Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #17211d; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d9e2dc; padding: 10px; text-align: left; }}
    th {{ background: #f5f7f4; }}
    .status {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #dff4ef;
    }}
  </style>
</head>
<body>
  <h1>NYC Taxi Drift Report</h1>
  <p>Generated at: {report['generated_at']}</p>
  <p>Status: <span class="status">{report['summary']['status']}</span></p>
  <h2>Feature Drift</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th>
        <th>PSI</th>
        <th>Severity</th>
        <th>Reference mean</th>
        <th>Current mean</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    DRIFT_HTML_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    """Calculate drift report files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(PROCESSED_DATA_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    features[PICKUP_HOUR_COL] = pd.to_datetime(features[PICKUP_HOUR_COL])
    predictions[PICKUP_HOUR_COL] = pd.to_datetime(predictions[PICKUP_HOUR_COL])

    report = build_report(features=features, predictions=predictions)

    with DRIFT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    write_html_report(report)

    print(f"Saved drift JSON to: {DRIFT_JSON_PATH}")
    print(f"Saved drift HTML to: {DRIFT_HTML_PATH}")


if __name__ == "__main__":
    main()
