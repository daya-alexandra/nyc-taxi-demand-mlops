const flagLabels = {
  demand_spike: { text: "Demand spike", level: "danger" },
  weather_risk: { text: "Weather risk", level: "warning" },
  missing_lag_signal: { text: "Missing lags", level: "warning" },
  high_error: { text: "High error", level: "danger" },
  high_demand: { text: "High demand", level: "warning" },
};

const views = document.querySelectorAll(".view");
const tabButtons = document.querySelectorAll(".tab-button");
const predictionForm = document.querySelector("#prediction-form");
const predictionValue = document.querySelector("#prediction-value");
const predictionTime = document.querySelector("#prediction-time");
const predictionFlags = document.querySelector("#prediction-flags");
const predictionsTable = document.querySelector("#predictions-table");
const predictionCount = document.querySelector("#prediction-count");
const experimentsTable = document.querySelector("#experiments-table");
const driftList = document.querySelector("#drift-list");
const driftStatus = document.querySelector("#drift-status");
const retrainButton = document.querySelector("#retrain-button");
const toast = document.querySelector("#toast");
const systemStatus = document.querySelector("#system-status");
const modelMetric = document.querySelector("#model-metric");
const historyMetric = document.querySelector("#history-metric");
const driftMetric = document.querySelector("#drift-metric");
const registryMetric = document.querySelector("#registry-metric");
const dataDriftMetric = document.querySelector("#data-drift-metric");
const targetDriftMetric = document.querySelector("#target-drift-metric");
const conceptDriftMetric = document.querySelector("#concept-drift-metric");

function switchView(viewName) {
  views.forEach((view) => {
    view.classList.toggle("active", view.id === `view-${viewName}`);
  });
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewName);
  });
  window.history.replaceState(null, "", viewName === "inference" ? "/ui" : `/ui/${viewName}`);
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 3600);
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(2);
}

function renderFlags(flags) {
  if (!flags || flags.length === 0) {
    return '<span class="flag">Normal</span>';
  }

  return flags
    .map((flag) => {
      const meta = flagLabels[flag] || { text: flag, level: "" };
      return `<span class="flag ${meta.level}">${meta.text}</span>`;
    })
    .join("");
}

function formPayload(form) {
  const payload = {};
  const data = new FormData(form);

  data.forEach((value, key) => {
    payload[key] = Number(value);
  });

  return payload;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json();
}

async function submitPrediction(event) {
  event.preventDefault();

  const payload = formPayload(predictionForm);
  const result = await requestJson("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  predictionValue.textContent = formatNumber(result.predicted_trip_count);
  predictionTime.textContent = result.created_at || "online";
  predictionFlags.innerHTML = renderFlags(result.anomaly_flags);

  await loadPredictions();
}

async function loadPredictions() {
  const data = await requestJson("/api/predictions");
  predictionCount.textContent = `${data.items.length} rows`;
  historyMetric.textContent = `${data.items.length} rows`;

  predictionsTable.innerHTML = data.items
    .map(
      (item) => `
        <tr>
          <td>${item.created_at}</td>
          <td>${item.source}</td>
          <td>${item.pu_location_id}</td>
          <td>${item.hour}</td>
          <td>${formatNumber(item.predicted_trip_count)}</td>
          <td>${formatNumber(item.actual_trip_count)}</td>
          <td>${formatNumber(item.absolute_error)}</td>
          <td>${renderFlags(item.anomaly_flags)}</td>
        </tr>
      `,
    )
    .join("");
}

async function loadDrift() {
  const data = await requestJson("/api/drift");
  driftStatus.textContent = data.summary.status;
  driftMetric.textContent = `${data.summary.active_alerts} active`;
  dataDriftMetric.textContent = data.summary.data_drift_features ?? "--";
  targetDriftMetric.textContent = data.summary.target_drift_severity || "--";
  conceptDriftMetric.textContent = data.summary.concept_drift_severity || "--";

  driftList.innerHTML = data.items
    .map(
      (item) => `
        <div class="alert-row">
          <div class="alert-title">${item.title}</div>
          <div class="alert-message">${item.message}</div>
          <span class="flag warning">${item.severity}</span>
        </div>
      `,
    )
    .join("");
}

async function loadExperiments() {
  const data = await requestJson("/api/experiments");
  registryMetric.textContent = data.items[0]?.registry_status || "unknown";

  experimentsTable.innerHTML = data.items
    .map(
      (item) => `
        <tr>
          <td>${item.name}</td>
          <td>${item.experiment}</td>
          <td>${item.model}</td>
          <td>${formatNumber(item.mae)}</td>
          <td>${formatNumber(item.rmse)}</td>
          <td>${formatNumber(item.r2)}</td>
          <td>${item.test_rows || "--"}</td>
          <td><span class="flag warning">${item.registry_status}</span></td>
        </tr>
      `,
    )
    .join("");
}

async function requestRetrain() {
  const data = await requestJson("/api/retrain", { method: "POST" });
  showToast(`${data.message} Command: ${data.command}`);
}

async function loadHealth() {
  const data = await requestJson("/health");
  systemStatus.textContent = data.model_file_exists ? "model ready" : "model missing";
  modelMetric.textContent = data.model_file_exists ? "ready" : "missing";
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

predictionForm.addEventListener("submit", (event) => {
  submitPrediction(event).catch((error) => showToast(error.message));
});

retrainButton.addEventListener("click", () => {
  requestRetrain().catch((error) => showToast(error.message));
});

function currentViewFromPath() {
  if (window.location.pathname.endsWith("/experiments")) {
    return "experiments";
  }
  if (window.location.pathname.endsWith("/monitoring")) {
    return "monitoring";
  }
  return "inference";
}

switchView(currentViewFromPath());
loadHealth().catch(() => {
  systemStatus.textContent = "api unavailable";
});
loadPredictions().catch((error) => showToast(error.message));
loadDrift().catch((error) => showToast(error.message));
loadExperiments().catch((error) => showToast(error.message));
