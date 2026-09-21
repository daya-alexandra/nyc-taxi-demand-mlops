FROM python:3.10-slim AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    DVC_NO_ANALYTICS=1

WORKDIR /app

COPY pyproject.toml requirements.txt requirements-lock.txt ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-lock.txt


FROM dependencies AS model-builder

COPY .dvc ./.dvc
COPY .dvcignore dvc.yaml dvc.lock ./
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN python -m dvc config core.no_scm true

ARG NYC_TAXI_DATA_MODE=auto
ENV NYC_TAXI_DATA_MODE=${NYC_TAXI_DATA_MODE} \
    MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db

# A clean clone has no DVC cache or model. Build a deterministic sample model
# in that case; a complete locally reproduced real-data run is kept when supplied.
RUN if [ ! -s models/baseline_demand_model.joblib ] \
        || [ ! -s reports/data_profile.json ] \
        || [ ! -s reports/baseline_metrics.json ] \
        || [ ! -s reports/model_registry.json ] \
        || [ ! -s reports/drift_report.json ]; then \
        python -m dvc repro --force; \
    fi \
    && test -s models/baseline_demand_model.joblib \
    && test -s reports/data_profile.json \
    && test -s reports/baseline_metrics.json \
    && test -s reports/model_registry.json \
    && test -s reports/drift_report.json


FROM dependencies AS runtime

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && chown app:app /app

COPY --from=model-builder --chown=app:app /app/.dvc ./.dvc
COPY --from=model-builder --chown=app:app /app/.dvcignore /app/dvc.yaml /app/dvc.lock ./
COPY --from=model-builder --chown=app:app /app/data/raw ./data/raw
COPY --from=model-builder --chown=app:app /app/data/interim ./data/interim
COPY --from=model-builder --chown=app:app /app/data/processed ./data/processed
COPY --from=model-builder --chown=app:app /app/models ./models
COPY --from=model-builder --chown=app:app /app/reports ./reports

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
