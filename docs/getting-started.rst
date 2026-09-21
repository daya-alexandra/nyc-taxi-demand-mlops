Getting started
===============

The canonical setup and run instructions are maintained in the repository
``README.md``. In short: create a Python 3.10 virtual environment, install
``requirements-dev-lock.txt``, run ``python -m dvc repro --force`` and start
``python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000``.

A clean clone uses a deterministic sample. Real mode requires the four NYC TLC
parquet files for March--June 2024 and the matching Open-Meteo CSV in
``data/raw``. The selected source is always written to
``reports/data_profile.json``.
