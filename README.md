# DiseasesIdentification

Lightweight Flask application that serves a small ML prediction API for disease identification together with a static frontend built with Tailwind.

Contents
- `backend/` — Flask app, model loader (`predictor.py`) and pickled models (`backend/models/`).
- `frontend/` — Tailwind-based static frontend in `frontend/public`.
- `requirements.txt` — Python dependencies for the backend.
- `Procfile` — gunicorn start command (for Heroku/Render).

Quick start (local)

1. Create a virtual environment and install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the Flask dev server:

```bash
python backend/app.py
```

Open http://127.0.0.1:5000 to view the frontend. API endpoints:
- `GET /models` — list available model names (from `backend/models/*.pkl`).
- `POST /predict/<model_name>` — send JSON body with features and receive a prediction.

Production deploy (Heroku / Render / similar)

- The repo includes a `Procfile` with a gunicorn start command:

```
web: gunicorn --chdir backend app:app --workers 3 --bind 0.0.0.0:$PORT
```

- Ensure `requirements.txt` is present at repository root (it is). Optionally add `runtime.txt` to pin Python version (e.g. `python-3.10.13`).
- If models are large or private, do NOT commit them to the repo. Instead store them in object storage and fetch them at startup using an environment variable (for example `MODEL_URLS`) or a small startup script.

AI agents used during development

- GitHub Copilot — used as a coding assistant to speed up iterative development, suggest code snippets and refactorings.
- presentation.ai — used to help create presentation assets and a short project pitch (notes, slide ideas and visuals).

Modeling notes

- Model type: XGBoost (trained using `xgboost` Python package).
- Hyperparameter tuning: performed with `optuna` for automated hyperparameter search and selection.

Datasets (cited)

This project used or references the following public datasets for training and benchmarking:

- Anemia Types Classification — Kaggle dataset (search "Anemia Types Classification" on Kaggle)
- Indian Liver Patient Records (ILPD) dataset
- Cardiovascular Disease dataset
- Diabetes prediction datasets (various publicly available diabetes datasets)

License & notes

- This repository contains pickled model files under `backend/models/`. If you did not create these models yourself, verify their license before redistributing. Large model files should be stored outside git and downloaded during deployment.

Contact / next steps

- To prepare a proper production deployment, I can help with:
  - Adding a `runtime.txt` and CI/CD manifest for Render or Heroku.
  - Adding a model download / caching step at startup.
  - Adding unit tests for the `Predictor` class and a basic integration test for the `/predict` endpoint.
# diseaseIdentifier
