Backend Flask server
=====================

Place your pickled model files into `backend/models/` as `<name>.pkl` (for example `blood.pkl`, `diabetes.pkl`).

Quick start:

1. Create a virtualenv and install dependencies:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt

2. Run the server:

   python backend/app.py

Endpoints:


If your XGBoost model file is named `best_xgb_model.pkl`, you can call it via the endpoint `POST /predict/best_xgb_model`.

Notes:
- The predictor tries to call `model.predict([sample])` when the pickled object exposes a `predict` method (scikit-learn-style). If your model uses a different interface, update `backend/predictor.py` accordingly.
