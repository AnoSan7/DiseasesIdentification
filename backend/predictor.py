import os
import pickle
from typing import Any, Dict

# optional import for xgboost handling
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None
try:
    import numpy as np  # type: ignore
except Exception:
    np = None


class Predictor:
    """Simple predictor that loads pickle files from backend/models.

    Expected file names: <model_name>.pkl (e.g. blood.pkl, diabetes.pkl)
    The loaded object should ideally expose a .predict(X) method (scikit-learn style).
    """

    def __init__(self, models_dir: str | None = None):
        here = os.path.dirname(__file__)
        self.models_dir = models_dir or os.path.join(here, 'models')
        self._cache: Dict[str, Any] = {}

    def model_path(self, name: str) -> str:
        fn = f"{name}.pkl" if not name.endswith('.pkl') else name
        return os.path.join(self.models_dir, fn)

    def load(self, name: str):
        if name in self._cache:
            return self._cache[name]
        path = self.model_path(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            self._cache[name] = obj
            return obj
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{name}': {e}")

    def predict(self, name: str, payload: dict):
        """Predict using model `name` with a payload dict.

        The payload shape expected depends on the pickled model. As a convenience,
        if payload is a dict we will attempt to convert it to a flat list of values
        in insertion order and call model.predict([values]).
        """
        model = self.load(name)
        if model is None:
            raise FileNotFoundError(f"Model '{name}' not found in {self.models_dir}")

        def _to_native(o):
            """Recursively convert numpy types/arrays to native Python types for JSON serialization."""
            try:
                if np is not None:
                    # numpy scalar
                    if isinstance(o, np.generic):
                        return o.item()
                    # numpy array
                    if isinstance(o, np.ndarray):
                        return o.tolist()
            except Exception:
                pass
            # lists/tuples
            if isinstance(o, (list, tuple)):
                return [_to_native(x) for x in o]
            return o

        def _coerce_sample(payload, expected_order: list | None = None):
            """Turn payload (dict or list) into a list of numeric values.

            This tries to convert string form values to float. If conversion
            fails for any feature, raise a ValueError with the feature name.
            """
            sample = []
            # If expected order provided and payload is dict, build values in that order
            if expected_order and isinstance(payload, dict):
                # build normalized lookup mapping to handle variations like spaces/underscores/casing
                def _norm(s: str) -> str:
                    return ''.join(ch.lower() for ch in s if ch.isalnum())

                lookup = { _norm(k): k for k in payload.keys() }
                ordered_keys = []
                for exp in expected_order:
                    if exp is None:
                        continue
                    key = lookup.get(_norm(exp))
                    if key is None:
                        raise KeyError(f"Missing feature required by model: '{exp}'")
                    ordered_keys.append(key)
                iter_items = [(k, payload[k]) for k in ordered_keys]
            elif isinstance(payload, dict):
                iter_items = list(payload.items())
            else:
                iter_items = None

            if iter_items is not None:
                for k, v in iter_items:
                    # treat empty strings as NaN
                    if v is None or (isinstance(v, str) and v.strip() == ''):
                        sample.append(float('nan'))
                        continue
                    # numpy scalar
                    try:
                        if np is not None and isinstance(v, np.generic):
                            sample.append(float(v))
                            continue
                    except Exception:
                        pass

                    # numeric types
                    if isinstance(v, (int, float)):
                        sample.append(float(v))
                        continue

                    # strings: try to parse to float (allow commas)
                    if isinstance(v, str):
                        try:
                            sample.append(float(v.replace(',', '')))
                            continue
                        except Exception:
                            raise ValueError(f"Could not convert feature '{k}' value {v!r} to float")

                    # other types: try generic float conversion
                    try:
                        sample.append(float(v))
                    except Exception:
                        raise ValueError(f"Could not convert feature '{k}' value {v!r} to float")

            elif isinstance(payload, list):
                # assume list-like of numbers
                for i, v in enumerate(payload):
                    try:
                        sample.append(float(v))
                    except Exception:
                        raise ValueError(f"Could not convert payload index {i} value {v!r} to float")
            else:
                try:
                    sample = [float(payload)]
                except Exception:
                    raise ValueError(f"Could not convert payload value {payload!r} to float")

            return sample

        # If model exposes feature names (scikit-learn style), use them to order features
        expected_order = None
        try:
            if hasattr(model, 'feature_names_in_'):
                expected_order = list(getattr(model, 'feature_names_in_'))
        except Exception:
            expected_order = None

        # Build numeric sample once (will be reused for predict and predict_proba)
        sample = _coerce_sample(payload, expected_order=expected_order)

        # XGBoost native Booster handling
        if xgb is not None and hasattr(xgb, 'Booster') and isinstance(model, xgb.Booster):
            # ensure numpy is available
            if np is None:
                raise RuntimeError('numpy is required to use xgboost Booster models')
            # build numeric sample
            dmat = xgb.DMatrix(np.array([sample]))
            try:
                res = model.predict(dmat)
            except Exception as e:
                raise RuntimeError(f"Model prediction failed: {e}")
            # convert to native types
            if hasattr(res, '__iter__'):
                try:
                    if len(res) == 1:
                        return _to_native(res[0])
                except Exception:
                    pass
                return _to_native(res)
            return _to_native(res)

        # primitive handling: if model has predict, try to call it (sklearn/xgboost.sklearn wrappers)
        if hasattr(model, 'predict'):
            try:
                res = model.predict([sample])
            except Exception as e:
                # final fallback: the model might be a callable expecting the dict
                try:
                    return model(payload)
                except Exception:
                    raise RuntimeError(f"Model prediction failed: {e}")
            # normalize output to native types
            # normalize output to native types
            if hasattr(res, '__iter__'):
                try:
                    if len(res) == 1:
                        result_value = _to_native(res[0])
                    else:
                        result_value = _to_native(res)
                except Exception:
                    result_value = _to_native(res)
            else:
                result_value = _to_native(res)

            # also try to compute probabilities when available (sklearn-style)
            probabilities = None
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([sample])
                    # take first row
                    if hasattr(proba, '__iter__'):
                        probabilities = _to_native(proba[0])
                    else:
                        probabilities = _to_native(proba)
            except Exception:
                probabilities = None

            if probabilities is not None:
                return {'result': result_value, 'probabilities': probabilities}
            return result_value

        # If model itself is callable
        if callable(model):
            try:
                return model(payload)
            except Exception as e:
                raise RuntimeError(f"Model callable prediction failed: {e}")

        # Can't handle this model type
        raise TypeError(f"Loaded model for '{name}' is not callable or has no predict()")
