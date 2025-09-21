from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
from predictor import Predictor
from flask import send_from_directory

app = Flask(__name__)
CORS(app)
predictor = Predictor(models_dir=str(Path(__file__).parent / 'models'))
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend' / 'public'


@app.get('/models')
def list_models():
    p = Path(predictor.models_dir)
    if not p.exists():
        return jsonify([])
    files = [f.stem for f in p.glob('*.pkl')]
    return jsonify(files)


@app.post('/predict/<model_name>')
def predict(model_name: str):
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'Expected JSON body'}), 400
    try:
        result = predictor.predict(model_name, data)
        # predictor may return a dict with 'result' and 'probabilities' or a primitive
        if isinstance(result, dict) and 'result' in result:
            return jsonify(result)
        return jsonify({'result': result})
    except FileNotFoundError:
        return jsonify({'error': f"Model '{model_name}' not found"}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve frontend static files (index.html + assets)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path: str):
    try:
        # full path to requested file in frontend/public
        if path and (FRONTEND_DIR / path).exists():
            return send_from_directory(str(FRONTEND_DIR), path)
        # fallback to index.html (SPA behaviour)
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    except Exception:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
