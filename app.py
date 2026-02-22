"""
Flask API for the Insurance Purchase Predictor.
Serves the web UI and exposes REST endpoints for predictions.
"""

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import the ML module
from Project_1_code import train, predict_insurance, get_model_info, MODEL_PATH

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Serve the main web UI."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Predict insurance purchase.
    Body: { "age": int, "salary": float }
    Returns: { "label": str, "probability": float, "confidence_level": str }
    """
    try:
        data = request.get_json(force=True)
        age = int(data.get('age', 0))
        salary = float(data.get('salary', 0))

        if not (18 <= age <= 65):
            return jsonify({'error': 'Age must be between 18 and 65'}), 400
        if salary < 0 or salary > 50_00_000:
            return jsonify({'error': 'Salary must be between 0 and 50,00,000'}), 400

        if not MODEL_PATH.exists():
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 503

        label, prob = predict_insurance(age, salary)

        # Determine confidence level
        if prob > 0.8 or prob < 0.2:
            confidence = 'High'
        elif prob > 0.6 or prob < 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        return jsonify({
            'label': label,
            'probability': round(prob, 4),
            'percentage': round(prob * 100, 1),
            'confidence_level': confidence,
            'age': age,
            'salary': salary,
        })

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    """Retrain the model and return the leaderboard."""
    try:
        results = train(silent=True)
        info = get_model_info()
        return jsonify({
            'message': 'Model trained successfully',
            'leaderboard': results,
            'model_info': info,
        })
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Return metadata about the current model."""
    info = get_model_info()
    if info is None:
        return jsonify({'error': 'No model trained yet'}), 404
    return jsonify(info)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
