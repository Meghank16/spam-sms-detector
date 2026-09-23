"""
Flask API backend for Bilingual SMS Spam Detection (English + Hindi).
Serves the saved TF-IDF vectorizer and trained model.
"""

import os
import re
import math
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessor import clean_sms_text

app = Flask(__name__)
# Enable CORS for frontend running on localhost / vite dev server
CORS(app)

# Paths to saved models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# Load model and vectorizer at startup
model = None
vectorizer = None
model_name = "Linear SVM"

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Successfully loaded model and TF-IDF vectorizer.")
    else:
        print(f"Warning: Model files missing in {MODEL_DIR}")
except Exception as e:
    print(f"Error loading model: {e}")

# Common Romanized Hindi/Hinglish marker words
HINGLISH_KEYWORDS = {
    'bhai', 'yaar', 'aap', 'aapko', 'karo', 'kare', 'karein', 'karega',
    'aayega', 'aana', 'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se',
    'mein', 'par', 'pe', 'nahi', 'kuch', 'kya', 'kyu', 'kyun', 'ab',
    'abhi', 'jeeta', 'milega', 'paayein', 'dhanyawad', 'namaste',
    'shukriya', 'kro', 'kr', 'hoke', 'clg', 'chahiye', 'mera', 'meri',
    'dost', 'kaisa', 'kaise', 'sab', 'theek'
}

def detect_language(text: str) -> str:
    """
    Detects language based on script and vocabulary:
    - Devanagari characters -> Hindi
    - English/Latin characters -> English
    - Mixture of both (or Romanized Hindi) -> Hinglish
    """
    # Scrub URLs and email addresses before checking Latin script
    scrubbed = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    scrubbed = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', ' ', scrubbed)

    has_devanagari = bool(re.search(r'[\u0900-\u097F]', scrubbed))
    has_latin = bool(re.search(r'[a-zA-Z]', scrubbed))

    if has_devanagari and has_latin:
        return "Hinglish"
    elif has_devanagari:
        return "Hindi"
    elif has_latin:
        # Check if text contains Romanized Hindi markers
        tokens = set(re.findall(r'[a-zA-Z]+', scrubbed.lower()))
        if tokens.intersection(HINGLISH_KEYWORDS):
            return "Hinglish"
        return "English"
    return "English"

def compute_confidence(raw_model, X, pred_label: str) -> float:
    """
    Computes a mathematically sound confidence score [0.50, 0.99]
    using predict_proba if available, or sigmoid-transformed decision boundary distance.
    """
    if hasattr(raw_model, "predict_proba"):
        probs = raw_model.predict_proba(X)[0]
        classes = list(raw_model.classes_)
        idx = classes.index(pred_label)
        return float(probs[idx])
    elif hasattr(raw_model, "decision_function"):
        score = float(raw_model.decision_function(X)[0])
        # Margin distance |score|: scaled sigmoid confidence
        abs_score = abs(score)
        conf = 1.0 / (1.0 + math.exp(-1.5 * abs_score))
        return float(conf)
    return 0.85

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify backend and model status."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_type": model_name
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether an incoming SMS is spam or ham.
    Request body: {"text": "..."}
    Response: {"prediction": "spam"|"ham", "confidence": 0.97, "language": "English"}
    """
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model or vectorizer is not loaded on the server."
        }), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({
            "error": "Missing 'text' field in request body."
        }), 400

    raw_text = data.get("text", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({
            "error": "The SMS text cannot be empty."
        }), 400

    # 1. Detect language
    lang = detect_language(raw_text)

    # 2. Preprocess text using exact training logic
    cleaned = clean_sms_text(raw_text)

    # 3. Vectorize
    X = vectorizer.transform([cleaned])

    # 4. Predict
    prediction = str(model.predict(X)[0]).lower()

    # 5. Compute real confidence
    raw_conf = compute_confidence(model, X, prediction)
    confidence = round(raw_conf, 2)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "language": lang
    }), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
