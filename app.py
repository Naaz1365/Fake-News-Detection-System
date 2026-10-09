
import re
import logging

import joblib
import numpy as np
import textstat

from flask import Flask, request, jsonify, render_template
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob


# --------------------------------------------------
# Custom transformers required to load saved pipeline
# --------------------------------------------------

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = []

        for text in X:
            words = text.split()
            word_vectors = [
                self.model.wv[word]
                for word in words
                if word in self.model.wv
            ]

            if word_vectors:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(
                    np.zeros(self.model.vector_size)
                )

        return csr_matrix(np.asarray(vectors, dtype=float))


def auxiliary_features(text):
    words = text.split()
    readability = textstat.flesch_reading_ease(text)
    polarity = TextBlob(text).sentiment.polarity
    word_count = len(words)

    punctuation_count = sum(
        char in ".,!?;:" for char in text
    )

    punctuation_ratio = (
        punctuation_count / len(text)
        if text else 0.0
    )

    return [
        readability,
        polarity,
        word_count,
        punctuation_ratio
    ]


class AuxiliaryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(
            np.asarray(
                [auxiliary_features(text) for text in X],
                dtype=float
            )
        )


# --------------------------------------------------
# Flask configuration and saved artifacts
# --------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


logging.basicConfig(level=logging.INFO)
import __main__

__main__.Word2VecTransformer = Word2VecTransformer
__main__.AuxiliaryTransformer = AuxiliaryTransformer
__main__.auxiliary_features = auxiliary_features

try:
    model = joblib.load("models/best_model.joblib")
    feature_union = joblib.load("models/feature_union.joblib")
except Exception:
    app.logger.exception("Could not load model artifacts")
    raise


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    # Validate request format
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json."
        }), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    text = data.get("text")

    if not isinstance(text, str):
        return jsonify({
            "error": "'text' must be a string."
        }), 400

    if not text.strip():
        return jsonify({
            "error": "Text cannot be empty."
        }), 400

    if len(text) > 20000:
        return jsonify({
            "error": "Text is too long. Maximum length is 20000 characters."
        }), 400

    cleaned = clean_text(text)

    if not cleaned:
        return jsonify({
            "error": "No usable English text found after preprocessing."
        }), 400

    try:
        features = feature_union.transform([cleaned])
        prediction = int(model.predict(features)[0])

        label = "FAKE" if prediction == 1 else "REAL"

        # This score indicates the magnitude of the decision,
        # NOT the probability that the news is true.
        score = None

        if hasattr(model, "decision_function"):
            decision_score = float(
                np.asarray(model.decision_function(features)).ravel()[0]
            )

            # Numerically stable sigmoid of the absolute score.
            magnitude = min(abs(decision_score), 500.0)
            score = round(
                float(1.0 / (1.0 + np.exp(-magnitude)) * 100),
                2
            )

        return jsonify({
            "label": label,
            "prediction": label,
            "credibility_score": None,
            "confidence_like_score": score,
            "score_note": (
                "This is a decision-derived score, not a calibrated "
                "probability or verified credibility score."
            ),
            "top_features": [],
            "explanation": (
                "Prediction generated using TF-IDF, Word2Vec and "
                "auxiliary text features. This output is not an "
                "independent fact-check."
            )
        }), 200

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed due to an internal processing error."
        }), 500


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Request is too large. Maximum request size is 1 MB."
    }), 413


if __name__ == "__main__":
    # Debug mode is disabled for safer local operation.
    app.run(debug=False)