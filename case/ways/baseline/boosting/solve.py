"""Classify a message with TF-IDF and CatBoost."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

MODEL_PATH = Path(__file__).with_name("catboost-proba.joblib")


def features(texts, word_vectorizer, char_vectorizer):
    lengths = np.array([[len(text), len(text.split())] for text in texts], dtype=np.float32)
    return hstack(
        [word_vectorizer.transform(texts), char_vectorizer.transform(texts), lengths],
        format="csr",
    )


def load_classifier():
    word_vectorizer, char_vectorizer, model, probability_threshold = joblib.load(MODEL_PATH)

    def classify(text):
        inputs = features([text], word_vectorizer, char_vectorizer)
        _, harmful_probability = model.predict_proba(inputs, thread_count=1)[0]
        return int(harmful_probability > probability_threshold)

    return classify


if __name__ == "__main__":
    task = json.load(sys.stdin)
    print(json.dumps({"answer": str(load_classifier()(task["input"]))}))
