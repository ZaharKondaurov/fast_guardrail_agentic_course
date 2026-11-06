"""Fit TF-IDF and CatBoost on the AR-12 training split."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from ..calibration import SEED, calibrate, split_train
from .solve import MODEL_PATH, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    args = parser.parse_args()
    texts, heldout, labels, heldout_labels = split_train(args.train)

    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8_000, dtype=np.float32)
    chars = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 4), min_df=2, max_features=8_000, dtype=np.float32,
    )
    words.fit(texts)
    chars.fit(texts)
    model = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1, border_count=32, rsm=0.1,
        random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False,
    )
    model.fit(features(texts, words, chars), labels)
    probabilities = model.predict_proba(features(heldout, words, chars), thread_count=1)[:, 1]
    probability_threshold = calibrate(probabilities, heldout_labels)
    joblib.dump((words, chars, model, probability_threshold), MODEL_PATH, compress=3)
    print(f"{MODEL_PATH}: probability threshold {probability_threshold:.4f}")


if __name__ == "__main__":
    main()
