"""Train-only holdout and cutoff shared by the learned baselines."""

import json

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42


def split_train(path):
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    texts = [row["input"] for row in rows]
    labels = [row["label"] for row in rows]
    return train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED,
    )


def calibrate(scores, labels):
    """Pick a strict score cutoff with at most 5% false positives on holdout."""
    benign = np.sort(np.asarray(scores)[np.asarray(labels) == 0])
    allowed = len(benign) // 20
    return float(benign[-allowed - 1])
