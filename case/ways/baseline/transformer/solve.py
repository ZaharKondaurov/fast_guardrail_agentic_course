"""Classify a message with the quantized CPU transformer."""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).with_name("model.onnx")
CONFIG_PATH = Path(__file__).with_name("model.json")
SEQUENCE_LENGTH = 128


def encode(texts):
    tokens = np.zeros((len(texts), SEQUENCE_LENGTH), dtype=np.int64)
    tokens[:, 0] = 257
    for index, text in enumerate(texts):
        # Limit the number of characters encoded before truncating to bytes.
        prefix = text[:SEQUENCE_LENGTH - 1].encode("utf-8")[:SEQUENCE_LENGTH - 1]
        tokens[index, 1:len(prefix) + 1] = np.frombuffer(prefix, dtype=np.uint8).astype(np.int64) + 1
    return tokens


def load_session(path=MODEL_PATH):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def load_classifier():
    session = load_session()
    cutoff = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["cutoff"]

    def classify(text):
        scores, = session.run(None, {"tokens": encode([text])})
        return int(scores[0] > cutoff)

    return classify


if __name__ == "__main__":
    task = json.load(sys.stdin)
    print(json.dumps({"answer": str(load_classifier()(task["input"]))}))
