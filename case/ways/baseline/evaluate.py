"""Evaluate an AR-12 baseline on labeled JSONL."""

import argparse
import json
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns

from ...metric.score import score_split


def evaluate_dataset(dataset, predict, min_messages_per_second):
    with dataset.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    for row in rows[:100]:
        predict(row["input"])

    answers = []
    timings = []
    for row in rows:
        start = perf_counter_ns()
        answer = predict(row["input"])
        timings.append(perf_counter_ns() - start)
        answers.append(answer)

    return score_split(
        [row["label"] for row in rows], answers, timings,
        min_messages_per_second=min_messages_per_second,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("matching", "boosting", "transformer"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-messages-per-second", type=float)
    args = parser.parse_args()
    solver = import_module(f".{args.candidate}.solve", package=__package__)
    try:
        report = evaluate_dataset(
            args.dataset, solver.load_classifier(), args.min_messages_per_second,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
