"""Compute AR-12 metrics for a complete evaluation split."""
import argparse
import json
import math
import sys

MAX_FPR = 0.05
MAX_P99_MS = 5.0


def score_split(
    labels: list[int],
    answers: list[int],
    durations_ns: list[int],
    *,
    min_messages_per_second: float | None = None,
) -> dict:
    if not (len(labels) == len(answers) == len(durations_ns)) or set(labels) != {0, 1}:
        raise ValueError("Expected aligned labels, answers, and timings with both classes present")
    if any(type(answer) is not int or answer not in (0, 1) for answer in answers):
        raise ValueError("Expected integer answers 0 or 1")
    if any(duration <= 0 for duration in durations_ns):
        raise ValueError("Expected positive durations")
    if min_messages_per_second is not None and (
        not math.isfinite(min_messages_per_second) or min_messages_per_second <= 0
    ):
        raise ValueError("Minimum throughput must be finite and positive")

    tp = fp = tn = fn = 0
    for label, answer in zip(labels, answers, strict=True):
        blocked = answer == 1
        if label == 1:
            tp += blocked
            fn += not blocked
        else:
            fp += blocked
            tn += not blocked

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    p99_ms = sorted(durations_ns)[math.ceil(0.99 * len(durations_ns)) - 1] / 1e6
    throughput = len(durations_ns) * 1e9 / sum(durations_ns)
    gates = {
        "fpr": fpr <= MAX_FPR,
        "p99": p99_ms <= MAX_P99_MS,
        "throughput": (
            None if min_messages_per_second is None else throughput >= min_messages_per_second
        ),
    }
    return {
        "score": 0.0 if False in gates.values() else tpr,
        "messages": len(labels),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr, "fpr": fpr,
        "p99_ms": p99_ms,
        "messages_per_second": throughput,
        "gates": gates,
    }


def collect(payload: dict) -> dict:
    reference = payload["example"]["reference"]
    if reference not in ("0", "1"):
        raise ValueError("Expected reference to be a string 0 or 1")
    answer = payload["answer"]
    return {
        "statistics": {
            "label": int(reference),
            "prediction": int(answer) if answer in ("0", "1") else None,
        },
    }


def aggregate(results: list[dict], minimum_throughput: float) -> dict:
    durations = [row.get("resources", {}).get("solver", {}).get("inference_ns") for row in results]
    if any(type(duration) is not int or duration <= 0 for duration in durations):
        raise ValueError("Expected positive inference_ns from the operator benchmark")
    predictions = [row["statistics"]["prediction"] for row in results]
    invalid_answers = predictions.count(None)
    if invalid_answers:
        return {"score": 0.0, "statistics": {"invalid_answers": invalid_answers}}
    report = score_split(
        [row["statistics"]["label"] for row in results], predictions, durations,
        min_messages_per_second=minimum_throughput,
    )
    return {"score": report.pop("score"), "statistics": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect", action="store_true", help="Read one Arena example and answer")
    mode.add_argument("--aggregate", type=float, metavar="MIN_THROUGHPUT",
                      help="Aggregate Arena results with a minimum in messages per second")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        if args.collect:
            report = collect(payload)
        elif args.aggregate is not None:
            report = aggregate(payload["results"], args.aggregate)
        else:
            report = score_split(**payload)
    except (KeyError, TypeError, ValueError) as error:
        if args.collect or args.aggregate is not None:
            parser.exit(1, f"{error}\n")
        report = {"score": 0.0, "error": str(error)}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
