import json
import sys


def collect(payload):
    if payload["answer"] not in ("0", "1"):
        raise ValueError("Expected answer to be a string 0 or 1")
    return {"statistics": {}}


def aggregate(results):
    if not results:
        raise ValueError("Expected a nonempty split")
    rows = [row["resources"]["solver"] for row in results]
    durations = [row["inference_ns"] for row in rows]
    if any(type(value) is not int or value <= 0 for value in durations):
        raise ValueError("Expected positive inference_ns from the benchmark")
    return {"score": len(durations) * 1e9 / sum(durations)}


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    report = aggregate(payload["results"]) if "--aggregate" in sys.argv else collect(payload)
    print(json.dumps(report))
