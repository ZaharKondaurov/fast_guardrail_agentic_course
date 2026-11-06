"""Evaluate a warmed-up classifier on labeled JSONL."""
import argparse
import json
import runpy
import sys
from pathlib import Path

from solve import RULES_PATH, load_classifier

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from case.ways.baseline.evaluate import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--rules", type=Path, default=RULES_PATH)
    model.add_argument("--solver", type=Path, help="Path to a solver with load_classifier()")
    parser.add_argument("--min-messages-per-second", type=float)
    args = parser.parse_args()

    if args.solver is None:
        predict = load_classifier(args.rules)
    else:
        predict = runpy.run_path(str(args.solver))["load_classifier"]()

    try:
        report = evaluate_dataset(args.dataset, predict, args.min_messages_per_second)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
