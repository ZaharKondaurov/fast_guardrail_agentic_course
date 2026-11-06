"""Build a frequency-based blocklist, excluding common benign n-grams."""
import argparse
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .solve import RULES_PATH, ngrams
else:
    from solve import RULES_PATH, ngrams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("--output", type=Path, default=RULES_PATH)
    args = parser.parse_args()

    counts = Counter()
    benign_counts = Counter()
    with args.train.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            message_counts = Counter(ngrams(row["input"]))
            counts.update(message_counts)
            if row["label"] == 0:
                benign_counts.update(message_counts)

    excluded = set(sorted(benign_counts, key=lambda phrase: (-benign_counts[phrase], phrase))[:100])
    eligible = counts.keys() - excluded
    rules = sorted(eligible, key=lambda phrase: (-counts[phrase], phrase))[:100]
    args.output.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(rules)} rules")


if __name__ == "__main__":
    main()
