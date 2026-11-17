"""Find train/evaluation duplicates after normalizing whitespace."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

if __package__:
    from .deduplicate import normalize_input
else:
    from deduplicate import normalize_input


def read_examples(path: Path):
    """Read IDs and original texts from raw AR-12 or Arena JSONL."""
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                example_id, content = row["id"], row["input"]
                if not isinstance(example_id, str) or not isinstance(content, str):
                    raise ValueError("id and input must be strings")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            yield {"id": example_id, "input": content}


def find_duplicates(train: Path, evaluation: Path) -> list[dict]:
    """Match evaluation examples against train, retaining every source match."""
    by_text = defaultdict(list)
    for example in read_examples(train):
        by_text[normalize_input(example["input"])].append(example)

    # Compare normalized text, but save the original text from both splits.
    duplicates = []
    for example in read_examples(evaluation):
        matches = by_text.get(normalize_input(example["input"]))
        if matches:
            duplicates.append({"evaluation": example, "train_matches": matches})
    return duplicates


def main() -> None:
    """Write one JSONL entry per duplicated evaluation example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        duplicates = find_duplicates(args.train, args.evaluation)
        with args.output.open("x", encoding="utf-8") as destination:
            for duplicate in duplicates:
                destination.write(json.dumps(duplicate, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.output}: {len(duplicates)} evaluation examples with train duplicates")


if __name__ == "__main__":
    main()
