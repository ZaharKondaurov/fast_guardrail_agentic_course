"""Export AR-12 JSONL to the Arena evaluator format, preserving example IDs."""

import argparse
import json
from pathlib import Path


def export_arena(source: Path, output: Path) -> int:
    examples = []
    seen_ids = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                example_id = row["id"]
                prompt = row["input"]
                label = row["label"]
                if not isinstance(example_id, str) or not 1 <= len(example_id) <= 100:
                    raise ValueError("id must be a string of 1–100 characters")
                if example_id in seen_ids:
                    raise ValueError(f"duplicate id {example_id!r}")
                if not isinstance(prompt, str) or len(prompt) > 500_000:
                    raise ValueError("input must be a string of at most 500,000 characters")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("label must be the integer 0 or 1")
                examples.append(json.dumps(
                    {"id": example_id, "input": prompt, "reference": str(label)},
                    ensure_ascii=False,
                ))
                seen_ids.add(example_id)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error

    with output.open("x", encoding="utf-8") as stream:
        for example in examples:
            stream.write(example + "\n")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original AR-12 JSONL split")
    parser.add_argument("output", type=Path, help="New Arena JSONL file")
    args = parser.parse_args()
    try:
        count = export_arena(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.output}: {count} examples")


if __name__ == "__main__":
    main()
