"""Create a training split and an Arena evaluation split from a list of IDs."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from export_arena import export_arena


def prepare_split(source: Path, manifest: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = manifest.read_text(encoding="utf-8").splitlines()
    heldout = set(ids)
    source_ids = {row["id"] for row in rows}
    if not ids or "" in heldout or len(ids) != len(heldout) or len(source_ids) != len(rows):
        raise ValueError("Expected unique source and evaluation IDs")
    if not heldout < source_ids:
        raise ValueError("Evaluation IDs must be a proper subset of source IDs")
    training = [row for row in rows if row["id"] not in heldout]
    evaluation = [row for row in rows if row["id"] in heldout]
    if any({row["label"] for row in split} != {0, 1} for split in (training, evaluation)):
        raise ValueError("Training and evaluation splits must both contain both classes")
    with TemporaryDirectory(prefix="split-", dir=output.parent) as directory:
        staged = Path(directory) / "data"
        staged.mkdir()
        for filename, records in (("train.jsonl", training), ("evaluation.jsonl", evaluation)):
            with (staged / filename).open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        export_arena(staged / "evaluation.jsonl", staged / "evaluation.arena.jsonl")
        staged.rename(output)
    print(f"{output}: {len(training)} training, {len(evaluation)} evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("evaluation_ids", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        prepare_split(args.source, args.evaluation_ids, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
