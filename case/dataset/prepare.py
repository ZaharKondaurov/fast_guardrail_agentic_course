"""Download the Hugging Face sources and export the dataset described in README.md."""

import argparse
import json
import random
from pathlib import Path

if __package__:
    from .deduplicate import deduplicate_splits, validate_disjoint
else:
    from deduplicate import deduplicate_splits, validate_disjoint


SPLITS = ("train", "public", "private")
PRIORITY = ("private", "public", "train")


def injection_records(rows, split):
    """Преобразует записи набора инъекций в общий формат датасета AR-12."""
    return [
        {"id": f"neuralchemy-{split}-{index}", "input": row["text"],
         "label": int(row["label"]), "class": row["category"]}
        for index, row in enumerate(rows)
    ]


def toxic_records(rows, split):
    """Добавляет токсичные примеры с единой меткой и идентификаторами источника."""
    return [
        {"id": f"toxicchat-{split}-{index}", "input": row["user_input"],
         "label": 1, "class": "toxic"}
        for index, row in enumerate(rows)
        if row["toxicity"] == 1
    ]


def download_records(args):
    """Load the two source datasets and build the initial split assignments."""
    from datasets import load_dataset

    injections = load_dataset(
        "neuralchemy/Prompt-injection-dataset", "full", revision=args.injection_revision
    )
    toxic_chat = load_dataset(
        "lmsys/toxic-chat", "toxicchat0124", revision=args.toxicchat_revision
    )
    toxic_test = toxic_records(toxic_chat["test"], "test")
    random.Random(args.seed).shuffle(toxic_test)
    midpoint = len(toxic_test) // 2
    additions = {
        "train": toxic_records(toxic_chat["train"], "train"),
        "public": toxic_test[:midpoint],
        "private": toxic_test[midpoint:],
    }
    split_map = {"train": "train", "public": "validation", "private": "test"}
    return {
        split: injection_records(injections[source_split], source_split) + additions[split]
        for split, source_split in split_map.items()
    }


def main():
    """Загружает закреплённые наборы, формирует разделы и пишет их в отдельные каталоги."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir-train", type=Path, required=True)
    parser.add_argument("--output-dir-test", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection-revision", default="main")
    parser.add_argument("--toxicchat-revision", default="main")
    parser.add_argument("--from-splits", type=Path, nargs=3, metavar=("TRAIN", "PUBLIC", "PRIVATE"))
    args = parser.parse_args()


    if args.from_splits:
        records = {}
        for split, path in zip(SPLITS, args.from_splits, strict=True):
            with path.open(encoding="utf-8") as source:
                records[split] = [json.loads(line) for line in source if line.strip()]
    else:
        records = download_records(args)

    cleaned = deduplicate_splits(records, PRIORITY)
    validate_disjoint(cleaned)
    paths = {
        split: (args.output_dir_train if split == "train" else args.output_dir_test)
        / f"ar12.{split}.jsonl"
        for split in SPLITS
    }
    for path in paths.values():
        if path.exists():
            parser.error(f"Output already exists: {path}")

    for split, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = cleaned[split]
        with path.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        toxic_count = sum(row["id"].startswith("toxicchat-") for row in rows)
        print(f"{path}: {len(rows)} rows ({toxic_count} from ToxicChat), "
              f"{len(records[split]) - len(rows)} duplicates removed")


if __name__ == "__main__":
    main()
