"""Download the Hugging Face sources and export the dataset described in README.md."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def injection_records(rows, split):
    return [
        {"id": f"neuralchemy-{split}-{index}", "input": row["text"],
         "label": int(row["label"]), "class": row["category"]}
        for index, row in enumerate(rows)
    ]


def toxic_records(rows, split):
    return [
        {"id": f"toxicchat-{split}-{index}", "input": row["user_input"],
         "label": 1, "class": "toxic"}
        for index, row in enumerate(rows)
        if row["toxicity"] == 1
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir-train", type=Path, required=True)
    parser.add_argument("--output-dir-test", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection-revision", default="main")
    parser.add_argument("--toxicchat-revision", default="main")
    args = parser.parse_args()

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
    records = {
        split: injection_records(injections[source_split], source_split) + additions[split]
        for split, source_split in split_map.items()
    }
    seen = set()
    args.output_dir_train.mkdir(parents=True, exist_ok=True)
    args.output_dir_test.mkdir(parents=True, exist_ok=True)
    # Keep evaluation examples when their text also appears in training.
    for split in ("private", "public", "train"):
        rows = records[split]
        if split == "train":
            path = args.output_dir_train / f"ar12.{split}.jsonl"
        else:
            path = args.output_dir_test / f"ar12.{split}.jsonl"
        kept = 0
        toxic_count = 0
        with path.open("w", encoding="utf-8") as output:
            for row in rows:
                prompt = row["input"]
                if prompt in seen:
                    continue
                seen.add(prompt)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                if row["id"].startswith("toxicchat-"):
                    toxic_count += 1
        print(f"{path}: {kept} rows ({toxic_count} from ToxicChat), "
              f"{len(rows) - kept} duplicates removed")


if __name__ == "__main__":
    main()
