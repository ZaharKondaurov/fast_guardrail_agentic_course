"""Normalize and check AR-12 dataset splits."""

from collections.abc import Iterable, Mapping


def normalize_input(text: str) -> str:
    """Collapse whitespace while preserving case and the original stored text."""
    return " ".join(text.split())


def deduplicate_splits(
    splits: Mapping[str, Iterable[dict]], priority: tuple[str, ...],
) -> dict[str, list[dict]]:
    """Keep the first normalized occurrence, by split priority then source order."""
    seen = set()
    result = {}
    for split in priority:
        result[split] = []
        for row in splits[split]:
            key = normalize_input(row["input"])
            if key not in seen:
                seen.add(key)
                result[split].append(row)
    return result


def validate_disjoint(splits: Mapping[str, Iterable[dict]]) -> None:
    """Reject repeated IDs or normalized messages within or across splits."""
    seen_ids = {}
    seen_texts = {}
    for split, rows in splits.items():
        for row in rows:
            identity = (split, row["id"])
            key = normalize_input(row["input"])
            for value, seen, kind in (
                (row["id"], seen_ids, "id"), (key, seen_texts, "normalized input"),
            ):
                if value in seen:
                    raise ValueError(f"Duplicate {kind}: {seen[value]} and {identity}")
                seen[value] = identity
