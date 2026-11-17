"""Тесты метрики. Обязательно — «жульнические» ответы, которые метрика должна ловить."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCORE = Path(__file__).parents[1] / "score.py"


def run(example: dict, answer: str) -> float:
    """Запускает метрику как отдельный процесс для проверки её stdin/stdout-контракта."""
    out = subprocess.run([sys.executable, str(SCORE)], input=json.dumps({"example": example, "answer": answer}),
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["score"]


def test_wrong_format_gets_zero():
    """Проверяет штраф за ответ в неверном формате."""
    assert run({"id": "x", "input": {}}, "not a valid answer") == 0.0


@pytest.mark.parametrize(
    ("override", "failed_gate"),
    [
        ({}, None),
        ({"answers": [1, 0] + [1] * 6 + [0] * 94}, "fpr"),
        ({"durations_ns": [6_000_000] * 102}, "p99"),
        ({"min_messages_per_second": 1001}, "throughput"),
    ],
)
def test_split_score(override, failed_gate):
    """Проверяет общий балл и диагностику при нарушении каждого гейта."""
    payload = {
        "labels": [1, 1] + [0] * 100,
        "answers": [1, 0] + [1] * 5 + [0] * 95,
        "durations_ns": [1_000_000] * 102,
    }
    payload.update(override)
    out = subprocess.run(
        [sys.executable, str(SCORE)], input=json.dumps(payload),
        capture_output=True, text=True, check=True,
    ).stdout
    report = json.loads(out)
    if failed_gate:
        assert report["gates"][failed_gate] is False
    else:
        assert report["tpr"] == 0.5
        assert report["fpr"] == 0.05
        assert report["p99_ms"] == 1.0
        assert report["messages_per_second"] == 1000
        assert report["gates"]["throughput"] is None
    assert report["recall"] == report["tpr"] == 0.5
    fp = 6 if failed_gate == "fpr" else 5
    assert report["precision"] == 1 / (1 + fp)
    assert report["score"] == report["f1"] == 2 / (3 + fp)
