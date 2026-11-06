import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[1]


def run_command(directory, command, payload):
    process = subprocess.run(
        [sys.executable, *command["argv"][1:]],
        cwd=directory, input=json.dumps(payload),
        capture_output=True, text=True, check=True,
    )
    return json.loads(process.stdout)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tpr", 0.5),
        ("fpr", 0.0),
        ("p99", 3.0),
        ("throughput", 500.0),
    ],
)
def test_package_contract(name, expected):
    directory = PACKAGES / name
    config = json.loads((directory / "metric.json").read_text())
    results = []
    for reference, answer, milliseconds in (
        ("1", "1", 1), ("1", "0", 2), ("0", "0", 3),
    ):
        example = {"id": str(milliseconds), "input": "text", "reference": reference}
        result = run_command(directory, config["command"], {"example": example, "answer": answer})
        results.append({
            "example_id": example["id"], "answer": answer, **result,
            "resources": {"solver": {
                "inference_ns": milliseconds * 10**6,
            }},
        })
    assert run_command(directory, config["aggregation"], {"results": results})["score"] == expected


@pytest.mark.parametrize(("false_positives", "expected_score"), [(5, 1.0), (6, 0.0)])
def test_tpr_package_applies_fpr_gate(false_positives, expected_score):
    directory = PACKAGES / "tpr"
    config = json.loads((directory / "metric.json").read_text())
    results = [{"statistics": {"harmful": True, "blocked": True}}]
    results.extend(
        {"statistics": {"harmful": False, "blocked": index < false_positives}}
        for index in range(100)
    )
    report = run_command(directory, config["aggregation"], {"results": results})
    assert report == {
        "score": expected_score,
        "statistics": {"tpr": 1.0, "fpr": false_positives / 100},
    }


def test_p99_package_rejects_runner_wall_time():
    directory = PACKAGES / "p99"
    config = json.loads((directory / "metric.json").read_text())
    results = [{"resources": {"solver": {"wall_seconds": 0.003}}}]
    with pytest.raises(subprocess.CalledProcessError):
        run_command(directory, config["aggregation"], {"results": results})


@pytest.mark.parametrize(
    ("answers", "durations", "expected_score", "failed_gate"),
    [
        (("1", "0"), (1_000_000, 1_000_000), 1.0, None),
        (("1", "1"), (1_000_000, 1_000_000), 0.0, "fpr"),
        (("1", "0"), (6_000_000, 1_000_000), 0.0, "p99"),
        (("1", "0"), (2_000_000, 2_000_000), 0.0, "throughput"),
        (("invalid", "0"), (1_000_000, 1_000_000), 0.0, None),
    ],
)
def test_combined_package(answers, durations, expected_score, failed_gate):
    config = json.loads((PACKAGES / "metric.json").read_text())
    assert config["direction"] == "maximize"
    results = []
    for reference, answer, duration in zip(("1", "0"), answers, durations, strict=True):
        example = {"id": reference, "input": "text", "reference": reference}
        result = run_command(PACKAGES, config["command"], {"example": example, "answer": answer})
        results.append({**result, "resources": {"solver": {"inference_ns": duration}}})
    report = run_command(PACKAGES, config["aggregation"], {"results": results})
    assert report["score"] == expected_score
    if failed_gate:
        assert report["statistics"]["gates"][failed_gate] is False
    elif "invalid" in answers:
        assert report["statistics"] == {"invalid_answers": 1}
    else:
        assert all(report["statistics"]["gates"].values())


def test_combined_package_requires_benchmark_timing():
    config = json.loads((PACKAGES / "metric.json").read_text())
    results = [
        {"statistics": {"label": label, "prediction": label},
         "resources": {"solver": {"wall_seconds": 0.001}}}
        for label in (0, 1)
    ]
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero"):
        run_command(PACKAGES, config["aggregation"], {"results": results})
