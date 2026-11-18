"""Mirror live campaign status to MLflow without sending dataset contents."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from evaluator.search import Journal


def active_turns(directory: Path) -> list[dict]:
    """Read paired start/end events for the latest active runs of each role."""
    latest = {}
    for path in sorted(directory.glob("*.usage.jsonl"), key=lambda entry: entry.name):
        role = path.name.split("-", 1)[0]
        latest[role] = path
    active = []
    for path in latest.values():
        run = path.with_name(path.name.removesuffix(".usage.jsonl") + ".json")
        try:
            if json.loads(run.read_text())["state"] != "running":
                continue
        except (OSError, ValueError):
            continue
        turns = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                # The writer may be appending a line while this snapshot is read.
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                turn = row.get("turn_id")
                if row["event"] == "turn_start" and turn:
                    turns[turn] = row
                elif row["event"] == "turn" and turn:
                    turns.pop(turn, None)
        active.extend(turns.values())
    return active


def private_state(run: Path, finalization: dict | None) -> str:
    """Read only the status of the matching sealed batch in operator storage."""
    pointer = run / "closed-dir"
    if not finalization or not pointer.is_file():
        return "waiting"
    closed = Path(pointer.read_text().strip())
    for path in (closed / "private").glob("*/record.json"):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if result.get("batch_digest") == finalization["batch_digest"]:
            return result["status"]
    return "waiting"


def snapshot(run: Path) -> dict:
    """Derive the phase, active roles and measurement counters from current state."""
    journal = Journal(run / "search.sqlite3").export()
    campaign = journal["campaign"]
    private = private_state(run, campaign["finalization"])
    directory = run / "agent-workspace/campaign/board/agent-runs"
    turns = active_turns(directory)
    records = sorted(
        path for path in directory.glob("orchestrator-*.json")
        if not path.name.endswith(".usage.jsonl")
    )
    orchestrator = "not_started"
    if records:
        try:
            orchestrator = json.loads(records[-1].read_text())["state"]
        except (OSError, ValueError):
            orchestrator = "unknown"
    roles = Counter(row["role"] for row in turns)
    measurements = Counter(
        row["comparison"]["split"] for row in journal["measurements"] if row["state"] == "running"
    )
    if private == "complete":
        phase = "complete"
    elif private == "failed" or orchestrator == "failed":
        phase = "failed"
    elif private == "running":
        phase = "private_evaluation"
    elif campaign["finalization"]:
        phase = "awaiting_private"
    elif roles["reviewer"]:
        phase = "review"
    elif roles["researcher"]:
        phase = "research"
    elif measurements["public"]:
        phase = "public_evaluation"
    elif measurements["train"]:
        phase = "train_evaluation"
    elif orchestrator == "not_started":
        phase = "ready"
    elif orchestrator == "complete":
        phase = "complete"
    elif not campaign["remaining_proposals"] or campaign["stopped"]:
        phase = "selecting_finalists"
    else:
        phase = "orchestrating"
    return {
        "phase": phase, "private": private, "orchestrator": orchestrator,
        "active_roles": {key: value for key, value in roles.items() if value},
        "active_tasks": [
            {"role": row["role"], "task": row.get("task")}
            for row in turns if row["role"] in {"researcher", "reviewer"}
        ],
        "active_measurements": campaign["running"],
        "counters": {
            "proposed": campaign["proposed"], "remaining_proposals": campaign["remaining_proposals"],
            "completed_measurements": campaign["completed_measurements"],
            "active_train": measurements["train"], "active_public": measurements["public"],
            "active_orchestrators": roles["orchestrator"],
            "active_researchers": roles["researcher"], "active_reviewers": roles["reviewer"],
            "active_unattributed": roles["unattributed"],
        },
    }


def monitor(client, experiment_id, campaign: Path, interval: float) -> None:
    """Publish changes to one MLflow run and finish it when the campaign ends."""
    tracking_run = client.create_run(experiment_id, run_name=campaign.name)
    run_id = tracking_run.info.run_id
    print(f"MLflow live run: {run_id}", flush=True)
    last = None
    step = 0
    try:
        while True:
            view = snapshot(campaign / ".run")
            if view != last:
                for key, value in view.items():
                    if key != "counters":
                        client.set_tag(run_id, f"arena.{key}",
                                       value if isinstance(value, str) else json.dumps(value))
                for key, value in view["counters"].items():
                    client.log_metric(run_id, key, value, step=step)
                step += 1
                last = view
                print(f"Phase: {view['phase']}; active roles: {view['active_roles']}", flush=True)
            if view["phase"] in {"complete", "failed"}:
                client.set_terminated(run_id, "FINISHED" if view["phase"] == "complete" else "FAILED")
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        client.set_terminated(run_id, "KILLED")


def main() -> None:
    """Connect the optional operator-side monitor to an MLflow tracking server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--interval", type=float, default=180, help="Polling interval in seconds (default: 180)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a finite positive number of seconds")
    if not (args.campaign / ".run/search.sqlite3").is_file():
        parser.error("Prepare the campaign before starting its monitor")
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    name = f"Arena / {args.campaign.name}"
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
    monitor(client, experiment_id, args.campaign.resolve(), args.interval)


if __name__ == "__main__":
    main()
