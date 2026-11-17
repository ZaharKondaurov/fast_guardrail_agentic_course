"""Operator entry points for the AR-12 private evaluator."""

import argparse
import os
import secrets
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

from evaluator.client import ClientError, pack_candidate
from evaluator.contracts import Submission
from evaluator.metric_package import load_metric, metric_package_provenance
from evaluator.search import Journal
from evaluator.service import Settings, create_app

ROOT = Path(__file__).resolve().parents[2]


def freeze_finalists(journal, candidate_ids, expected_count, output):
    if expected_count < 1 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Finalist IDs must be unique and K must be positive")
    if len(candidate_ids) != expected_count:
        raise ValueError(f"Exactly {expected_count} finalist IDs are required")
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as directory:
        workspace = Path(directory)
        candidates = []
        for index, candidate_id in enumerate(candidate_ids):
            snapshot = workspace / str(index)
            journal.restore(candidate_id, snapshot)
            candidates.append(pack_candidate(candidate_id, snapshot))
        submission = Submission(candidates=candidates)
        bundle = workspace / "finalists.json"
        bundle.write_text(submission.model_dump_json(indent=2) + "\n", encoding="utf-8")
        journal.finalize(submission)
        os.link(bundle, output)


def private_settings(args):
    dataset = args.dataset.resolve()
    state = args.state.resolve()
    if dataset.is_relative_to(ROOT) or state.is_relative_to(ROOT):
        raise ValueError("Private data and state must be outside the project repository")
    if not dataset.is_file():
        raise ValueError("Provide the operator's private Arena JSONL file")
    with dataset.open("rb") as stream:
        if not any(line.strip() for line in stream):
            raise ValueError("Private is empty; skip the private evaluator")
    if not 1 <= args.winners <= args.finalists:
        raise ValueError("Expected 1 <= winners <= finalists")
    if Journal(args.journal).status()["max_solutions"] != args.max_solutions:
        raise ValueError("The journal proposal limit differs from the campaign configuration")
    metric = args.metric.resolve()
    if metric_package_provenance(metric) is None:
        raise ValueError("A frozen metric package is required")
    if load_metric(metric).version in {"ar12-gated-tpr-v3", "ar12-p99-v1", "ar12-throughput-v1"}:
        raise ValueError("This metric requires a trusted warm inference timer; the stock runner cannot supply it")

    state.mkdir(mode=0o700, exist_ok=True)
    if state.stat().st_mode & 0o077:
        raise ValueError("Private state directory must be accessible only to its owner")
    token_file = state / "submit-token"
    try:
        with os.fdopen(os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    except FileExistsError:
        pass
    if token_file.is_symlink() or token_file.stat().st_mode & 0o077:
        raise ValueError("Private submit token must be an owner-only file")
    return Settings(
        dataset_path=dataset,
        metric_path=metric,
        state_dir=state,
        token=token_file.read_text(encoding="utf-8").strip(),
        runner_image=args.runner_image,
        runner_timeout=args.runner_timeout,
        runner_network="none",
        uses_model=False,
        execution_slots=1,
        release_id=args.release_id,
        finalists=args.finalists,
        output_count=args.winners,
        split=getattr(args, "split", "private"),
        search_mode="campaign",
        search_journal=args.journal.resolve(),
        search_max_solutions=args.max_solutions,
        search_stop_path=args.stop_evidence.resolve() if args.stop_evidence else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    freeze = commands.add_parser("freeze")
    for command in (serve, freeze):
        command.add_argument("--journal", type=Path, required=True)
        command.add_argument("--finalists", type=int, required=True, metavar="K")
    for name in ("dataset", "state", "metric"):
        serve.add_argument(f"--{name}", type=Path, required=True)
    serve.add_argument("--runner-image", required=True)
    serve.add_argument("--runner-timeout", type=float, default=30)
    serve.add_argument("--release-id", required=True)
    serve.add_argument("--max-solutions", type=int, required=True)
    serve.add_argument("--winners", type=int, required=True, metavar="N")
    serve.add_argument("--stop-evidence", type=Path)
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--split", choices=("public", "private"), default="private")
    freeze.add_argument("--candidate", action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            freeze_finalists(Journal(args.journal), args.candidate, args.finalists, args.output)
            print(args.output)
        else:
            uvicorn.run(create_app(private_settings(args)), host=args.host, port=args.port,
                        access_log=False)
    except (OSError, ValueError, ClientError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
