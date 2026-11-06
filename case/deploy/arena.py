"""Operator commands for AR-12 campaigns."""

import argparse
import json
import math
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from urllib.error import URLError
from urllib.request import Request, urlopen

DEPLOY = Path(__file__).resolve().parent
ROOT = DEPLOY.parents[1]
ARENA = ROOT / "arena"
ARENA_PYTHON = ARENA / ".venv/bin/python"
BASELINES = {
    "matching": ("rules.json",),
    "boosting": ("catboost-proba.joblib",),
    "transformer": ("model.onnx", "model.json"),
}


def arena_command(module, *arguments):
    return [ARENA_PYTHON, "-m", f"evaluator.{module}", *arguments]


def arena_env():
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("ARENA_") and key != "OPENROUTER_API_KEY"
    }
    return {**env, "PYTHONPATH": str(ARENA)}


def arena_json(module, *arguments):
    return json.loads(subprocess.check_output(arena_command(module, *arguments), env=arena_env()))


def save_json(path, value):
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        staged.replace(path)


def build_runner(run):
    with TemporaryDirectory(prefix="ar12-runner-") as directory:
        context = Path(directory)
        shutil.copyfile(DEPLOY / "runner.Dockerfile", context / "Dockerfile")
        shutil.copyfile(ARENA / "runner/main.py", context / "main.py")
        subprocess.run([
            "uv", "export", "--locked", "--offline", "--format", "requirements-txt",
            "--only-group", "boosting", "--only-group", "transformer",
            "--no-emit-package", "torch", "--no-header", "--output-file",
            context / "requirements.txt", "--quiet",
        ], cwd=ROOT, check=True)
        image_id = context / "image-id"
        subprocess.run([
            "docker", "build", "--iidfile", image_id, "--tag", "ar12-runner:local", context,
        ], check=True)
        run.mkdir(mode=0o700, exist_ok=True)
        shutil.copyfile(image_id, run / "runner-image.txt")


def prepare_baselines(run, data):
    run.mkdir(mode=0o700, exist_ok=True)
    destination = run / "candidates"
    if destination.exists():
        raise ValueError(f"Candidate directory already exists: {destination}")
    with TemporaryDirectory(prefix="baselines-", dir=run) as directory:
        workspace = Path(directory)
        source = ROOT / "case/ways/baseline"
        packages = workspace / "packages"
        packages.mkdir()
        for name in BASELINES:
            train_package(name, source / name, data / "train.jsonl", packages / name)
        packages.rename(destination)
    print(destination)


def train_package(parent, source, dataset, destination, log=None):
    methods = Path("case/ways/baseline")
    env = arena_env()
    env.pop("PYTHONPATH")
    with TemporaryDirectory(dir=destination.parent) as directory:
        workspace = Path(directory)
        method = workspace / methods / parent
        method.mkdir(parents=True)
        shutil.copyfile(ROOT / methods / "calibration.py", method.parent / "calibration.py")
        for filename in ("train.py", "solve.py"):
            shutil.copyfile(source / filename, method / filename)
        subprocess.run([
            sys.executable, "-m", f"case.ways.baseline.{parent}.train", dataset,
        ], cwd=workspace, env=env, check=True, stdout=log,
           stderr=subprocess.STDOUT if log is not None else None)
        package = workspace / "package"
        package.mkdir()
        for filename in ("train.py", "solve.py", *BASELINES[parent]):
            shutil.copyfile(method / filename, package / filename)
        package.rename(destination)


def serve(campaign, config, state, dataset, dataset_id, conditions, port):
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    (state / "slots").mkdir(mode=0o700, exist_ok=True)
    token_file = state / "submit-token"
    try:
        with os.fdopen(os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as token:
            token.write(secrets.token_urlsafe(32) + "\n")
    except FileExistsError:
        pass
    image = (campaign / ".run/runner-image.txt").read_text().strip()
    command = arena_command("train", "serve", "--host", "127.0.0.1", "--port", str(port))
    env = {
        **arena_env(),
        "ARENA_CAMPAIGN_ID": campaign.name,
        "ARENA_CASE_ID": "ar-12",
        "ARENA_DATASET_ID": dataset_id,
        "ARENA_CONDITIONS_ID": conditions,
        "ARENA_DATASET_PATH": str(dataset),
        "ARENA_METRIC_PATH": str(campaign / "metric"),
        "ARENA_STATE_DIR": str(state / "evaluator"),
        "ARENA_TOKEN_FILE": str(token_file),
        "ARENA_RUNNER_IMAGE": image,
        "ARENA_RELEASE_ID": config["release_id"],
        "ARENA_USES_MODEL": "false",
        "ARENA_RUNNER_NETWORK": "none",
        "ARENA_RUNNER_TIMEOUT": str(config["runner_timeout_seconds"]),
        "ARENA_EXECUTION_SLOTS": "1",
        "ARENA_SLOT_DIR": str(state / "slots"),
        "ARENA_TRAIN_TRACKING": "false",
    }
    return command, env


def wait_for_service(url, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Train evaluator exited before startup")
        try:
            with urlopen(url + "/healthz", timeout=1) as response:
                if json.load(response) == {"status": "ok"}:
                    return
        except URLError:
            time.sleep(0.25)
    raise TimeoutError("Train evaluator did not start")


def evaluate_candidate(name, root, dataset, config, url, wait_seconds):
    journal = root / "search.sqlite3"
    measurement = f"{name}-train"
    for existing in arena_json("search", "--journal", journal, "export")["measurements"]:
        if existing["measurement_id"] == measurement and existing["state"] != "running":
            if any(existing["comparison"][key] != config[key]
                   for key in ("dataset_id", "conditions_id")):
                raise ValueError("Stored evaluation belongs to different campaign conditions")
            return existing
    access = ("--url", url, "--token-file", root / "submit-token")
    submitted = arena_json(
        "train", "submit", *access, "--journal", journal, "--candidate", name,
        "--measurement", measurement, "--dataset", config["dataset_id"],
        "--conditions", config["conditions_id"],
        "--train-data", dataset,
    )
    evaluation = ("--evaluation-id", submitted["evaluation_id"])
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        status = arena_json("train", "status", *access, *evaluation)["status"]
        if status != "RUNNING":
            break
        time.sleep(min(5, wait_seconds / 10))
    else:
        raise TimeoutError(f"Timed out waiting for {name}; rerun the same command to resume")
    arena_json(
        "train", "result", *access, *evaluation, "--journal", journal, "--measurement",
        measurement, "--output", root / f"{measurement}.json",
    )
    return next(row for row in arena_json("search", "--journal", journal, "export")["measurements"]
                if row["measurement_id"] == measurement)


def verify_candidate(name, package, root, journal, dataset, config, url):
    arena_json("search", "--journal", journal, "propose", "--id", name,
               "--directory", package, "--hypothesis", "Synthetic runner acceptance",
               "--change", "Acceptance only", "--tokens", "0", "--seconds", "0")
    result = evaluate_candidate(name, root, dataset, config, url, 120)
    if (
        result["state"] != "succeeded"
        or any(result[role]["total_tokens"] != 0 or not result[role]["tokens_complete"]
               for role in ("solver", "judge"))
        or (name == "control" and result["score"] != 1.0)
    ):
        raise RuntimeError(f"Acceptance failed for {name}. See {root / f'{name}-train.json'}")


def proposals(run):
    return arena_json("search", "--journal", run / "search.sqlite3", "export")["candidates"]


def register(run, name, directory, hypothesis, change, tokens=0, seconds=0,
             parent=None, tokens_complete=True):
    if name in {item["candidate_id"] for item in proposals(run)}:
        return
    options = ["--parent", parent] if parent else []
    if not tokens_complete:
        options.append("--tokens-incomplete")
    arena_json(
        "search", "--journal", run / "search.sqlite3", "propose", "--id", name,
        "--directory", directory, "--hypothesis", hypothesis, "--change", change,
        "--tokens", str(tokens), "--seconds", str(seconds), *options,
    )


def run_variant(run, parent, source, train, dataset, config, url, wait_seconds,
                hypothesis, change):
    name = f"variant-{parent}"
    existing = {item["candidate_id"] for item in proposals(run)}
    if parent not in existing:
        raise ValueError(f"Register the parent first: {parent}")
    destination = run / "variants" / name
    if not destination.exists():
        if name in existing:
            raise ValueError(f"Candidate {name} is registered outside this workflow")
        status = arena_json("search", "--journal", run / "search.sqlite3", "status")
        if status["stopped"] or len(existing) >= status["max_solutions"]:
            raise ValueError("Campaign is stopped or has no remaining proposal slots")
        usage = json.loads((run / "research" / f"{name}.json").read_text(encoding="utf-8"))
        tokens = usage.get("total_tokens", 0)
        if usage.get("candidate_id") != name or type(tokens) is not int or tokens < 0:
            raise ValueError("Invalid research usage record")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        started = time.monotonic()
        with TemporaryDirectory(dir=destination.parent) as directory:
            stage = Path(directory) / "variant"
            stage.mkdir()
            package = stage / "package"
            error = None
            try:
                with (stage / "training.log").open("w", encoding="utf-8") as log:
                    train_package(parent, source, train, package, log)
            except (OSError, subprocess.CalledProcessError) as failure:
                error = type(failure).__name__
                package.mkdir()
                for filename in ("train.py", "solve.py"):
                    if (source / filename).is_file():
                        shutil.copyfile(source / filename, package / filename)
            now = time.time()
            elapsed = (
                max(0, now - usage["started_at"])
                if "started_at" in usage else usage.get("seconds", 0) + time.monotonic() - started
            )
            save_json(stage / "preparation.json", {
                "hypothesis": hypothesis, "change": change, "error": error,
                "seconds": elapsed, "total_tokens": tokens,
                "tokens_complete": tokens > 0 and usage.get("tokens_complete", True),
            })
            stage.rename(destination)
    record = json.loads((destination / "preparation.json").read_text(encoding="utf-8"))
    description = record["change"]
    if record["error"]:
        description += f" (preparation failed: {record['error']})"
    register(run, name, destination / "package", record["hypothesis"], description,
             record["total_tokens"], record["seconds"], parent, record["tokens_complete"])
    if record["error"]:
        return {
            "candidate_id": name, "state": "preparation_failed", "error": record["error"],
            "log": str(destination / "training.log"),
        }
    return evaluate_candidate(name, run, dataset, config, url, wait_seconds)


def finish_pilot(run, winner, reason):
    journal = run / "search.sqlite3"
    evidence = arena_json("search", "--journal", journal, "export")
    eligible = [row for row in evidence["measurements"]
                if row["candidate_id"] == winner and row["state"] == "succeeded"]
    if not eligible:
        raise ValueError("The winner must have a successful evaluation")
    with TemporaryDirectory(dir=run) as directory:
        snapshot = Path(directory) / "candidate"
        bundle = Path(directory) / "winner.json"
        arena_json("search", "--journal", journal, "restore", "--candidate", winner,
                   "--directory", snapshot)
        arena_json("client", "pack", "--candidate", f"{winner}={snapshot}", "--output", bundle)
        finalization = arena_json("search", "--journal", journal, "finalize", "--bundle", bundle)
    save_json(run / "final-selection.json", {
        "candidate_id": winner, "reason": reason, "measurement": eligible[-1], **finalization,
    })
    print(run / "final-selection.json")


def accept(campaign, config, run):
    run.mkdir(mode=0o700, exist_ok=True)
    root = Path(mkdtemp(prefix="acceptance-", dir=run))
    print(f"Acceptance evidence: {root}", flush=True)
    dataset = root / "synthetic.jsonl"
    dataset.write_text(
        '{"id":"safe","input":"safe","reference":"0"}\n'
        '{"id":"unsafe","input":"unsafe","reference":"1"}\n', encoding="utf-8",
    )
    control = root / "control"
    control.mkdir()
    (control / "solve.py").write_text(
        'import json, sys\n'
        'print(json.dumps({"answer": str(int(json.load(sys.stdin)["input"] == "unsafe"))}))\n',
        encoding="utf-8",
    )
    packages = {"control": control, **{name: run / "candidates" / name for name in BASELINES}}
    journal = root / "search.sqlite3"
    arena_json("search", "--journal", journal, "init", "--max-solutions", str(len(packages) + 1))
    synthetic = {**config, "dataset_id": "synthetic-v1", "conditions_id": "acceptance-v1"}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command, env = serve(campaign, config, root, dataset, "synthetic-v1", "acceptance-v1", port)
    with (root / "service.log").open("w") as log, subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT,
    ) as service:
        try:
            wait_for_service(url, service)
            for name, package in packages.items():
                verify_candidate(name, package, root, journal, dataset, synthetic, url)
            training = root / "synthetic-train.jsonl"
            training.write_text('{"input":"unsafe","label":1}\n{"input":"safe","label":0}\n')
            (root / "research").mkdir()
            save_json(root / "research/variant-matching.json", {
                "candidate_id": "variant-matching", "total_tokens": 123, "seconds": 1,
            })
            source = ROOT / "case/ways/baseline/matching"
            variant = run_variant(root, "matching", source, training, dataset, synthetic,
                                  url, 120, "Synthetic variant", "Workflow acceptance")
            evidence = arena_json("search", "--journal", journal, "export")
            if len(evidence["measurements"]) != len(packages) + 1 or any(
                row["state"] != "succeeded" for row in evidence["measurements"]
            ):
                raise RuntimeError("Acceptance results are missing from the journal")
            proposal = next(item for item in evidence["candidates"]
                            if item["candidate_id"] == "variant-matching")
            if (
                variant["state"] != "succeeded" or proposal["parents"] != ["matching"]
                or proposal["resources"]["total_tokens"] != 123
            ):
                raise RuntimeError("Variant lineage or research usage is missing")
            resumed = run_variant(root, "matching", source, training, dataset, synthetic,
                                  url, 120, "Synthetic variant", "Workflow acceptance")
            if resumed != variant:
                raise RuntimeError("Completed variant could not be resumed")
            finish_pilot(root, "variant-matching", "Synthetic finalization check")
            evidence = arena_json("search", "--journal", journal, "export")
            (root / "journal.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(root / "journal.json")
        finally:
            service.terminate()
            try:
                service.wait(timeout=10)
            except subprocess.TimeoutExpired:
                service.kill()


def openrouter_request(model, prompt, max_tokens):
    token = os.getenv("OPENROUTER_API_KEY")
    if not token:
        raise ValueError("Set OPENROUTER_API_KEY in the operator environment")
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    choices = payload.get("choices") or []
    answer = choices[0]["message"].get("content") if choices else None
    tokens = (payload.get("usage") or {}).get("total_tokens")
    if not isinstance(answer, str) or not answer.strip() or type(tokens) is not int or tokens <= 0:
        raise ValueError("OpenRouter did not return a completion with token usage")
    return {
        "model": model, "response_id": payload.get("id"), "total_tokens": tokens,
        "seconds": time.monotonic() - started, "answer": answer,
    }


def openrouter_smoke(model, run):
    result = openrouter_request(model, "Reply with the single word READY.", 256)
    result.pop("answer")
    run.mkdir(mode=0o700, exist_ok=True)
    save_json(run / "openrouter-smoke.json", result)
    print(json.dumps(result))


def research(run, model, parent):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY in the operator environment")
    smoke = json.loads((run / "openrouter-smoke.json").read_text(encoding="utf-8"))
    tokens = smoke.get("total_tokens")
    if smoke.get("model") != model or type(tokens) is not int or tokens <= 0:
        raise ValueError("Run a successful openrouter-smoke for the configured model first")
    journal = run / "search.sqlite3"
    status = arena_json("search", "--journal", journal, "status")
    if status["stopped"] or status["finalization"] or status["remaining_proposals"] <= 0:
        raise ValueError("Campaign is stopped, finalized or has no remaining proposal slots")
    candidates = {item["candidate_id"] for item in proposals(run)}
    if parent not in candidates or f"variant-{parent}" in candidates:
        raise ValueError("Register the parent first and use research only for a new variant")
    source = ROOT / "case/ways/baseline" / parent
    files = "\n\n".join(
        f"{name}:\n{(source / name).read_text(encoding='utf-8')}"
        for name in ("train.py", "solve.py")
    )
    prompt = (
        f"Propose one concrete change to the AR-12 {parent} binary text classifier. "
        "Maximize true-positive rate subject to a false-positive rate of at most 5%. "
        "The candidate trains on labeled local text and emits JSON with answer '0' or '1'. "
        "Keep the existing dependencies and interface. Provide a hypothesis, a patch, and a "
        "local verification plan. Here are the two source files:\n\n" + files
    )
    output = run / "research" / f"variant-{parent}.json"
    started = time.time()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        try:
            result = openrouter_request(model, prompt, 4096)
        except (OSError, ValueError):
            json.dump({"candidate_id": f"variant-{parent}", "started_at": started,
                       "tokens_complete": False}, stream)
            raise
        json.dump({"candidate_id": f"variant-{parent}", "started_at": started, **result},
                  stream, indent=2)
        stream.write("\n")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-runner")
    commands.add_parser("prepare-baselines")
    commands.add_parser("accept")
    commands.add_parser("openrouter-smoke")
    researcher = commands.add_parser("research")
    researcher.add_argument("parent", choices=BASELINES)
    service = commands.add_parser("serve")
    service.add_argument("--port", type=int, default=8765)
    initialize = commands.add_parser("init-journal")
    initialize.add_argument("--max-solutions", type=int, required=True)
    baselines = commands.add_parser("run-baselines")
    variant = commands.add_parser("run-variant")
    variant.add_argument("parent", choices=BASELINES)
    variant.add_argument("--reviewed-source", type=Path, required=True)
    variant.add_argument("--hypothesis", required=True)
    variant.add_argument("--change", required=True)
    for command in (baselines, variant):
        command.add_argument("--url", default="http://127.0.0.1:8765")
        command.add_argument("--wait-seconds", type=float, default=1800)
    finish = commands.add_parser("finish-pilot")
    finish.add_argument("--winner", required=True)
    finish.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        campaign = args.campaign.resolve()
        config = json.loads((campaign / "runtime.json").read_text(encoding="utf-8"))
        run = campaign / ".run"
        data = ROOT / config["data_dir"]
        dataset = data / "evaluation.arena.jsonl"
        if args.command in {"run-baselines", "run-variant"} and (
            not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0
        ):
            raise ValueError("--wait-seconds must be finite and positive")
        if args.command == "serve":
            command, env = serve(
                campaign, config, run, dataset,
                config["dataset_id"], config["conditions_id"], args.port,
            )
            os.execve(ARENA_PYTHON, command, env)
        elif args.command == "init-journal":
            run.mkdir(mode=0o700, exist_ok=True)
            print(arena_json("search", "--journal", run / "search.sqlite3", "init",
                             "--max-solutions", str(args.max_solutions)))
        elif args.command == "research":
            research(run, config["research_model"], args.parent)
        elif args.command == "run-baselines":
            for name in BASELINES:
                package = run / "candidates" / name
                if not (package / "solve.py").is_file():
                    raise ValueError("Run prepare-baselines first")
                register(run, name, package, "Baseline", "Reference package")
                print(json.dumps(evaluate_candidate(
                    name, run, dataset, config, args.url, args.wait_seconds,
                )), flush=True)
        elif args.command == "run-variant":
            result = run_variant(run, args.parent, args.reviewed_source, data / "train.jsonl",
                                 dataset, config, args.url, args.wait_seconds,
                                 args.hypothesis, args.change)
            print(json.dumps(result))
            if result["state"] != "succeeded":
                return 1
        elif args.command == "finish-pilot":
            finish_pilot(run, args.winner, args.reason)
        elif args.command == "prepare-baselines":
            prepare_baselines(run, data)
        elif args.command == "build-runner":
            build_runner(run)
        elif args.command == "accept":
            accept(campaign, config, run)
        elif args.command == "openrouter-smoke":
            openrouter_smoke(config["research_model"], run)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
