"""Run AR-12 agent roles with the Arena Hermes transport and role prompts."""

import argparse
import json
import os
import time
from pathlib import Path

from case.agents import hermes_runtime as runtime
from case.agents.private_status import private_status, wait_private
from case.agents.usage import UsageLog
from evaluator.client import search_readiness
from evaluator.search import Journal
from evaluator.reliability import request_timeout

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "arena"
ROLES = ("prepare", "orchestrator", "researcher", "reviewer")


def prompt_for(role, campaign, assignment, journal_path=None, private=None, train_url=None, public_url=None):
    """Собирает инструкции роли, условия кампании и разрешённые адреса оценщиков."""
    documents = [ARENA / "prompts" / f"{role}.md"]
    if role == "orchestrator":
        documents.insert(0, ARENA / "prompts/start.md")
    documents.extend([
        ARENA / "common/research-protocol.md",
        ARENA / "common/search-history.md",
        ARENA / "common/isolation.md",
        ROOT / "case/README.md",
        campaign / "campaign.md",
    ])
    sections = [f"Campaign directory: {campaign}\nRole: {role}\nAssignment: {assignment}"]
    sections.append(f"Canonical shared journal: {journal_path}")
    sections.append("Campaign runtime configuration:\n" + (campaign / "runtime.json").read_text())
    if train_url:
        sections.append(
            f"Train URL: {train_url}; token file: /run/arena/train-token. "
            "Training data: /workspace/data/train.jsonl. Use "
            "/workspace/data/evaluation.arena.jsonl as --train-data for Arena train submit. "
            "Prepared packages: /workspace/baselines. Use /opt/arena/.venv/bin/python "
            "for evaluator modules and /opt/ar12/.venv/bin/python for training."
        )
        metric = json.loads((campaign / "metric/metric.json").read_text(encoding="utf-8"))
        if metric["version"] == "ar12-f1-v1":
            minimum_throughput = metric["aggregation"]["argv"][-1]
            sections.append(
                "After collecting a train result, report its diagnostics with "
                f"python {campaign / 'metric/score.py'} --train-report {minimum_throughput} "
                "< train-result.json. Report F1, TPR, FPR, recall, precision, p99, "
                "throughput and gate statuses. Use F1 to compare candidates."
            )
    if public_url:
        sections.append(
            f"Public URL: {public_url}; token file: /run/arena/public-token. "
            "For reviewed candidates, pack one frozen snapshot per public submission. "
            "Record public start and terminal result in evaluator.search with split public, "
            "the configured release conditions and only the aggregate score or failure. "
            "Compare these results before selecting private finalists."
        )
    sections.append(
        "For each new proposal record elapsed preparation seconds and actual model usage "
        "attributable to that proposal. If usage is unavailable or partial, use "
        "--tokens-incomplete. Record failed proposals too. Use variant- IDs for newly "
        "proposed solutions so the cost report can distinguish them from baselines."
    )
    if private:
        sections.append("Private submit access and exact finalist counts:\n" + json.dumps(private))
    else:
        sections.append("This pilot has empty private. Finalize the selected snapshot in the journal.")
    for document in documents:
        sections.append(f"Source: {document}\n{document.read_text(encoding='utf-8')}")
    sections.append(
        "Use evaluator.search and evaluator.train as tools. All directions use the same "
        "canonical journal. Start every delegated prompt with ARENA_ROLE: researcher or "
        "ARENA_ROLE: reviewer on line one and ARENA_TASK: candidate-id on line two. "
        "For each delegated researcher pass arena/prompts/researcher.md "
        "and the campaign, scope and quota. For independent review pass "
        "arena/prompts/reviewer.md and the exact snapshot. Read the referenced common files. "
        "Store claims, findings and reviews in the campaign board. Use isolated worktrees. "
        "The operator's preparation and private services run outside your workspace. "
        "Use only configured model access. Do not execute operator deploy commands."
    )
    return "\n\n".join(sections)


def run_role(factory, prompt, role, journal=None, read_private=None, check=lambda: None):
    """Выполняет роль Hermes и продолжает поиск до подтверждённой финализации."""
    previous = journal.status() if journal else None
    if role == "orchestrator" and previous["finalization"] and not previous["running"]:
        complete = not previous["remaining_proposals"] or previous["stopped"]
        if complete and (read_private is None or wait_private(
            read_private(), previous, read_private, time.sleep
        )):
            return {"completed": True, "final_response": "Campaign already finalized"}
    agent = factory()
    history = None
    idle = 0
    try:
        while True:
            result = agent.run_conversation(
                prompt,
                system_message="Follow the supplied Arena role, campaign and access boundaries.",
                conversation_history=history,
            )
            check()
            if result.get("failed") or not result.get("completed"):
                raise RuntimeError("Hermes turn failed; campaign remains incomplete")
            if role != "orchestrator":
                return result
            status = journal.status()
            complete = status["finalization"] and not status["running"] and (
                status["stopped"] or not status["remaining_proposals"]
            )
            if complete:
                if read_private is None:
                    return result
                if wait_private(read_private(), status, read_private, time.sleep):
                    return result
            # Ограничиваем повторные ходы при неизменном журнале, чтобы не расходовать токены на опрос.
            idle = idle + 1 if status == previous else 0
            if idle >= 2:
                raise RuntimeError("Two turns made no journal progress; inspect saved agent result")
            history = result.get("messages")
            if not history:
                raise RuntimeError("Hermes returned no conversation history")
            prompt = (
                "Continue the campaign under its existing conditions. Current journal status: "
                + json.dumps(status)
                + ". Honor stop requests. Collect pending results, review snapshots and finalize "
                "when authorized. If private is configured, submit the frozen batch and wait "
                "only for status. Never read private scores."
            )
            previous = status
    finally:
        agent.close()


def main():
    """Проверяет готовность кампании и запускает выбранную роль с учётом ресурсов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=ROLES)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--assignment", default="Execute the configured campaign role")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--train-url")
    parser.add_argument("--public-url")
    parser.add_argument("--private-config", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    config = json.loads((campaign / "runtime.json").read_text())
    access = runtime.credentials(args.credentials_file)
    model = config["research_model"]
    if access.get("model") != model:
        raise ValueError("Model credentials differ from the campaign model")
    journal = Journal(args.journal) if args.journal else None
    if args.role == "orchestrator":
        if journal is None:
            raise ValueError("Orchestrator requires the canonical journal")
        preparation = json.loads((campaign / ".run/preparation.json").read_text())
        if preparation.get("state") != "ready":
            raise ValueError("Complete campaign preparation before search")
    read_private = None
    private = None
    if args.role == "orchestrator" and config.get("private_required", True) and not args.private_config:
        raise ValueError("This campaign requires configured private access before search")
    if args.role == "orchestrator" and config.get("public_required") and not args.public_url:
        raise ValueError("This campaign requires configured public access before search")
    if args.private_config:
        if journal is None:
            raise ValueError("Private access requires the canonical journal")
        private = json.loads(args.private_config.read_text())
        # Проверяем условия закрытой серии до первого модельного хода.
        search_readiness(private["url"], Path(private["token_file"]), require_campaign=True,
                         max_solutions=journal.status()["max_solutions"],
                         finalists=private["finalists"], output_count=private["output_count"],
                         release_id=config["release_id"])
        read_private = lambda: private_status(private["url"], Path(private["token_file"]))
    home = Path(os.environ.get("HERMES_HOME", "/tmp/ar12-hermes"))
    runtime.prepare_home(home, model)
    timeout = access.get("model_request_timeout", request_timeout(access.get("model_timeout", 21600)))
    output = campaign / "board" / "agent-runs"
    output.mkdir(parents=True, exist_ok=True)
    record = output / f"{args.role}-{time.time_ns()}.json"
    record.write_text(json.dumps({"state": "running", "role": args.role, "model": model}))
    usage_log = UsageLog(record.with_suffix(".usage.jsonl"))
    state = runtime.install(
        access["model_base_url"], model, timeout, access.get("api_key", "arena"),
        usage_log=usage_log,
    )
    from run_agent import AIAgent

    def create_agent():
        """Создаёт агента с фиксированной моделью и меткой для раздельного учёта вызовов."""
        agent = AIAgent(
            model=model, base_url=runtime.CONFIG_BASE_URL, api_key="arena", provider="custom",
            api_mode="chat_completions", reasoning_config={"enabled": True, "effort": "max"},
            enabled_toolsets=["terminal", "file", "delegation", "todo"], max_iterations=10000,
            cwd=str(ROOT), skip_memory=True, skip_background_review=True,
            skip_context_files=True, save_trajectories=False, quiet_mode=True,
        )
        agent._ar12_usage_label = (args.role, args.assignment)
        return agent

    try:
        result = run_role(
            create_agent,
            prompt_for(args.role, campaign, args.assignment, args.journal, private, args.train_url,
                       args.public_url),
            args.role, journal, read_private, state.check,
        )
    except Exception:
        record.write_text(json.dumps({"state": "failed", "role": args.role, "model": model}))
        raise
    record.write_text(json.dumps({"state": "complete", "role": args.role, "model": model,
                                  "final_response": result.get("final_response", "")}, ensure_ascii=False))
    print(json.dumps({"answer": result.get("final_response", "")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
