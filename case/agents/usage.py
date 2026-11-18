"""Persist model usage and agent turn duration without recording prompts."""

import json
import hashlib
import os
import re
import threading
from contextvars import ContextVar
from pathlib import Path
from time import monotonic
from tempfile import mkstemp

agent_label = ContextVar("agent_label", default=("unattributed", None))
ROLE_HEADER = re.compile(r"^ARENA_ROLE: (researcher|reviewer)\s+ARENA_TASK: ([\w.-]{1,100})(?:\s|$)")


class UsageLog:
    def __init__(self, path: Path):
        """Открывает учёт запуска с общей блокировкой записи для параллельных задач."""
        self.path = path
        self.lock = threading.Lock()

    def save_response(self, response, role, task, generation_parameters):
        """Атомарно сохраняет полный последний выход модели, отдельно удерживая обрезанный."""
        choices = [
            {"index": choice.index, "finish_reason": choice.finish_reason,
             "message": choice.message.model_dump(mode="json", exclude_none=True)}
            for choice in response.choices
        ]
        snapshot = {
            "role": role, "task": task, "generation_id": response.id,
            "model": response.model, "choices": choices,
            "generation_parameters": generation_parameters,
        }
        stem = self.path.name.removesuffix(".usage.jsonl")
        # Хеш метки исключает пути из названий задач и разводит параллельных работников.
        label = hashlib.sha256(json.dumps([role, task]).encode()).hexdigest()[:16]
        names = [f"{stem}.last-response.json", f"{stem}.last-response-{label}.json"]
        if any(choice["finish_reason"] == "length" for choice in choices):
            # Успешное продолжение не должно затереть ответ, который мог зациклиться.
            names.extend([f"{stem}.last-truncated-response.json",
                          f"{stem}.last-truncated-response-{label}.json"])
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
        directory = self.path.parent / "responses"
        directory.mkdir(mode=0o700, exist_ok=True)
        for name in names:
            descriptor, temporary = mkstemp(prefix=".response-", dir=directory)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                os.replace(temporary, directory / name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def write(self, event, started, response=None, *, turn_id=None, status=None, request=None):
        """Добавляет событие с ролью, временем и подтверждённым расходом без текста запроса."""
        role, task = agent_label.get()
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", None)
        row = {
            "event": event, "role": role, "task": task,
            "seconds": monotonic() - started,
        }
        if event == "model":
            row["total_tokens"] = tokens if type(tokens) is int and tokens >= 0 else None
            row["generation_id"] = getattr(response, "id", None)
            # Только числовые лимиты и режим reasoning; промпты, заголовки и ключи не сохраняем.
            request = request or {}
            parameters = {key: request[key] for key in (
                "max_tokens", "max_completion_tokens", "reasoning_effort",
            ) if key in request}
            reasoning = (request.get("extra_body") or {}).get("reasoning")
            if reasoning is not None:
                parameters["reasoning"] = reasoning
            row["generation_parameters"] = parameters
        if turn_id is not None:
            row["turn_id"] = turn_id
            row["status"] = status
        with self.lock:
            if event == "model" and response is not None:
                # Запись происходит в транспортном адаптере до обработки ответа Hermes.
                self.save_response(response, role, task, parameters)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")


def summarize(directory: Path):
    """Суммирует время и токены по ролям и задачам с отдельным учётом пропусков."""
    groups = {}
    for path in sorted(directory.glob("*.usage.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["event"] == "turn_start":
                continue
            key = (row["role"], row["task"])
            totals = groups.setdefault(key, {
                "role": key[0], "task": key[1], "calls": 0, "known_tokens": 0,
                "calls_without_usage": 0, "model_seconds": 0.0, "turn_seconds": 0.0,
            })
            if row["event"] == "turn":
                totals["turn_seconds"] += row["seconds"]
                continue
            totals["calls"] += 1
            totals["model_seconds"] += row["seconds"]
            # Отсутствие usage учитывается отдельно от подтверждённого нулевого расхода.
            if row["total_tokens"] is None:
                totals["calls_without_usage"] += 1
            else:
                totals["known_tokens"] += row["total_tokens"]
    return list(groups.values())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
