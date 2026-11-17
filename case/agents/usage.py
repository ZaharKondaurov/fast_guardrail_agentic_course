"""Persist model usage and agent turn duration without recording prompts."""

import json
import re
import threading
from contextvars import ContextVar
from pathlib import Path
from time import monotonic

agent_label = ContextVar("agent_label", default=("unattributed", None))
ROLE_HEADER = re.compile(r"^ARENA_ROLE: (researcher|reviewer)\s+ARENA_TASK: ([\w.-]{1,100})(?:\s|$)")


class UsageLog:
    def __init__(self, path: Path):
        """Открывает учёт запуска с общей блокировкой записи для параллельных задач."""
        self.path = path
        self.lock = threading.Lock()

    def write(self, event, started, response=None):
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
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


def summarize(directory: Path):
    """Суммирует время и токены по ролям и задачам с отдельным учётом пропусков."""
    groups = {}
    for path in sorted(directory.glob("*.usage.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
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
