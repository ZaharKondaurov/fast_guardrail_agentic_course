"""Проверки сохранения ответа до того, как Hermes отбрасывает вырожденную генерацию."""

import json
from types import SimpleNamespace
from time import monotonic
from unittest.mock import Mock

from case.agents.usage import UsageLog, agent_label, summarize


def response(text, reason="stop", generation="gen-1"):
    """Создаёт ответ SDK с текстом, рассуждением и вызовом инструмента без реального API."""
    message = Mock()
    message.model_dump.return_value = {
        "role": "assistant", "content": text, "reasoning_content": "thinking",
        "tool_calls": [{"id": "call-1", "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"}}],
    }
    return SimpleNamespace(id=generation, model="test-model", usage=SimpleNamespace(total_tokens=42),
                           choices=[SimpleNamespace(index=0, finish_reason=reason, message=message)])


def test_last_response_preserves_full_output_and_is_private(tmp_path):
    """Не обрезает повторяющийся текст и сохраняет диагностические поля с правами владельца."""
    log = UsageLog(tmp_path / "orchestrator-1.usage.jsonl")
    text = "повтор " * 10000
    log.write("model", monotonic(), response(text, "length"))
    path = tmp_path / "responses/orchestrator-1.last-response.json"
    saved = json.loads(path.read_text())
    assert saved["choices"][0]["message"]["content"] == text
    assert saved["choices"][0]["message"]["reasoning_content"] == "thinking"
    assert saved["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "terminal"
    assert saved["choices"][0]["finish_reason"] == "length"
    assert saved["generation_id"] == "gen-1"
    assert path.stat().st_mode & 0o777 == 0o600
    assert summarize(tmp_path)[0]["known_tokens"] == 42


def test_continuation_and_api_error_do_not_erase_truncated_response(tmp_path):
    """Удерживает обрезанный ответ после успешного продолжения и ошибки следующего запроса."""
    log = UsageLog(tmp_path / "run.usage.jsonl")
    log.write("model", monotonic(), response("loop", "length"))
    log.write("model", monotonic(), response("recovered", generation="gen-2"))
    log.write("model", monotonic())
    assert json.loads((tmp_path / "responses/run.last-response.json").read_text())["generation_id"] == "gen-2"
    assert json.loads((tmp_path / "responses/run.last-truncated-response.json").read_text())["generation_id"] == "gen-1"
    assert not list((tmp_path / "responses").glob(".response-*"))


def test_delegated_tasks_have_separate_last_outputs(tmp_path):
    """Ответ исследователя не затирает отдельную копию последнего выхода оркестратора."""
    log = UsageLog(tmp_path / "run.usage.jsonl")
    for label, text in [(('orchestrator', 'main'), 'parent'), (('researcher', '../variant'), 'child')]:
        token = agent_label.set(label)
        try:
            log.write("model", monotonic(), response(text))
        finally:
            agent_label.reset(token)
    snapshots = [json.loads(path.read_text()) for path in (tmp_path / "responses").glob("run.last-response-*.json")]
    assert {(row["role"], row["task"]) for row in snapshots} == {
        ('orchestrator', 'main'), ('researcher', '../variant'),
    }


def test_logs_actual_generation_settings_without_request_contents(tmp_path):
    """Показывает отправленный бюджет и reasoning без сохранения запроса и секретов."""
    path = tmp_path / "run.usage.jsonl"
    request = {"max_tokens": 16384, "messages": [{"content": "private-prompt"}],
               "api_key": "secret-key", "extra_body": {"reasoning": {"enabled": False}}}
    UsageLog(path).write("model", monotonic(), response("answer"), request=request)
    expected = {"max_tokens": 16384, "reasoning": {"enabled": False}}
    assert json.loads(path.read_text())["generation_parameters"] == expected
    snapshot = tmp_path / "responses/run.last-response.json"
    assert json.loads(snapshot.read_text())["generation_parameters"] == expected
    assert "private-prompt" not in path.read_text() + snapshot.read_text()
    assert "secret-key" not in path.read_text() + snapshot.read_text()
