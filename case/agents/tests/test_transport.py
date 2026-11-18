"""Проверки параметров запросов Hermes без вызова внешнего провайдера."""

import pytest

from case.agents.hermes_runtime import gateway_request


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
def test_transport_preserves_hermes_output_budget(key):
    """Передаёт явно заданный бюджет вывода вместо скрытого лимита провайдера."""
    request = {"messages": [{"role": "user", "content": "Continue"}], key: 16384}
    result = gateway_request(request, "campaign-model")
    assert result[key] == 16384
    assert result["model"] == "campaign-model"
    assert result["stream"] is False


def test_transport_preserves_budget_from_extra_body():
    """Сохраняет бюджет вспомогательного запроса, переданный через extra_body SDK."""
    assert gateway_request({"extra_body": {"max_tokens": 4096}}, "model")["max_tokens"] == 4096


def test_transport_does_not_invent_budget_when_hermes_omits_it():
    """Не вводит произвольный лимит токенов и не сериализует sentinel SDK."""
    assert "max_tokens" not in gateway_request({"max_tokens": object()}, "model")
    assert "max_completion_tokens" not in gateway_request({}, "model")


def test_transport_honors_reasoning_off_on_empty_answer_recovery():
    """Не включает reasoning обратно после восстановительного запроса Hermes."""
    result = gateway_request({"reasoning_effort": "none", "extra_body": {
        "reasoning": {"enabled": False, "effort": "none"},
    }}, "model")
    assert result["reasoning_effort"] == "none"
    assert result["extra_body"]["reasoning"] == {"enabled": False, "effort": "none"}


def test_transport_does_not_override_auxiliary_reasoning_or_provider_default():
    """Сохраняет настройку вспомогательного запроса и отсутствие явного режима."""
    assert gateway_request({"reasoning_effort": "low"}, "model")["reasoning_effort"] == "low"
    assert "reasoning_effort" not in gateway_request({}, "model")
