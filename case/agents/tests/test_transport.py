"""Проверки передачи лимитов выхода через адаптер без обращения к модели."""

import pytest

from case.agents.hermes_runtime import gateway_request


@pytest.mark.parametrize("name", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("extra_body", [False, True])
def test_preserves_explicit_output_budget(name, extra_body):
    """Сохраняет лимит как из аргументов SDK, так и из extra_body."""
    budget = {name: 16384}
    request = {"extra_body": budget} if extra_body else budget
    result = gateway_request(request, "campaign-model")
    assert result[name] == 16384
    assert result["model"] == "campaign-model"
    assert result["stream"] is False


@pytest.mark.parametrize("value", [None, 0, -1, True, "16384", object()])
def test_omits_unset_or_invalid_output_budget(value):
    """Не передаёт sentinel SDK и некорректные значения как лимит токенов."""
    result = gateway_request({"max_tokens": value}, "model")
    assert "max_tokens" not in result


def test_does_not_set_default_budget_or_mutate_request():
    """Не вводит новый лимит и не изменяет исходные аргументы Hermes."""
    request = {"extra_body": {"max_completion_tokens": 8192}}
    assert "max_tokens" not in gateway_request({}, "model")
    assert "max_completion_tokens" not in gateway_request({}, "model")
    gateway_request(request, "model")
    assert request == {"extra_body": {"max_completion_tokens": 8192}}
