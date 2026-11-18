from unittest.mock import Mock

import pytest

from case.agents.run import run_role


def status(remaining=1, finalized=False):
    """Создаёт минимальный снимок журнала для проверки переходов оркестратора."""
    return {"remaining_proposals": remaining, "finalization": finalized,
            "running": [], "stopped": False}


def test_orchestrator_continues_until_journal_finalized():
    """Проверяет продолжение поиска до фиксации итогового пакета."""
    agent = Mock()
    agent.run_conversation.return_value = {
        "completed": True, "messages": [{"role": "assistant", "content": "continue"}],
        "final_response": "done",
    }
    journal = Mock()
    journal.status.side_effect = [status(), status(0), status(0, True)]
    result = run_role(lambda: agent, "start", "orchestrator", journal)
    assert result["final_response"] == "done"
    assert agent.run_conversation.call_count == 2
    agent.close.assert_called_once()


def test_orchestrator_does_not_claim_private_success_without_submission():
    """Проверяет, что пустой private-статус не означает завершение серии."""
    agent = Mock()
    agent.run_conversation.return_value = {"completed": True, "messages": ["history"]}
    journal = Mock()
    journal.status.return_value = status(0, True)
    with pytest.raises(RuntimeError, match="no journal progress"):
        run_role(lambda: agent, "submit", "orchestrator", journal, lambda: None)
    assert agent.close.call_count == 3


def test_failed_turn_closes_agent():
    """Проверяет освобождение ресурсов агента при ошибке хода."""
    agent = Mock()
    agent.run_conversation.return_value = {"completed": False, "failed": True}
    with pytest.raises(RuntimeError, match="turn failed"):
        run_role(lambda: agent, "review", "reviewer")
    agent.close.assert_called_once()


def test_completed_private_does_not_start_agent():
    """Проверяет повторный запуск после завершения private без обращения к модели."""
    factory = Mock()
    journal = Mock()
    journal.status.return_value = status(0, True)
    result = run_role(factory, "start", "orchestrator", journal, lambda: "complete")
    assert result["completed"]
    factory.assert_not_called()


def test_failed_private_does_not_start_agent():
    """Проверяет остановку при обнаружении завершившейся ошибкой private-серии."""
    factory = Mock()
    journal = Mock()
    journal.status.return_value = status(0, True)
    with pytest.raises(ValueError, match="failed"):
        run_role(factory, "start", "orchestrator", journal, lambda: "failed")
    factory.assert_not_called()


def test_human_stopped_finalized_pilot_is_complete():
    """Проверяет завершение пилота после операторской остановки и финализации."""
    factory = Mock()
    journal = Mock()
    journal.status.return_value = {**status(2, True), "stopped": True}
    assert run_role(factory, "start", "orchestrator", journal)["completed"]
    factory.assert_not_called()


def test_repetition_recovers_with_fresh_agent_and_existing_journal():
    """Восстанавливает оркестратор без повторного использования вырожденной истории."""
    failed, recovered = Mock(), Mock()
    failed.run_conversation.return_value = {
        "failed": True, "completed": False,
        "error": "Model output entered a repetition loop and was truncated mid-loop; "
                 "refusing to continue a degenerate response.",
        "messages": ["repeated text"],
    }
    recovered.run_conversation.return_value = {"completed": True, "final_response": "done"}
    factory = Mock(side_effect=[failed, recovered])
    journal = Mock()
    journal.status.side_effect = [status(), status(), status(0, True)]
    result = run_role(factory, "Original campaign boundaries", "orchestrator", journal)
    assert result["final_response"] == "done"
    prompt = recovered.run_conversation.call_args.args[0]
    assert "Original campaign boundaries" in prompt
    assert "canonical journal" in prompt
    assert "repeated text" not in prompt
    assert recovered.run_conversation.call_args.kwargs["conversation_history"] is None
    failed.close.assert_called_once()
    recovered.close.assert_called_once()


def test_repetition_recovery_is_bounded():
    """Ограничивает число новых попыток, если модель продолжает зацикливаться."""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run_conversation.return_value = {
            "failed": True, "error": "Model output entered a repetition loop",
        }
    factory = Mock(side_effect=agents)
    journal = Mock()
    journal.status.return_value = status()
    with pytest.raises(RuntimeError, match="repetition_loop"):
        run_role(factory, "start", "orchestrator", journal)
    assert factory.call_count == 3
    for agent in agents:
        agent.close.assert_called_once()


def test_unspecified_failure_is_not_retried():
    """Не скрывает неизвестные ошибки автоматическим перезапуском оркестратора."""
    agent = Mock()
    agent.run_conversation.return_value = {"failed": True, "error": "tool failure"}
    factory = Mock(return_value=agent)
    journal = Mock()
    journal.status.return_value = status()
    with pytest.raises(RuntimeError, match="unspecified_failure"):
        run_role(factory, "start", "orchestrator", journal)
    factory.assert_called_once()
    agent.close.assert_called_once()


def test_gateway_failure_prevents_repetition_recovery():
    """Сохраняет остановку при ошибке API, даже если результат похож на зацикливание."""
    agent = Mock()
    agent.run_conversation.return_value = {
        "failed": True, "error": "Model output entered a repetition loop",
    }
    factory = Mock(return_value=agent)
    journal = Mock()
    journal.status.return_value = status()
    with pytest.raises(RuntimeError, match="gateway unavailable"):
        run_role(factory, "start", "orchestrator", journal,
                 check=Mock(side_effect=RuntimeError("gateway unavailable")))
    factory.assert_called_once()
    agent.close.assert_called_once()


def test_idle_orchestrator_gets_concrete_research_instruction_and_fresh_context():
    """После двух бесплодных ходов восстанавливает поиск с исходными ограничениями."""
    idle, recovered = Mock(), Mock()
    idle.run_conversation.return_value = {
        "completed": True, "messages": [{"role": "assistant", "content": "baselines done"}],
        "final_response": "Baselines completed",
    }
    recovered.run_conversation.return_value = {"completed": True, "final_response": "done"}
    journal = Mock()
    journal.status.side_effect = [status(15), status(15), status(15), status(0, True)]
    observe = Mock()
    result = run_role(Mock(side_effect=[idle, recovered]), "Original boundaries", "orchestrator",
                      journal, observe=observe)
    assert result["final_response"] == "done"
    assert idle.run_conversation.call_count == 2
    assert "15 proposal slots remaining" in idle.run_conversation.call_args.args[0]
    prompt = recovered.run_conversation.call_args.args[0]
    assert "Original boundaries" in prompt
    assert "NOT complete" in prompt
    assert recovered.run_conversation.call_args.kwargs["conversation_history"] is None
    assert observe.call_count == 3
    idle.close.assert_called_once()
    recovered.close.assert_called_once()


def test_idle_recovery_stops_after_two_restarts():
    """Не превращает отсутствие прогресса в бесконечное расходование токенов."""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run_conversation.return_value = {"completed": True, "messages": ["history"]}
    factory = Mock(side_effect=agents)
    journal = Mock()
    journal.status.return_value = status(15)
    with pytest.raises(RuntimeError, match="after recovery limit"):
        run_role(factory, "start", "orchestrator", journal)
    assert factory.call_count == 3
    for agent in agents:
        assert agent.run_conversation.call_count == 2
        agent.close.assert_called_once()


def test_partial_response_is_diagnosed_without_automatic_retries():
    """Не выдаёт неполный ответ за успех и не запускает новый цикл продолжений."""
    agent = Mock()
    agent.run_conversation.return_value = {"completed": False, "partial": True,
                                           "final_response": "truncated summary"}
    factory = Mock(return_value=agent)
    journal = Mock()
    journal.status.return_value = status(15)
    observe = Mock()
    with pytest.raises(RuntimeError, match="partial_response"):
        run_role(factory, "start", "orchestrator", journal, observe=observe)
    factory.assert_called_once()
    observe.assert_called_once_with(agent.run_conversation.return_value)
    agent.close.assert_called_once()
