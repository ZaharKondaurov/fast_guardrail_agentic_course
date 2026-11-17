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
    agent.close.assert_called_once()


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
