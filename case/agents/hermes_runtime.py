"""Hermes transport adapter for the AR-12 model gateway."""

import json
import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any
from time import monotonic
from uuid import uuid4

from case.agents.usage import ROLE_HEADER, agent_label

CONFIG_BASE_URL = "http://arena-gateway.invalid/v1"
REQUEST_TIMEOUT = 131460
RUNNER_TIMEOUT = 604800


def gateway_message(message: dict[str, Any]) -> dict[str, Any]:
    """Оставляет поля сообщения, которые понимает модельный gateway."""
    result = {key: message[key] for key in (
        "role", "content", "tool_calls", "tool_call_id", "reasoning_content"
    ) if key in message}
    if isinstance(result.get("tool_calls"), list):
        result["tool_calls"] = [{
            "id": call["id"],
            "type": call["type"],
            "function": {key: call["function"][key] for key in ("name", "arguments")},
        } for call in result["tool_calls"]]
    return result


def gateway_request(request: dict[str, Any], model: str) -> dict[str, Any]:
    """Нормализует запрос Hermes и закрепляет выбранную для кампании модель."""
    extra = request.get("extra_body")
    if isinstance(extra, dict):
        request = request | extra
    result = {
        key: request[key]
        for key in ("messages", "tools", "tool_choice")
        if key in request
    }
    if "messages" in result:
        result["messages"] = [gateway_message(message) for message in result["messages"]]
    # Сохраняем бюджет, выбранный Hermes: без него провайдер использует собственный лимит.
    for key in ("max_tokens", "max_completion_tokens"):
        value = request.get(key)
        if type(value) is int and value > 0:
            result[key] = value
    # Hermes временно отключает reasoning после пустого обрезанного ответа.
    # SDK принимает OpenRouter reasoning через extra_body, а не отдельный аргумент.
    if isinstance(request.get("reasoning_effort"), str):
        result["reasoning_effort"] = request["reasoning_effort"]
    if isinstance(request.get("reasoning"), dict):
        result["extra_body"] = {"reasoning": request["reasoning"]}
    result.update(model=model, stream=False)
    return result


class ModelState:
    def __init__(self) -> None:
        """Создаёт потокобезопасный флаг ошибки для остановки всех модельных вызовов."""
        self.failed = threading.Event()
        self.failure = None

    def check(self) -> None:
        """Блокирует дальнейшие запросы после ошибки модельного gateway."""
        if self.failed.is_set():
            raise RuntimeError(self.failure or "model gateway failed; further requests disabled")

    def fail(self, error: Exception) -> None:
        """Сохраняет безопасную диагностику без URL, ключа и содержимого запроса."""
        status = getattr(error, "status_code", None)
        message = f"Model request failed ({type(error).__name__}"
        if type(status) is int:
            message += f", HTTP {status}"
        cause = error.__cause__
        if cause is not None:
            message += f", cause {type(cause).__name__}"
        self.failure = message + "); further requests disabled"
        self.failed.set()


def prepare_home(home: Path, model: str) -> None:
    """Создаёт конфигурацию Hermes для заданной модели и длительных задач."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        Path(__file__).with_name("hermes.yaml").read_text(encoding="utf-8").replace(
            "__CAMPAIGN_MODEL__", json.dumps(model)
        ), encoding="utf-8"
    )
    os.environ.update({
        "HERMES_HOME": str(home),
        "HERMES_SINGLE_QUERY_SESSION": "1",
        "OPENAI_BASE_URL": CONFIG_BASE_URL,
        "OPENAI_API_KEY": "arena-runtime-injects-credentials",
        "OPENAI_MODEL": model,
        "HERMES_API_TIMEOUT": str(REQUEST_TIMEOUT),
        "HERMES_API_CALL_STALE_TIMEOUT": str(REQUEST_TIMEOUT),
        "TERMINAL_TIMEOUT": str(RUNNER_TIMEOUT),
        "TERMINAL_MAX_FOREGROUND_TIMEOUT": str(RUNNER_TIMEOUT),
        "TERMINAL_LIFETIME_SECONDS": str(RUNNER_TIMEOUT),
    })


def install(base_url: str, model: str, timeout: float, api_key: str = "arena", *, usage_log=None) -> ModelState:
    """Подключает транспорт Hermes к заданному gateway и включает учёт вызовов."""
    import httpx
    import openai
    from openai.resources.chat.completions import AsyncCompletions, Completions

    state = ModelState()
    original_init = openai.OpenAI.__init__
    original_async_init = openai.AsyncOpenAI.__init__
    original_create = Completions.create
    original_async_create = AsyncCompletions.create

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        """Настраивает синхронный SDK-клиент на заданный маршрут без скрытых повторов."""
        kwargs.update(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
        original_init(self, *args, **kwargs)

    @wraps(original_async_init)
    def initialize_async(self, *args, **kwargs):
        """Настраивает асинхронный SDK-клиент с теми же параметрами маршрута."""
        kwargs.update(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        kwargs["http_client"] = httpx.AsyncClient(trust_env=False, timeout=timeout)
        original_async_init(self, *args, **kwargs)

    @wraps(original_create)
    def create(self, *args, **kwargs):
        """Отправляет синхронный запрос и записывает полученный расход или отсутствие учёта."""
        state.check()
        started = monotonic()
        response = None
        wire = gateway_request(kwargs, model)
        try:
            response = original_create(self, *args, **wire)
            return response
        except Exception as error:
            state.fail(error)
            state.check()
        finally:
            if usage_log is not None:
                usage_log.write("model", started, response, request=wire)

    @wraps(original_async_create)
    async def create_async(self, *args, **kwargs):
        """Отправляет асинхронный запрос и учитывает его в журнале текущей роли."""
        state.check()
        started = monotonic()
        response = None
        wire = gateway_request(kwargs, model)
        try:
            response = await original_async_create(self, *args, **wire)
            return response
        except Exception as error:
            state.fail(error)
            state.check()
        finally:
            if usage_log is not None:
                usage_log.write("model", started, response, request=wire)

    # Оба клиента SDK используют один маршрут, включая вызовы делегированных задач.
    openai.OpenAI.__init__ = initialize
    openai.AsyncOpenAI.__init__ = initialize_async
    Completions.create = create
    AsyncCompletions.create = create_async

    import tools.async_delegation as async_delegate
    import tools.delegate_tool as delegate
    from run_agent import AIAgent

    if usage_log is not None:
        original_run = AIAgent.run_conversation

        @wraps(original_run)
        def track_turn(self, *args, **kwargs):
            """Устанавливает роль и задачу на время хода для атрибуции вложенных запросов."""
            prompt = args[0] if args else kwargs.get("user_input", "")
            label = getattr(self, "_ar12_usage_label", None)
            if label is None:
                match = ROLE_HEADER.match(prompt) if isinstance(prompt, str) else None
                label = match.groups() if match else ("unattributed", None)
                self._ar12_usage_label = label
            # Метка действует только в контексте текущего хода и сбрасывается при выходе.
            token = agent_label.set(label)
            started = monotonic()
            turn_id = uuid4().hex
            status = "failed"
            try:
                usage_log.write("turn_start", started, turn_id=turn_id, status="running")
                result = original_run(self, *args, **kwargs)
                if result.get("completed") and not result.get("failed"):
                    status = "complete"
                return result
            finally:
                try:
                    usage_log.write("turn", started, turn_id=turn_id, status=status)
                finally:
                    agent_label.reset(token)

        AIAgent.run_conversation = track_turn

    async_delegate._STALE_IDLE_SECONDS = RUNNER_TIMEOUT
    async_delegate._STALE_IN_TOOL_SECONDS = RUNNER_TIMEOUT
    delegate._HEARTBEAT_STALE_CYCLES_IDLE = RUNNER_TIMEOUT // 30
    delegate._HEARTBEAT_STALE_CYCLES_IN_TOOL = RUNNER_TIMEOUT // 30
    AIAgent._try_recover_primary_transport = lambda *args, **kwargs: False
    AIAgent._dump_api_request_debug = lambda *args, **kwargs: None
    return state


def credentials(path: Path) -> dict[str, Any]:
    """Загружает файл доступа и проверяет наличие адреса модельного API."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("model_base_url"), str):
        raise ValueError("invalid credentials")
    return data
