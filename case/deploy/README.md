# Smoke-проверки AR-12

Выполняйте команды в Bash из корня репозитория. Для проверок нужны окружения `.venv` и `arena/.venv`. Docker-приёмка использует локальный Docker daemon, собранный образ runner и стартовые пакеты кандидатов. Команды подготовки приведены в [инструкции пилота](../campaign/ar-12-pilot/launch.md#подготовка-и-приёмка).

## 1. Ядро Arena

```bash
(cd arena && timeout 60 .venv/bin/python -m evaluator.smoke)
```

Успех — код выхода 0 и строка `smoke passed`. Штатная проверка охватывает программную и модельную метрики с подставными ответами, видимость public-score и запечатанную private-серию.

## 2. Метрика и сквозная Docker-приёмка

```bash
campaign=case/campaign/ar-12-pilot

PYTHONPATH=arena arena/.venv/bin/python -m evaluator.prepare verify \
  --metric "$campaign/metric"
.venv/bin/python -m pytest -q case/metric/tests

.venv/bin/python case/deploy/arena.py --campaign "$campaign" accept
```

`accept` запускает временный HTTP-оценщик на синтетических данных. Контрольный решатель, три baseline и тестовый `variant-matching` должны пройти оценку с записью в журнал. Команда выводит путь к новому `.run/acceptance-*/journal.json`. В том же каталоге находятся `service.log`, результаты проверок и `final-selection.json`. Токены исследователя в этом журнале подставные. Нулевые токены решателя и оценщика ожидаемы, поскольку они работают без LLM.

### Дополнительные штатные проверки Docker runner

Проверяют изоляцию немодельного решателя, исполнение восстановленного пакета и большие входы на том же образе AR-12.

```bash
PYTHONPATH=arena arena/.venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy

checks = runpy.run_path("arena/tests/docker_smoke.py")
image = Path("case/campaign/ar-12-pilot/.run/runner-image.txt").read_text().strip()
with TemporaryDirectory(prefix="ar12-runner-smoke-") as directory:
    for name in ("check_program", "check_restored_program", "check_large_inputs"):
        checks[name](image, Path(directory))
        print(f"{name}: passed")
PY
```

## 3. Реальный OpenRouter smoke

Нужен обычный API-ключ OpenRouter для запросов к модели. Если он записан в локальном `.env` в формате Bash, выполните:

```bash
(
  set -a
  source .env
  set +a
  .venv/bin/python case/deploy/arena.py \
    --campaign case/campaign/ar-12-pilot openrouter-smoke
)
```

Команда отправляет один синтетический запрос к модели из `runtime.json`. При успехе в `case/campaign/ar-12-pilot/.run/openrouter-smoke.json` сохраняются `model`, `response_id`, положительный `total_tokens` и `seconds`.

После проверок запишите дату, результаты и пути к свидетельствам в `preparation.md` кампании. Smoke использует отдельный синтетический журнал и не расходует места в бюджете поиска.
