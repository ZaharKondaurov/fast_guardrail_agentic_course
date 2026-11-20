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

## Закрытая оценка AR-12

В пилоте AR-12 `private` пуст, поэтому его штатный поиск заканчивается локальным выбором по train. Для кампании с непустым `private` оператор заранее фиксирует `K`, `N`, лимит предложений и release ID. Подготовьте отдельный закрытый каталог оператора и Arena JSONL с полями `id`, `input` и `reference`. Для каждой серии сохраняйте один и тот же каталог состояния. Запускайте оценщик под учётной записью оператора с доступом к Docker и к каноническому журналу. Доступ к каталогу состояния, датасету и Docker socket остаётся в контуре оператора.

Из корня репозитория задайте значения переменных из условий кампании. `private_state` указывает на новый или существующий каталог серии вне репозитория. Для существующего каталога требуются права `0700`.

```bash
export PYTHONPATH="$PWD/arena"
arena_python=arena/.venv/bin/python

"$arena_python" -m case.deploy.private serve \
  --dataset "$private_dataset" --state "$private_state" \
  --metric "$frozen_metric" --runner-image "$runner_image" \
  --journal "$journal" --release-id "$release_id" \
  --max-solutions "$max_solutions" --finalists "$K" --winners "$N"
```

Сервис слушает `127.0.0.1:8766` и создаёт `$private_state/submit-token`. Перед поиском проверьте готовность через разрешённую сеть или туннель. Выдайте агенту URL и отдельную копию submit-токена.

```bash
"$arena_python" -m evaluator.client search-readiness \
  --url "$private_url" --token-file "$submit_token" --require-campaign \
  --max-solutions "$max_solutions" --finalists "$K" \
  --output-count "$N" --release-id "$release_id"
```

После завершения поиска и ревью выберите ровно `K` кандидатов. Передавайте по одному `--candidate` на каждого. Ниже пример для `K=2`.

```bash
"$arena_python" -m case.deploy.private freeze --journal "$journal" \
  --finalists "$K" --candidate variant-matching --candidate variant-boosting \
  --output finalists.json
"$arena_python" -m evaluator.client submit \
  --url "$private_url" --token-file "$submit_token" --bundle finalists.json
"$arena_python" -m evaluator.client status \
  --url "$private_url" --token-file "$submit_token"
```

`freeze` восстанавливает неизменяемые снимки из журнала и фиксирует пакет в Arena. Private принимает один пакет на серию. Повторная отправка идентичного пакета возвращает его статус. Итоговый `record.json` с баллами и победителями находится в каталоге состояния оценщика. Клиент получает только служебный статус. Если поиск досрочно остановлен оператором, передайте `--stop-evidence` по [протоколу Arena](../../arena/deploy/README.md).

Штатный runner поддерживает программную метрику качества, например замороженный пакет TPR/FPR из пилота для синтетической приёмки. Основная метрика `ar12-f1-v1` вычисляет F1 на штатном runner. Для диагностики p99 и пропускной способности нужен доверенный замер прогретого инференса. Без него поля скорости и соответствующие гейты остаются пустыми.

## Графики в артефактах

Экспортёр `python -m case.deploy.export_campaign --campaign <campaign>` сохраняет `lineage.svg`, отдельные `score-train.svg` и `score-public.svg`, а также точки и AUC в `results/score-curves.json`. При нескольких наборах сравнения имена графиков получают отдельные суффиксы; разные данные и условия не смешиваются. Для отрисовки используется matplotlib из корневого `.venv`, установленный вместе с группой `boosting`.

Ось X — порядковый номер предложения из журнала (`proposal`). Для maximize линия вычисляется как `best[i] = max(best[i-1], score[i])`, для minimize — через min. Это ретроспективная кривая: даже поздний результат относится к индексу своего кандидата, а не к числу предложений на момент завершения замера. Если у кандидата несколько успешных сопоставимых измерений, используется последнее по порядку завершения. Неоценённые и неуспешные предложения сохраняют прежний максимум; отсутствующая оценка не заменяется нулём.

AUC — ступенчатый интеграл по `[0, N]`: значение в индексе `i` действует на `[i, i+1)`, без линейной интерполяции; `mean_quality` — AUC/N. Для F1 до первого оценённого предложения явно принимается начальное значение 0. Для прочих метрик начальное качество не придумывается, и полный AUC может быть неизвестен. Этот график не показывает, когда оценки стали доступны оркестратору; AUC здесь не является ROC-AUC классификатора. Private-оценки в графики не включаются.

Обновить только графики в существующих артефактах:

```bash
PYTHONPATH=arena arena/.venv/bin/python -m case.deploy.score_curves \
  --campaign case/campaign/ar-12-first
```
