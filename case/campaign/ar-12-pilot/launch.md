# Запуск пилота AR-12

## Подготовка и приёмка

Команды выполняются из корня репозитория. Скрипт запуска и Dockerfile общие для AR-12 и лежат в `case/deploy/`. Настройки этого запуска хранятся в [runtime.json](runtime.json). Нужны Docker с локальным daemon, Python 3.12 и оба виртуальных окружения. У оператора должен быть доступ к Docker socket.

```bash
uv sync --locked --group boosting --group transformer
(cd arena && uv sync --locked)
pilot=case/campaign/ar-12-pilot
pilot_python=.venv/bin/python
"$pilot_python" case/deploy/arena.py --campaign "$pilot" \
  prepare-pilot --max-solutions 6
```

`prepare-pilot` — отдельный запуск подготовки. Укажите утверждённый лимит вместо примера `6`, если он отличается. Команда готовит локальный split, проверяет ранее замороженную метрику, использует имеющиеся стартовые пакеты и образ runner либо создаёт их, проводит синтетическую Docker-приёмку и создаёт пустой `.run/search.sqlite3`. Результат сохраняется в `.run/preparation.json`. Поиск и сервис кампании запускаются отдельно. Проверка OpenRouter выполняется отдельно одним синтетическим запросом.

### Ручная подготовка

```bash
uv sync --locked --group boosting --group transformer
(cd arena && uv sync --locked)
pilot=case/campaign/ar-12-pilot
pilot_python=.venv/bin/python
data=case/dataset/ar12-pilot-v1/data

test -d "$data" || "$pilot_python" case/dataset/prepare_split.py \
  case/dataset/ar12-train/ar12.train.jsonl case/dataset/ar12-pilot-v1/evaluation-ids.txt "$data"
test -d "$pilot/.run/candidates" || "$pilot_python" case/deploy/arena.py --campaign "$pilot" prepare-baselines
"$pilot_python" case/deploy/arena.py --campaign "$pilot" build-runner
```

`prepare-baselines` обучает три стартовых решения на `case/dataset/ar12-pilot-v1/data/train.jsonl` и сохраняет автономные пакеты в `.run/candidates/`. При наличии готовых пакетов этот шаг можно пропустить. `build-runner` сохраняет ID собранного образа.

После подготовки выполните [smoke-проверки Arena, Docker и OpenRouter](../../deploy/README.md) и запишите результаты в `preparation.md`.

## Поиск

После приёмки и утверждения бюджета оператор запускает оценщик в отдельном терминале. При ручной подготовке сначала создайте журнал. Пример использует предложенный лимит 6.

```bash
test -f "$pilot/.run/search.sqlite3" || "$pilot_python" case/deploy/arena.py \
  --campaign "$pilot" init-journal --max-solutions 6
"$pilot_python" case/deploy/arena.py --campaign "$pilot" serve
```

Оценщик слушает `127.0.0.1:8765`, создаёт токен отправки в `.run/submit-token` и оценивает на 256 зафиксированных примерах. Таймаут runner на пример — 30 секунд.

### Агентный поиск

Агентные роли запускаются через `case/agents/run.py`. Оркестратор загружает `arena/prompts/start.md` и `arena/prompts/orchestrator.md`, исследователь — `researcher.md`, независимый ревьюер — `reviewer.md`. Роли исполняются Hermes с доступом к рабочим файлам и инструментам. Настройки модели находятся в конфигурации кампании и выданном оператором файле доступа.

Внутри подготовленного ограниченного окружения интерфейс запуска имеет следующий вид:

```bash
python -m case.agents.run orchestrator   --campaign /workspace/campaign   --journal /search/search.sqlite3   --credentials-file /run/arena/model-access.json
```

Перед поиском требуется `.run/preparation.json` со статусом `ready`. Оператор запускает отдельный train-сервис и выдаёт его разрешённый адрес и submit-токен в рабочие материалы кампании. Доступ к обучающим данным требует внутреннего модельного endpoint согласно условиям пилота.

Оркестратор сам формирует направления и поручения исследователям, ведёт общий журнал и привлекает независимого ревьюера. Для вызова роли отдельно замените `orchestrator` на `researcher` или `reviewer` и передайте поручение через `--assignment`. Для основной кампании с непустым private добавьте `--private-config` с разрешёнными агенту URL, путём submit-токена, K и N. Оценщик сохраняет закрытые данные и итоговые результаты в контуре оператора.

### Контейнер агентов

Оператор собирает образ из корня репозитория:

```bash
docker buildx build --load -f case/deploy/agents.Dockerfile \
  -t ar12-hermes:configured .
```

Затем оператор подготавливает отдельную рабочую копию текущей кампании:

```bash
.venv/bin/python case/deploy/agents.py stage \
  --campaign case/campaign/ar-12-pilot \
  --train case/dataset/ar12-pilot-v1/data/train.jsonl \
  --evaluation case/dataset/ar12-pilot-v1/data/evaluation.arena.jsonl \
  --workspace /path/to/ar12-workspace
```

Для запуска нужны образ с закреплённым Hermes и адаптером Arena, модельный gateway, отдельный каталог общего журнала и выделенная сеть с разрешёнными маршрутами к API. Образ должен содержать зависимости обучения выбранных бейзлайнов. Адрес `127.0.0.1` внутри контейнера относится к самому контейнеру; оператор выдаёт достижимые из этой сети адреса сервисов. Сеть и доступ к gateway проверяются на синтетических данных до поиска.

```bash
.venv/bin/python case/deploy/agents.py run \
  --workspace /path/to/ar12-workspace --credentials /path/to/model-access.json \
  --journal-dir /path/to/canonical-search --image ar12-hermes:configured \
  --network ar12-research --role orchestrator \
  --train-url "$train_url" --train-token "$train_token"
```

Пути и имя образа в примере задаёт оператор. `model-access.json` содержит `model`, `model_base_url` и при необходимости `api_key` для внутреннего gateway. Train-сервис можно привязать к адресу операторского интерфейса сети через `serve --host`. Для кампании с private добавьте `--private-config --config /path/to/agent-access.json --private-token "$private_token"`; файл должен содержать `url`, `token_file`, `finalists`, `output_count`, а указанный token-файл должен быть доступен в контейнере. Права на Docker socket и закрытые файлы сохраняются у оценщика. `--print-command` показывает параметры контейнера без запуска.

### Команды исходного пилотного сценария

Следующие команды обслуживают короткие одношаговые предложения через OpenRouter и ручное применение патча. Агентный поиск использует промпты ролей и общий CLI Arena. Для ручного запуска проведите стартовые пакеты через общий журнал:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" run-baselines
```

Затем по очереди обработайте `matching`, `boosting` и `transformer`. Команда `research` требует успешного smoke и зарегистрированного родителя. Она отправляет описание задачи и два исходных файла базового решения, сохраняет гипотезу и патч, а также usage по ID кандидата.

```bash
candidate=matching
"$pilot_python" case/deploy/arena.py --campaign "$pilot" research "$candidate"
```

Предложение сохраняется в `.run/research/variant-<имя>.json`. Локальный исполнитель применяет патч в отдельной директории и передаёт итоговые `train.py` и `solve.py` независимому вызову модели через OpenRouter:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" review-variant "$candidate" \
  --source /path/to/reviewed-method
```

Ревьюер получает предложение и исходный код. Перед отправкой проверьте, что в них отсутствуют примеры датасета и секреты. Ревью сохраняет вердикт, замечания, риски и отдельный учёт токенов в `.run/reviews/variant-<имя>.json`. При `needs_information` уточните материалы и повторите ревью. При `reject` можно передать причину через `reject-variant`.

Для `approve` локальный исполнитель читает замечания и явно запускает обучение:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" run-variant "$candidate" \
  --reviewed-source /path/to/reviewed-method \
  --hypothesis 'Гипотеза варианта' --change 'Что изменено'
```

`run-variant` сверяет вердикт с предложением и точными байтами двух файлов перед новым обучением. Изменение любого из них требует повторного ревью. Команда обучает и упаковывает проверенный код, регистрирует родителя и затраты, отправляет пакет в train-оценщик и импортирует результат в журнал. При сбое обучения она регистрирует неудачное предложение. При прерывании ожидания повтор той же команды продолжает оценку без нового обучения и расхода места. Время подготовки включает время ревью. Оценку качества даёт последующий train-замер.

После исчерпания лимита оркестратор выбирает победителя по журналу и фиксирует выбор:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" finish-pilot \
  --winner variant-matching --reason 'Основание выбора по результатам пилота'
```

Победитель в команде приведён как пример. Итог сохраняется в `.run/final-selection.json`. Средние затраты по новым вариантам выгружаются [командой отчёта](report.md).
