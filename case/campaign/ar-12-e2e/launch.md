# Запуск сквозного теста AR-12

После `prepare-pilot` запустите `bash case/deploy/e2e.sh setup`. Скрипт создаст сеть `ar12-research` при необходимости, экспортирует закрытые выборки и сохранит путь к каталогу серии в `.run/closed-dir`. Затем в трёх отдельных терминалах выполните:

```bash
bash case/deploy/e2e.sh train
bash case/deploy/e2e.sh public
bash case/deploy/e2e.sh private
```

На Linux с локальным Docker Engine скрипт определяет адрес оценщиков по шлюзу Docker-сети. Если Docker работает в отдельной VM и шлюз недоступен на хосте, задайте `AR12_BIND_HOST` при запуске. Если закрытые выборки уже экспортированы, передайте `AR12_CLOSED_DIR=/путь/к/каталогу` при первом `setup`, чтобы продолжить ту же серию.

Когда оценщики запущены, добавьте `OPENROUTER_API_KEY` в корневой `.env` и выполните:

```bash
bash case/deploy/e2e-openrouter.sh
```

Скрипт создаёт временный файл доступа из `.env` и удаляет его после завершения. Если файл доступа уже подготовлен, его можно передать напрямую:

```bash
bash case/deploy/e2e.sh run /путь/к/model-access.json
```

Команда проверяет три оценщика и делает один синтетический запрос к модели с проверкой учёта токенов. Затем она создаёт или переиспользует рабочую копию и запускает Hermes. Если адрес оценщиков внутри контейнера отличается от адреса привязки на хосте, задайте `AR12_EVAL_HOST` (например, `host.docker.internal`).

## Ручные команды

Выполняйте команды из корня репозитория. Соберите образ Hermes и настройте доступ к OpenRouter и выделенную сеть по [инструкции пилота](../ar-12-pilot/launch.md). Адреса оценщиков должны быть доступны из сети контейнера. Храните закрытое состояние вне рабочей копии агента.

```bash
campaign=case/campaign/ar-12-e2e
data=case/dataset/ar12-pilot-v1/data
export PYTHONPATH="$PWD/arena"
arena_python=arena/.venv/bin/python
.venv/bin/python case/deploy/arena.py --campaign "$campaign" prepare-pilot --max-solutions 6
journal="$campaign/.run/search.sqlite3"
runner_image=$(cat "$campaign/.run/runner-image.txt")

umask 077
closed=$(mktemp -d /tmp/ar12-e2e.XXXXXX)
for split in public private; do
  .venv/bin/python case/dataset/export_arena.py \
    "case/dataset/ar12-test/ar12.$split.jsonl" "$closed/$split.jsonl"
done
```

Сохраните путь `closed` для продолжения этой же серии. В отдельных терминалах запустите три сервиса. `bind_host` — адрес операторского интерфейса выделенной сети.

```bash
.venv/bin/python case/deploy/arena.py --campaign "$campaign" serve --host "$bind_host"
```

```bash
# Public, отдельный процесс
"$arena_python" -m case.deploy.private serve --split public --port 8767 --host "$bind_host" \
  --dataset "$closed/public.jsonl" --state "$closed/public" --metric "$campaign/metric" \
  --runner-image "$runner_image" --journal "$journal" --release-id ar12-e2e-v1 \
  --max-solutions 6 --finalists 2 --winners 1
```

```bash
# Private, отдельный процесс
"$arena_python" -m case.deploy.private serve --port 8766 --host "$bind_host" \
  --dataset "$closed/private.jsonl" --state "$closed/private" --metric "$campaign/metric" \
  --runner-image "$runner_image" --journal "$journal" --release-id ar12-e2e-v1 \
  --max-solutions 6 --finalists 2 --winners 1
```

В файле `private_config` укажите `url` закрытого сервиса, `token_file` со значением `/run/arena/private-token`, `finalists` со значением 2, `output_count` со значением 1. `credentials` — файл доступа к OpenRouter. `workspace` — новый каталог агента. Укажите доступные из контейнера адреса `train_url` и `public_url`.

```bash
.venv/bin/python case/deploy/agents.py stage --campaign "$campaign" \
  --train "$data/train.jsonl" --evaluation "$data/evaluation.arena.jsonl" --workspace "$workspace"
.venv/bin/python case/deploy/agents.py run --workspace "$workspace" \
  --credentials "$credentials" --journal-dir "$campaign/.run" \
  --image ar12-hermes:configured --network ar12-research \
  --train-url "$train_url" --train-token "$campaign/.run/submit-token" \
  --public-url "$public_url" --public-token "$closed/public/submit-token" \
  --private-config --config "$private_config" --private-token "$closed/private/submit-token"
```

Успешный сквозной тест оставляет шесть предложений в журнале, замеры train и public и завершённый private-пакет из двух снимков. Оператор проверяет одного победителя в `$closed/private/*/record.json`. Для средних затрат укажите журнал этой кампании в [команде отчёта](../ar-12-pilot/report.md). Отчёт отдельно показывает подготовку и время завершённых train/public-замеров. Для среднего по токенам учитываются только предложения с полным учётом.
