# Запуск пилота AR-12

## Подготовка и приёмка

Команды выполняются из корня репозитория. Скрипт запуска и Dockerfile общие для AR-12 и лежат в `case/deploy/`. Настройки этого запуска хранятся в [runtime.json](runtime.json). Нужны Docker с локальным daemon, Python 3.12 и оба виртуальных окружения. У оператора должен быть доступ к Docker socket.

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

После приёмки и утверждения бюджета оператор один раз создаёт журнал, затем запускает оценщик в отдельном терминале. Пример использует предложенный лимит 6.

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" init-journal --max-solutions 6
"$pilot_python" case/deploy/arena.py --campaign "$pilot" serve
```

Оценщик слушает `127.0.0.1:8765`, создаёт токен отправки в `.run/submit-token` и оценивает на 256 зафиксированных примерах. Таймаут runner на пример — 30 секунд. Поиск запускается отдельно по [start.md](../../../arena/prompts/start.md) с путём `case/campaign/ar-12-pilot`. В отдельном терминале оркестратор или локальный исполнитель проводит стартовые пакеты через общий журнал:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" run-baselines
```

Затем по очереди обработайте `matching`, `boosting` и `transformer`. Команда `research` требует успешного smoke и зарегистрированного родителя. Она отправляет описание задачи и два исходных файла базового решения, сохраняет гипотезу и патч, а также usage по ID кандидата.

```bash
candidate=matching
"$pilot_python" case/deploy/arena.py --campaign "$pilot" research "$candidate"
```

Предложение сохраняется в `.run/research/variant-<имя>.json`. Ревьюер проверяет патч. Локальный исполнитель сохраняет согласованные `train.py` и `solve.py` в отдельной директории. Затем явно передаёт эту директорию следующей команде:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" run-variant "$candidate" \
  --reviewed-source /path/to/reviewed-method \
  --hypothesis 'Гипотеза варианта' --change 'Что изменено'
```

`run-variant` обучает и упаковывает проверенный код, регистрирует родителя и затраты, отправляет пакет в train-оценщик и импортирует результат в журнал. Если обучение упало, команда регистрирует неудачное предложение и сохраняет статус подготовки. Если ожидание оценки прервалось, повтор той же команды продолжает ожидание без нового обучения и расхода места. Время подготовки измеряется от начала запроса DeepSeek до готовности пакета, включая ожидание ручного ревью. Патч применяется локально после ревью.

После исчерпания лимита оркестратор выбирает победителя по журналу и фиксирует выбор:

```bash
"$pilot_python" case/deploy/arena.py --campaign "$pilot" finish-pilot \
  --winner variant-matching --reason 'Основание выбора по результатам пилота'
```

Победитель в команде приведён как пример. Итог сохраняется в `.run/final-selection.json`. Средние затраты по новым вариантам выгружаются [командой отчёта](report.md).
