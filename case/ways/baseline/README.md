# Команды для baseline AR-12

Имена кандидатов — `matching` (словарь n-грамм), `boosting` (CatBoost), `transformer` (ONNX INT8). Все команды выполняются из корня репозитория.

Подготовьте окружение для всех кандидатов:

```bash
uv sync --locked --group boosting --group transformer
```

## Переобучение и оценка на train и public

Команды ниже перемешивают строки полного AR‑12 train с seed 42 и передают одну и ту же копию всем трём скриптам обучения. CatBoost и трансформер выделяют из неё 20% для подбора порога. Оператор последовательно измеряет каждый фильтр на исходных train и public. При пороге `1000` сообщений/с применяются те же гейты, что и в текущем пакете метрики AR‑12.

```bash
set -eu
train=case/dataset/ar12-train/ar12.train.jsonl
public=case/dataset/ar12-test/ar12.public.jsonl
results_root=case/ways/baseline/.results
mkdir -p "$results_root"
results=$(mktemp -d "$results_root/run-XXXXXX")
printf 'Results: %s\n' "$results"

shuffled_train="$results/ar12.train-shuffled.jsonl"
.venv/bin/python - "$train" "$shuffled_train" <<'PY'
from pathlib import Path
from random import Random
import sys

source, destination = map(Path, sys.argv[1:])
with source.open(encoding="utf-8") as stream:
    rows = [line.rstrip("\n") for line in stream if line.strip()]
Random(42).shuffle(rows)
with destination.open("x", encoding="utf-8") as stream:
    for row in rows:
        stream.write(row + "\n")
PY

for candidate in matching boosting transformer; do
  .venv/bin/python -m "case.ways.baseline.${candidate}.train" "$shuffled_train"
done

for candidate in matching boosting transformer; do
  .venv/bin/python -m case.ways.baseline.evaluate \
    "$candidate" "$train" --min-messages-per-second 1000 > "$results/$candidate.train.json"
  .venv/bin/python -m case.ways.baseline.evaluate \
    "$candidate" "$public" --min-messages-per-second 1000 > "$results/$candidate.public.json"
  printf 'Saved train and public reports for %s\n' "$candidate"
done
```

В каждом каталоге прогона сохраняются перемешанная копия train и шесть JSON-отчётов. Оценщик загружает модель один раз, прогревает её и считает TPR, FPR, p99, пропускную способность и итоговый балл. Подробные public-замеры предназначены для оператора. Каталог `.results/` исключён из Git.

## Запуск на одном сообщении

```bash
candidate=matching
printf '%s\n' '{"input":"What is photosynthesis?"}' |
  .venv/bin/python -m "case.ways.baseline.${candidate}.solve"
```

Решатель читает JSON со стандартного ввода и возвращает `{"answer":"0"}` или `{"answer":"1"}`.
