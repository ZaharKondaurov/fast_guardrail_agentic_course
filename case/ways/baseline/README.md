# Команды для baseline AR-12

Имена кандидатов — `matching` (словарь n-грамм), `boosting` (CatBoost), `transformer` (ONNX INT8). Все команды выполняются из корня репозитория.

Подготовьте окружение для всех кандидатов:

```bash
uv sync --locked --group boosting --group transformer
```

Выберите кандидата и пути к исходным размеченным JSONL. В примере для локальной проверки оценка выполняется на train.

```bash
candidate=matching
train=case/dataset/ar12-train/ar12.train.jsonl
evaluation_data="$train"

uv run --no-sync python -m "case.ways.baseline.${candidate}.train" "$train"
printf '%s\n' '{"input":"What is photosynthesis?"}' | uv run --no-sync python -m "case.ways.baseline.${candidate}.solve"
uv run --no-sync python -m case.ways.baseline.evaluate "$candidate" "$evaluation_data" --min-messages-per-second 1000
```

Решатель читает JSON со стандартного ввода и возвращает `{"answer":"0"}` или `{"answer":"1"}`. Оценщик загружает модель один раз, прогревает её и считает TPR, FPR, p99, пропускную способность и итоговый балл. Укажите в `--min-messages-per-second` порог кампании. Закрытые разделы оценивает оператор.
