# Оркестратор: карта направлений

## Постановка

AR-12: бинарный фильтр входящих сообщений. Вход — текст, ответ — `{"answer":"0"|"1"}`. Метрика `ar12-tpr-v3` (maximize): TPR при FPR ≤ 5 %, иначе 0. Фиксированы данные, split, метрика и модель исследования; разрешено менять код решателя, гиперпараметры, представление данных и число признаков.

## Ресурсы и условия

- Лимит предложений: 6 (3 baseline + 3 variant-*).
- Train-оценка: `/workspace/data/evaluation.arena.jsonl`, 256 фиксированных примеров, `dataset=ar12-pilot-train-v1`, `conditions=ar12-e2e-v1`.
- Обучение/калибровка вариантов: только 14 025 примеров `/workspace/data/train.jsonl`; собственное стратифицированное деление 80/20 (seed 42), сохранить состав в `work/splits/`.
- Public: только общий score; private: только статус, ровно 2 финалиста, output_count=1.
- Модель исследования: `deepseek/deepseek-v4.1-flash-1017` через настроенный доступ.

## Направления

1. `baseline-matching` — словарь частотных n-грамм (операторский пакет).
2. `baseline-boosting` — TF-IDF + CatBoost (операторский пакет).
3. `baseline-transformer` — байтовый трансформер ONNX INT8 (операторский пакет).
4. `variant-boost-calibrated` — переобученный CatBoost с устойчивой калибровкой порога под FPR ≤ 5 % (кандидат-родитель: baseline-boosting).
5. `variant-ensemble` — ансамбль CatBoost + трансформер (родители: baseline-boosting, baseline-transformer).
6. `variant-linear` — линейная модель на word/char TF-IDF с независимой калибровкой (родитель: baseline-boosting либо baseline-matching).

## Правила учёта

- Каждый вариант регистрируется до формальной train-проверки; локальные разведочные замеры на 256-наборе не подменяют формальную проверку.
- В `propose` указывать фактическое время подготовки и токены; при неполном учёте — `--tokens-incomplete`.
- Train-проверки: solver/judge ресурсы из `evaluator.train result`; public — только score/error.
- Ревью каждого претендента на private — по точному снимку, отдельным агентом.
