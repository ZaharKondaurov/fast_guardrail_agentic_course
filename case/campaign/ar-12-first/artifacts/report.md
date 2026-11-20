# Отчёт кампании AR-12 (ar-12-first)

## Постановка

Бинарный гардрейл входящих сообщений: ответ `0` — пропустить, `1` — заблокировать. Метрика `ar12-f1-v1` (maximize) считает чистый F1 класса «заблокировать» на замороженном пакете `metric/`; TPR, FPR, recall, precision, p99, throughput и гейты — диагностика. Данные: 11 420 train-примеров для обучения, закреплённая 256-подвыборка train для официального train-замера Arena, 1 114 public и 1 123 private у оператора. Решатель и метрика немодельные (LLM не используется); таймаут runner — 30 с на пример.

## Схема валидации

- Обучение и настройка — только на 11 420 train-строках.
- 12% стратифицированный holdout train (seed 42, 1 371 строка) — пороги и веса ансамблей; 5-fold stratified OOF (seed 42) — пороги линейных и CatBoost-моделей.
- 256-подвыборка (`evaluation.arena.jsonl`, 128/128) не использовалась для обучения, подбора признаков или калибровки; только официальный train-замер Arena через `evaluator.train`.
- Сопоставимый ключ train: `split=train`, `dataset=ar12-deduplicated-v2-train-holdout-v1`, `subset` = digest 256 ID, `conditions=ar12-first-f1-v1`.
- Public: один общий score на кандидата; замеры записаны в журнал с `split=public`, `dataset=ar12-deduplicated-v2-public`, `subset=ar12-public-full-v1`. Private агенту не раскрывается.

## Предложения и train-диагностика (256 примеров)

Все 18 предложений зарегистрированы до проверки; лимит исчерпан. F1 — официальный train-score Arena; p99/throughput равны `null`, потому что штатный runner не передаёт доверенное `inference_ns`, а Docker wall time не подменяет время инференса.

| # | ID | F1 | TPR | FPR | Recall | Precision | p99 | Throughput | Гейты |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | matching (baseline) | 0.86275 | 0.8594 | 0.1328 | 0.8594 | 0.8661 | null | null | fpr fail; p99/throughput null |
| 2 | boosting (baseline) | 0.96525 | 0.9766 | 0.0469 | 0.9766 | 0.9542 | null | null | fpr pass; p99/throughput null |
| 3 | transformer (baseline) | 0.95686 | 0.9531 | 0.0391 | 0.9531 | 0.9606 | null | null | fpr pass; p99/throughput null |
| 4 | variant-lin-wordchar | 0.98824 | 0.9844 | 0.0078 | 0.9844 | 0.9921 | null | null | fpr pass |
| 5 | variant-lin-char | 0.99219 | 0.9922 | 0.0078 | 0.9922 | 0.9922 | null | null | fpr pass |
| 6 | variant-lin-robust | 0.99219 | 0.9922 | 0.0078 | 0.9922 | 0.9922 | null | null | fpr pass |
| 7 | variant-svc | 0.99219 | 0.9922 | 0.0078 | 0.9922 | 0.9922 | null | null | fpr pass |
| 8 | variant-cat-tuned | 0.98425 | 0.9766 | 0.0078 | 0.9766 | 0.9921 | null | null | fpr pass |
| 9 | variant-cat-feat | 0.98824 | 0.9844 | 0.0078 | 0.9844 | 0.9921 | null | null | fpr pass |
| 10 | variant-ens-lc | 0.99219 | 0.9922 | 0.0078 | 0.9922 | 0.9922 | null | null | fpr pass |
| 11 | variant-trans-192 | 0.95276 | 0.9453 | 0.0391 | 0.9453 | 0.9603 | null | null | fpr pass |
| 12 | variant-trans-wide | 0.99213 | 0.9844 | 0.0000 | 0.9844 | 1.0000 | null | null | fpr pass |
| 13 | variant-ens-ct | 0.99608 | 0.9922 | 0.0000 | 0.9922 | 1.0000 | null | null | fpr pass |
| 14 | variant-ens-lct | 0.99608 | 0.9922 | 0.0000 | 0.9922 | 1.0000 | null | null | fpr pass |
| 15 | variant-ens-all4 | 0.99608 | 0.9922 | 0.0000 | 0.9922 | 1.0000 | null | null | fpr pass |
| 16 | variant-ens-rct | 0.99608 | 0.9922 | 0.0000 | 0.9922 | 1.0000 | null | null | fpr pass |
| 17 | variant-trans-wide-full | 0.99611 | 1.0000 | 0.0078 | 1.0000 | 0.9922 | null | null | fpr pass |
| 18 | variant-ens-lct-full | 0.99611 | 1.0000 | 0.0078 | 1.0000 | 0.9922 | null | null | fpr pass |

Train-256 почти насыщен: 0.9961 — один неверный пример из 256. Различия между верхними кандидатами находятся в пределах шума.

## Public-сравнение (1114 примеров)

| ID | Public F1 |
| --- | --- |
| variant-ens-lct-full | 0.95755 |
| variant-ens-ct | 0.95737 |
| variant-ens-lct | 0.95737 |
| variant-lin-char | 0.95714 |
| variant-trans-wide-full | 0.95524 |

Public заметно труднее train-256: все кандидаты теряют ~0.035–0.04 F1. Это главный сигнал переноса; private может отличаться и от public.

## Финалисты

Заморожены ровно 3 кандидата (см. `selection.md`):

1. `variant-ens-lct-full` — 0.2 linear wordchar + 0.4 CatBoost + 0.4 byte+word transformer (full train); лучший public 0.95755, train 0.99611. Ревью: допустим.
2. `variant-lin-char` — char-only 2–6 TF-IDF + logistic regression; public 0.95714, train 0.99219; независимый символьный механизм, быстрый. Ревью: допустим.
3. `variant-trans-wide-full` — byte+hashed-word Transformer INT8, обучен на всех 11 420; public 0.95524, train 0.99611 (TPR 1.0, 1 FP); независимый механизм. Ревью: допустим.

Bundle digest: `c0cbb0c6a3ae71058aa290ba429b3eadc182aeb52a14903063344b972eacb133`; `finalize` записан в 2026-11-18T20:03:02Z. Private-пакет отправлен один раз в 2026-11-18T20:03Z, серия принята и завершена со статусом `complete` (2026-11-18T21:25Z). Агенту доступен только статус; private-оценки, ранжирование и выбор оператора не раскрываются и в отчёт не включены.

## Ресурсы

- Подготовка предложений: 12 878 с измеренного времени; 35 089 525 токенов с флагом `tokens_incomplete=true` (часть расхода атрибутирована аллокацией между кандидатами одной исследовательской серии; нулевой полный расход не заявлялся).
- Решатель: 0 токенов (все кандидаты немодельные, полный учёт), 5 881.8 с суммарного времени на 256-примерных train-замерах.
- Оценщик: 0 токенов, 510.4 с суммарного времени агрегатора. Public/private ресурсы остаются в закрытом контуре.

## Отрицательные результаты и ограничения

- Частотный словарь `matching` не проходит гейт FPR (0.1328) и даёт F1 0.8627 — слабейший baseline.
- `variant-trans-192` (0.9528) проиграл `variant-trans-wide` (0.9921): удлинение контекста без word-канала не помогло.
- CatBoost-семейство на 256 уступает линейным моделям (0.984–0.988 против 0.992), хотя добавляет разнообразие в ансамблях.
- Крупные бленды (7 моделей, stacking) на общем holdout не превзошли простой 3-компонентный бленд; лучший holdout F1 0.9962 у 4-компонентного варианта в пределах шума.
- Public-разрыв и near-duplicate кластеры train ограничивают перенос; пороги переносились с train-holdout/OOF, а не калибровались на private.
- Train-256 не различает верхние кандидаты; public-1114 и разнообразие механизмов использованы как основные критерии отбора.

## Артефакты

- Журнал: `/search/search.sqlite3` (18 предложений, 18 train + 5 public замеров, `finalize`).
- Результаты: `results/<id>.train.json`, `results/<id>.diagnostics.json`, `results/public-*.status.json`.
- Ревью: `board/reviews/<id>.md`.
- Отбор: `selection.md`; условия и состояние: `campaign.md`; гипотезы/находки: `board/claims/`, `board/findings/`.
