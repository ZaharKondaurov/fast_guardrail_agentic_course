# Ревью: variant-lin-char

- **Кандидат:** `variant-lin-char`
- **Digest:** `4f4e39ef5b51f347b1e02a017a23bb45066b25352eae510e3a5c7e6bbe23ff3c` (digest_version 2)
- **Снимок:** `/workspace/campaign/work/reviews/variant-lin-char` (restore из `/search/search.sqlite3`)
- **Родитель:** `variant-lin-wordchar`; proposal 5 из 18
- **Автор:** researcher-a (`/workspace/campaign/board/claims/researcher-a.md`, `/workspace/campaign/board/findings/researcher-a.md`)
- **Ревьюер:** независимый reviewer (не автор кандидата)
- **Дата ревью:** 2026-11-18

## Проверенные материалы

- Снимок целиком: `solve.py`, `features.py`, `meta.json`, `model.npz`, `vocab.txt` (6 файлов с `__pycache__`, digest сходится побайтово).
- `board/claims/researcher-a.md`, `board/findings/researcher-a.md`.
- `campaign.md`, `common/research-protocol.md`, `common/public-evaluation.md`, `common/search-history.md`.
- Журнал: `/search/search.sqlite3` (`export`, `status`), запись `train-lin-char-1`.
- Официальный замер: `/workspace/campaign/results/variant-lin-char.train.json`; диагностика: `/workspace/campaign/results/variant-lin-char.diagnostics.json`.
- Train-артефакты порога: `/workspace/campaign/work/researcher-a/exp/runs/char26c/{lr128.npy,labels.npy,ids.json,report.json}`.

## Условия и свидетельства

1. **Область изменений.** Изменены только признаки, гиперпараметры и порог (char 2–6 TF-IDF, LogisticRegression C=128, порог 1.1021). Метрика `ar12-f1-v1`, split, данные, runner не менялись; `conditions_id=ar12-first-f1-v1`, `dataset_id=ar12-deduplicated-v2-train-holdout-v1`, `subset_id=8fc2…`. Это разрешено п. «Разрешённые изменения» `campaign.md`.
2. **Воспроизводимость снимка.** Digest восстановленного каталога, посчитанный по `evaluator.contracts.candidate_digest` (все файлы + sha256), совпал с зарегистрированным `4f4e39ef…`. Побайтовые sha256 файлов совпадают с манифестом researcher-a (`solve.py a508d9e8…`, `features.py 4cfe1c80…`, `meta.json 04ed43c0…`, `model.npz cac354d4…`, `vocab.txt aa23ca77…`). `solve.py` читает только файлы своего каталога, зависимость — NumPy; версия sklearn при обучении 1.9.1, python 3.12 зафиксированы в `meta.json`; runner image `sha256:57d8eb19…` зафиксирован в `preparation.json` и `params.harness_version`.
3. **Провенанс порога.** `threshold_score = 1.1021318582525157` — ровно argmax class-balanced OOF F1, пересчитанный по сохранённым OOF-скорам `lr128.npy`/`labels.npy` (11420 строк, 5-fold, seed 42): balanced F1 `0.9918333480970888`, plain F1 `0.994042891183479`, tp/fp/fn `7509/35/55` — совпадает с `meta.json` и findings. 256-набор в выборе порога не участвовал: оптимум на 256 дал бы другую точку, а 256-замер сохранён один раз после заморозки.
4. **Отсутствие зашитых ответов.** В коде и материалах снимка нет ID, lookup-таблиц, узнавания тестовых формулировок или ветвлений по конкретному тексту: только детерминированные char-ngram-признаки, словарь из train и линейные веса. `features.py` содержит общий код нормализации/deobfuscation и таблицы гомоглифов/leet, но активна только одна char-вьюха (`CONFIG`), таблицы не привязаны к тестовым примерам. Совпадение n-грамм с train — разрешённое обучение признаков.
5. **Один ответ на пример, без выбора лучшего.** `solve.py` детерминирован, порог один; официальный runner сохранил по одному ответу на пример. Я независимо воспроизвёл официальный train-замер чистым Python (без NumPy): все **256/256** ответов совпали с `variant-lin-char.train.json`, F1 `0.9921875`, tp/fp/tn/fn `127/1/127/1`; ошибки те же, что в findings (`neuralchemy-train-2325` FP, `toxicchat-train-21` FN). 256 ID записи в точности равны ID `evaluation.arena.jsonl`.
6. **Ресурсы.** Официальный train-замер: solver `252.92 s`, `0` токенов, `tokens_complete=true`; judge `29.99 s`, `0` токенов, `tokens_complete=true` (метрика программная, `uses_model=false` — нулевые токены подтверждены). `preparation` в записи `null`; расходы подготовки учтены в `propose`: 3 103 339 токенов с флагом `tokens_complete=false` (измеренная часть, неполнота отмечена). Это соответствует правилу «немодельный решатель — ноль токенов; неизвестное не выдаётся за полный ноль».
7. **Место в кампании.** Замер сопоставим с другими кандидатами на том же `subset_id 8fc2…`; это официальный train-замер на 256, а не public/private. Локальная диагностика researcher-a (`work/researcher-a/…/eval256.json`) отделена от официального результата и с ним совпадает.

## Вывод о допустимости

**допустим**

Нарушений области изменений, зашитых ответов, узнавания тестовых ID/формулировок, изменений метрики/split или утечки 256 в обучение/калибровку не найдено. Снимок, digest, родители, параметры и провенанс порога подтверждены независимо.

## Качество и риск переобучения (отдельно)

- Сильное evidence: официальный 256 F1 `0.99219` (2 ошибки), OOF balanced F1 `0.99183`; пакет 4.5 МБ, ~1 с/пример при лимите 30 с.
- **Риск переобучения:** train содержит кластеры почти-дубликатов; random-fold OOF и 256 из того же пула завышают оценку для действительно новых формулировок. Из ~20 OOF-конфигураций разница топ-6 — 2–8 ошибок из 11420, т.е. выбор C=128 на краю тренда статистически не отделён от C=64.
- Char-only не использует идентичность слов: OOF-ошибки на `toxic` 8.3%, на plain-text `control` 6/37; часть «семантических» токсичных промптов неотличима от benign по поверхности.
- Порог выбран под баланс 50/50, объявленный для 256; если private имеет другую распространённость, оптимум F1 сместится (риск качества, не нарушение).
- Пустой вход даёт score `4.53 > 1.10` → ответ 1 (крайний случай упаковки, в train/256 не наблюдался).
- 256 — не независимая валидация: это выделенная подвыборка train, а не public/private.

## Минимальная дополнительная проверка

Не требуется для допустимости. Для качества желательно сравнение на внешнем public при первой возможности и проверка калибровки порога на несбалансированной подвыборке train.