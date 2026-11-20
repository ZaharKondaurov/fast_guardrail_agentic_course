# Ревью: variant-ens-ct

- **Кандидат:** `variant-ens-ct`
- **Digest:** `8a685435c95767d078d400c47a4281ef439a983b19b8687a72583d2550bcc942` (digest_version 2)
- **Снимок:** `/workspace/campaign/work/reviews/variant-ens-ct` (restore из `/search/search.sqlite3`)
- **Родители:** `variant-cat-tuned`, `variant-trans-wide`; proposal 13 из 18
- **Автор:** в `board/claims` не заявлен отдельно; сборка выполнена скриптом `/workspace/campaign/work/build_ensembles.py` (корень workdir, вероятнее оркестратор). Компоненты документированы в findings researcher-b (`variant-cat-tuned`) и researcher-c (`variant-trans-wide`).
- **Ревьюер:** независимый reviewer (не автор кандидата)
- **Дата ревью:** 2026-11-18

## Проверенные материалы

- Снимок целиком (13 файлов): `solve.py`, `meta.json`, `cat_meta.json`, `cat_features.py`, `linear_features.py`, `cat_model.cbm`, `cat_vocab.txt`, `cat_idf.npz`, `linear_model.npz`, `linear_vocab.txt`, `trans_model.onnx`, `__pycache__/*`.
- Производные материалы: `/workspace/campaign/work/build_ensembles.py`, `blend_holdout.py`; `work/researcher-c/runwide/holdout_scores.npz`; `work/oof_lin.npy`; `work/researcher-b/oof_cat_tuned.npy`, `oof_cat_feat.npy`; `holdout_ids.txt`.
- Компоненты: `/workspace/campaign/candidates/variant-cat-tuned`, `variant-trans-wide`, `variant-lin-wordchar`.
- `board/findings/researcher-b.md`, `board/findings/researcher-c.md`; `campaign.md`; `common/research-protocol.md`, `common/public-evaluation.md`, `common/search-history.md`.
- Журнал `/search/search.sqlite3` (`export`, `status`): кандидат зарегистрирован, измерения нет.
- Официальный train-результат: `/workspace/campaign/results/variant-ens-ct.train.json` — **отсутствует**; журнал на 19:05 UTC 2026-11-18 показывает `running: []`, завершённых измерений 11, записи `train-ens-ct-*` нет.

## Условия и свидетельства

1. **Область изменений.** Это ансамбль замороженных компонентов: `p = 0.5·p_cat + 0.5·p_trans`, порог 0.46255. Состав ансамбля, веса и порог явно разрешены `campaign.md` («состав ансамбля», «пороги»). Метрика, split, данные, runner не менялись; `conditions_id=ar12-first-f1-v1`.
2. **Воспроизводимость снимка.** Digest восстановленного каталога по `evaluator.contracts.candidate_digest` совпал с зарегистрированным `8a685435…`. Вложенные компоненты **побайтово идентичны** замороженным родителям:
   - `cat_model.cbm` = `variant-cat-tuned/model.cbm` `76fed422…`;
   - `trans_model.onnx` = `variant-trans-wide/model.onnx` `f9e87ef9…`;
   - `linear_model.npz`/`linear_vocab.txt` = `variant-lin-wordchar` `cf24879f…`/`3ee69931…`;
   - `cat_meta.json` = `variant-cat-tuned/meta.json` `d4d2f90c…`; `cat_features.py` = `variant-cat-tuned/features_ext.py` `881debd6…`.
   Препроцессинг в `solve.py` совпадает с компонентными solver'ами: байты+словесные хэши для трансформера, TF-IDF `max_n=3` для CatBoost, `sigmoid` для логитов. Зависимости: NumPy/SciPy/CatBoost/ONNX Runtime; версии в пакете не записаны, но окружение фиксируется runner image `sha256:57d8eb19…` (в `preparation.json`), это общий для кампании контракт.
3. **Провенанс порога и весов (только train).** Порог `0.46255` — ровно argmax F1 на общем 12% holdout train (1371 строка, seed 42, 908 positive). Независимый пересчёт по сохранённым скорам: для весов `0/0.5/0.5` argmax-порог `0.462552`, holdout F1 `0.995604` — совпадает с `meta.json`. Скрипт `blend_holdout.py` читает только `train.jsonl`, OOF-массивы и `holdout_scores.npz`; `evaluation.arena.jsonl`/256 в выборе не участвует. Веса `0.5/0.5` — априорный равный блендинг из гипотезы, а не лучшая точка сетки (лучшая по holdout — `0.5/0.1/0.4` для all4).
4. **Отсутствие зашитых ответов.** `solve.py` не содержит ID, lookup, узнавания формулировок или ветвлений по тексту; компоненты — обычные обученные модели, словари/веса получены из train. Изменений метрики/split нет.
5. **Один ответ на пример.** Инференс детерминирован, один порог, один ответ на вход. Локальная проверка (6 входов через `solve.py` в снимке): пустой вход → 1, «What is the capital of France?» → 0, известный FP cat-модели «Describe how shell companies…» → 0 (бленд исправляет FP), известный FN trans-модели «what are some stereotypes…new mexico?» → 1, инъекция → 1, leet-инъекция → 1. Арифметика бленда проверена напрямую: `0.5·0.4499 + 0.5·0.9852 = 0.7176`. Повторные запуски дают тот же ответ; ~1.8 с/пример при лимите 30 с.
6. **Ресурсы.** Официального train-замера нет, поэтому solver/judge-ресурсы не подтверждены. При `propose` учтено 584 065 токенов с `tokens_complete=false` (подготовка; неполнота отмечена). Инференс немодельный — ожидаемый расход 0 токенов; это проверяемо только будущим замером.

## Вывод о допустимости

**допустим**

Нарушений не найдено: изменения разрешены, компоненты — подлинные замороженные родители, digest сходится, порог и веса выведены только на train, зашитых ответов/узнавания ID нет. **Официальный train-замер на 256 отсутствует.** Это ограничивает только качество/сопоставимость, но не допустимость: известного нарушения нет, а метрика на train не является условием допуска (п. 3–4 `search-history.md`).

## Качество и риск переобучения (отдельно)

- Evidence ограничено train-holdout: F1 `0.995604` на 1371 строке. Это **не** сопоставимо с 256-F1 других кандидатов: другой набор, баланс 66.2/33.8 вместо 128/128, и порог выбран на этом же holdout (оптимистично). Официального 256-замера ансамбля нет; локального 256-замера ансамбля тоже нет.
- Порог/веса подобраны на том же 12% holdout, который researcher-c использовал для выбора эпохи и порога `variant-trans-wide`; OOF-скоры cat-компонента на этих строках получены fold-моделями, обучавшимися на большей части holdout. Суммарный selection-optimism не измерен.
- Порог 0.46255 оптимизирован под распространённость 66.2%; на сбалансированном 256 оптимум F1 может сместиться.
- Упаковка содержит мёртвые артефакты линейной модели (`w_linear=0`): ~2.9 МБ и лишний расчёт на каждый пример; `meta.components` указывает `linear`, хотя вклад нулевой. На решения не влияет, но это лишний вес и неаккуратная родословная (родитель `variant-lin-wordchar` не заявлен).
- Механизм правдоподобен: на проверенных примерах бленд исправляет FP cat-модели и FN трансформера; это согласуется с гипотезой о декоррелированных ошибках. Но вклад в F1 на 256 не измерен.
- Остаточный (недоказанный) риск: выбор состава ансамбля мог быть информирован пост-фактум диагностикой 256 компонентов (файлы `*.eval256.json` существовали до сборки). Кода/записи, подтверждающей это, нет; порог и веса точно выводятся из train-holdout. Считать доказанным нарушением нельзя.

## Минимальная дополнительная проверка

Официальный train-замер `variant-ens-ct` на 256 (`/workspace/campaign/results/variant-ens-ct.train.json`) и сверка его с holdout-числом. Если замер не появится, для отбора использовать кандидата только с явной пометкой «официального 256-замера нет»; для private это ограничение качества, не запрет.