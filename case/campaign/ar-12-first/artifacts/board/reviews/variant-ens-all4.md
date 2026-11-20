# Ревью: variant-ens-all4

- **Кандидат:** `variant-ens-all4`
- **Digest:** `14b0f5a2f994b8a3feea49d5de9b0b44253b1154bdff57e020884aafc87de3b5` (digest_version 2)
- **Снимок:** `/workspace/campaign/work/reviews/variant-ens-all4` (restore из `/search/search.sqlite3`)
- **Родители:** `variant-cat-tuned`, `variant-cat-feat`, `variant-trans-wide`; proposal 15 из 18
- **Автор:** в `board/claims` не заявлен отдельно; сборка выполнена скриптом `/workspace/campaign/work/build_all4.py` (корень workdir, вероятнее оркестратор). Компоненты документированы в findings researcher-b и researcher-c.
- **Ревьюер:** независимый reviewer (не автор кандидата)
- **Дата ревью:** 2026-11-18

## Проверенные материалы

- Снимок целиком (13 файлов): `solve.py`, `meta.json`, `cat_tuned_meta.json`, `cat_feat_meta.json`, `cat_features.py`, `numeric_features.py`, `cat_tuned.cbm`, `cat_feat.cbm`, `trans_model.onnx`, `vocab.txt`, `idf.npz`, `__pycache__/*`.
- Производные материалы: `/workspace/campaign/work/build_all4.py`, `blend_holdout.py`, `blend_large.py`; `work/researcher-c/runwide/holdout_scores.npz`; `work/oof_lin.npy`; `work/researcher-b/oof_cat_tuned.npy`, `oof_cat_feat.npy`; `holdout_ids.txt`.
- Компоненты: `/workspace/campaign/candidates/{variant-cat-tuned,variant-cat-feat,variant-trans-wide}`.
- `board/findings/researcher-b.md`, `board/findings/researcher-c.md`; `campaign.md`; `common/research-protocol.md`, `common/public-evaluation.md`, `common/search-history.md`.
- Журнал `/search/search.sqlite3` (`export`, `status`): кандидат зарегистрирован, измерения нет.
- Официальный train-результат: `/workspace/campaign/results/variant-ens-all4.train.json` — **отсутствует**; журнал на 19:05 UTC 2026-11-18 показывает `running: []`, завершённых измерений 11, записи `train-ens-all4-*` нет.

## Условия и свидетельства

1. **Область изменений.** Ансамбль: `p = 0.5·p_cat_tuned + 0.1·p_cat_feat + 0.4·p_trans`, порог 0.4287. Состав ансамбля, веса и порог разрешены `campaign.md`. Метрика, split, данные и runner не менялись (`conditions_id=ar12-first-f1-v1`).
2. **Воспроизводимость снимка.** Digest восстановленного каталога по `evaluator.contracts.candidate_digest` совпал с зарегистрированным `14b0f5a2…`. Компоненты побайтово идентичны родителям:
   - `cat_tuned.cbm` = `variant-cat-tuned/model.cbm` `76fed422…`;
   - `cat_feat.cbm` = `variant-cat-feat/model.cbm` `8f21cfc2…`;
   - `trans_model.onnx` = `variant-trans-wide/model.onnx` `f9e87ef9…`;
   - `vocab.txt`/`idf.npz` = `variant-cat-tuned` `49af27e8…`/`1824fd6c…` (совпадают и с cat-feat);
   - `cat_features.py` = `features_ext.py` `881debd6…`; `numeric_features.py` = `variant-cat-feat/numeric_features.py` `90deb23c…`; `cat_tuned_meta.json`/`cat_feat_meta.json` совпадают с метаданными родителей.
   Препроцессинг в `solve.py` повторяет компонентные solver'ы: TF-IDF `max_n=3`, числовые признаки дописываются с индекса `len(vocab)` (120000…120035, ширина 120036), байты+словесные хэши для ONNX, `sigmoid` для логита. Зависимости (NumPy/SciPy/CatBoost/ONNX Runtime) в пакете не записаны; окружение фиксируется runner image `sha256:57d8eb19…`.
3. **Провенанс порога и весов (только train).** Порог `0.4287` — ровно argmax F1 на общем 12% holdout train (1371 строка, seed 42, 908 positive). Независимый пересчёт по сохранённым скорам: для весов `0.5/0.1/0.4` argmax-порог `0.4287`, holdout F1 `0.996156`; это же значение — вершина полной сетки весов (шаг 0.1, 66 комбинаций), вторая точка `0.5/0/0.5` даёт `0.995604`. Число `0.9962` из гипотезы подтверждается. Скрипты `blend_holdout.py`/`blend_large.py` читают только train и сохранённые OOF/holdout-скоры; `evaluation.arena.jsonl`/256 в выборе не участвует.
4. **Отсутствие зашитых ответов.** `solve.py` не содержит ID, lookup, узнавания формулировок или ветвлений по тексту. `numeric_features.SUSPICIOUS` — родовой список инструкционной лексики домена (ignore/disregard/system/prompt/…), а не привязка к тестовым примерам; он входит в компонент `variant-cat-feat` и одинаков при обучении и инференсе. Изменений метрики/split нет.
5. **Один ответ на пример.** Инференс детерминирован, один порог. Локальная проверка (6 входов через `solve.py` в снимке): пустой вход → 1, «What is the capital of France?» → 0, известный FP «Describe how shell companies…» → 0 (бленд исправляет), известный FN «what are some stereotypes…new mexico?» → 1, инъекция → 1, leet-инъекция → 1. Арифметика бленда проверена напрямую: на примере с `p_tuned=0.4499, p_feat=0.2271, p_trans=0.9852` получено `0.5·0.4499 + 0.1·0.2271 + 0.4·0.9852 = 0.6418`. Повторные запуски дают тот же ответ; ~1.8 с/пример при лимите 30 с.
6. **Ресурсы.** Официального train-замера нет; solver/judge-ресурсы не подтверждены. При `propose` учтено 300 000 токенов с `tokens_complete=false` (подготовка; неполнота отмечена). Инференс немодельный — ожидаемый расход 0 токенов, проверяемо только будущим замером.

## Вывод о допустимости

**допустим**

Нарушений не найдено: изменения разрешены, компоненты — подлинные замороженные родители, digest сходится, веса и порог выведены только на train, зашитых ответов/узнавания ID нет. **Официальный train-замер на 256 отсутствует**; это ограничение качества и сопоставимости, но не допустимости (известного нарушения нет; train-замер не является условием допуска).

## Качество и риск переобучения (отдельно)

- Единственное evidence — holdout F1 `0.996156` на 1371 строке. Оно оптимистично (порог и веса выбраны на этом же holdout) и **не сопоставимо** с 256-F1 других кандидатов: иной набор, баланс 66.2/33.8, другая процедура. Официального и локального 256-замера ансамбля нет.
- `variant-cat-tuned` и `variant-cat-feat` — почти одна модель (та же TF-IDF-сетка и CatBoost, отличаются 36 числовыми признаками; OOF +0.0004). Поэтому «4-way» фактически два механизма, а вес 0.1 у cat-feat добавляет в основном шум/подстройку. Прирост над `variant-ens-ct` на holdout — `+0.00055` (~1 пример), статистически не установлен.
- Веса выбраны как лучшая точка сетки из 66 комбинаций на тех же 1371 строках, где OOF-скоры cat-компонентов получены fold-моделями, обучавшимися на большей части holdout; суммарный selection-optimism не измерен.
- Порог 0.4287 оптимизирован под распространённость 66.2%; на сбалансированном 256 оптимум может сместиться. Локальная проверка показала, что «What is the capital of France?» проходит с `p=0.4175` — всего на 0.011 ниже порога, т.е. граница хрупкая.
- Остаточный (недоказанный) риск: выбор состава/весов мог быть информирован диагностикой 256 компонентов, существовавшей до сборки. Кода/записи, подтверждающей это, нет; числа точно выводятся из train-holdout.

## Минимальная дополнительная проверка

Официальный train-замер `variant-ens-all4` на 256 (`/workspace/campaign/results/variant-ens-all4.train.json`) и сверка с holdout. Если замера не будет, кандидат можно рассматривать только с явной пометкой «нет 256-подтверждения»; для private это ограничение качества, не запрет.