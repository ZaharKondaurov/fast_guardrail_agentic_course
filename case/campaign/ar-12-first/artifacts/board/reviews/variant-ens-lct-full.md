# Ревью — `variant-ens-lct-full`

- **Кандидат:** `variant-ens-lct-full`, предложение 18
- **Digest (journal):** `d68a3d972b9a20bbe200ee4e43ef5a7ef448424e5822251aff71e3171e14a287`
- **Снимок:** `/workspace/campaign/work/reviews/variant-ens-lct-full` (восстановлен `evaluator.search restore`; digest v2)
- **Родители:** `variant-lin-wordchar`, `variant-cat-tuned`, `variant-trans-wide-full`
- **Автор:** оркестратор кампании (сборка из замороженных компонентов; отдельного build-скрипта для этого ID в `work/` нет); компоненты — `variant-lin-wordchar` (root `work/train_linear.py`), `variant-cat-tuned` (researcher-b), `variant-trans-wide-full` (researcher-d)
- **Ревьюер:** независимое ревью Arena (не автор кандидата)
- **Дата проверки:** 2026-11-18, ~18:20–18:45 UTC

## Что проверено

Прочитаны `solve.py`, `cat_features.py`, `linear_features.py`, `meta.json`, `cat_meta.json`, `model.npz`/`idf.npz`/`.cbm`/`model.onnx`, board findings researcher-b/d (researcher-a — как контекст линейного семейства), claims researcher-b/c, `work/build_ensembles.py` (шаблон сборки `variant-ens-lct`), `work/blend_holdout.py`, `work/researcher-c/train_full.py`, `work/researcher-c/variant-trans-wide-full-hypothesis.md`, журнал `evaluator.search`.

### Совпадение компонентов с замороженными родителями

| файл в ансамбле | sha256 (первые 16) | родитель | совпадение |
|---|---|---|---|
| `linear_features.py` | `adcaa86ed90ab915` | `variant-lin-wordchar/features.py` | да |
| `linear_vocab.txt` | `3ee69931cc6e694e` | `variant-lin-wordchar/vocab.txt` | да |
| `linear_model.npz` | `cf24879f788756dd` | `variant-lin-wordchar/model.npz` | да |
| `cat_features.py` | `881debd6be450729` | `variant-cat-tuned/features_ext.py` | да |
| `cat_vocab.txt` | `49af27e8c8cd5e8f` | `variant-cat-tuned/vocab.txt` | да |
| `cat_idf.npz` | `1824fd6c850e8e9e` | `variant-cat-tuned/idf.npz` | да |
| `cat_model.cbm` | `76fed42282808cfd` | `variant-cat-tuned/model.cbm` | да |
| `cat_meta.json` | `d4f2f90cdd8370cd` | `variant-cat-tuned/meta.json` | да |
| `trans_model.onnx` | `c9a063fda435f6e5` | `variant-trans-wide-full/model.onnx` | да |

`trans_model.onnx` — именно полный вариант (`c9a063fd…`), а не родитель `variant-trans-wide` (`f9e87ef9…`). `solve.py` побайтово совпадает с `variant-ens-lct/solve.py` (`e9513344…`); `meta.json` отличается только `candidate_id`, `note` и именем trans-компонента. Это новая сборка, а не технический повтор проверки.

### Разрешённые изменения, данные, метрика, split

- Изменён только состав ансамбля (замена trans-компонента на `variant-trans-wide-full`) при тех же весах/пороге; это разрешённая область («состав ансамбля», «пороги»).
- В пакете нет train-строк, эталонов, ID или таблиц ответов; `solve.py` не содержит lookup, распознавания ID/формулировок, сетевых вызовов. 256-набор не входит в train (0 совпадений ID/текста).
- Метрика `ar12-f1-v1`, split и лимит не менялись.

### Происхождение порога и весов — важное замечание

`meta.json`:

```
"w_linear": 0.2, "w_cat": 0.4, "w_trans": 0.4,
"threshold": 0.4888,
"components": {"trans": "variant-trans-wide-full"},
"threshold_source": "12% train holdout (seed 42) blend F1 argmax"
```

Фактически 0.4888 и 0.2/0.4/0.4 — это значения, перенесённые из `variant-ens-lct`, где trans-компонентом был `variant-trans-wide` (обучен на 10 049 строках, holdout чистый). Для `variant-ens-lct-full` trans-компонент `variant-trans-wide-full` обучен на всех 11 420 строках, **включая те же 1371 holdout-строки**. Поэтому:

- строка `threshold_source` в `meta.json` для этого кандидата неточна: порог не является argmax F1 на чистом holdout именно этого бленда; он унаследован;
- веса 0.2/0.4/0.4 также не пересчитаны под новый компонент;
- это не утечка 256 и не изменение метрики/split: порог и веса остаются train-derived константами, а не результатом подбора на 256. Но доказательство переноса порога на полный компонент отсутствует.

Это ограничение качества/документации, а не установленное нарушение: происхождение 0.4888 прослеживается до `variant-ens-lct` и его holdout, а не является неизвестной подсказкой или 256-подгонкой.

### Отсутствие измерений

- Локального 256-замера `variant-ens-lct-full` в `candidates/` или `work/` нет.
- Официального train-замера `results/variant-ens-lct-full.train.json` на момент проверки нет; в журнале нет measurement для этого ID.
- Собственного board finding/claims для этого кандидата нет; есть только родительские findings (a/b/d) и meta.
- Единственное косвенное свидетельство — локальный 256-замер trans-компонента `variant-trans-wide-full`: F1 `0.996109` (128/1/127/0), `work/researcher-c/eval256-variant-trans-wide-full.json`. Это замер **компонента**, а не ансамбля; переносить его на `variant-ens-lct-full` нельзя.

### Sanity-check

В восстановленном снимке `solve.py` запущен на 7 входах из 256 (один процесс на вход, cwd = каталог кандидата): ответы стабильны при повторе, время ~1.8–1.9 с/пример, ошибок/невалидного JSON нет. Формула `p = 0.2*p_linear + 0.4*p_cat + 0.4*p_trans` и порог 0.4888 совпадают с `solve.score()`. На известных ошибках родителей: `neuralchemy-train-1114` (true 1) — 1, `neuralchemy-train-2325` (true 0) — 1 (FP), `toxicchat-train-286` (true 1) — 1. Это только sanity-check, не оценка F1.

## Доступ к данным, ресурсы, один ответ на пример

- Агент использовал только `train.jsonl` и локальный 256-набор; public/private данные и закрытые результаты не читались. Public-замера для ID нет; private-статус не раскрывался.
- Подготовка: `propose.resources = 300 000 tokens, tokens_complete=false, 180 s`. Это неполный учёт подготовки, не inference. Inference немодельный: LLM-вызовов в `solve.py` нет, токены решателя структурно нулевые; локальный sanity-check ~1.8–1.9 с/пример.
- Официального train-замера и его `resources.solver/judge` нет; один ответ на пример официально не проверен. Локальный `solve.py` детерминирован при повторе.

## Вывод о допустимости

**допустим**

Установленных нарушений нет: изменения в разрешённой области, компоненты побайтово совпадают с замороженными родителями, 256-набор не использован для обучения/калибровки/порога, зашитых ответов и lookup нет, метрика/split не менялись, снимок и lineage зарегистрированы. Отдельно фиксирую дефект документации: `threshold_source` в `meta.json` описывает не тот holdout, на котором реально выбран порог для этого бленда; порог унаследован из `variant-ens-lct`.

## Качество и риск переобучения

**Качество: недостаточно сведений.** Точный снимок `variant-ens-lct-full` не измерен ни локально на 256, ни официально. Неизвестны F1, ошибки и калибровка ансамбля после замены trans-компонента. Риски: (1) порог 0.4888 и веса 0.2/0.4/0.4 не проверены на полном компоненте, а holdout, на котором они выбирались, теперь входит в его обучение; (2) полный trans-компонент имеет более экстремальные логиты (train-диагностика), поэтому калибровка бленда может сдвинуться; (3) кандидат почти дублирует `variant-ens-lct` по механизму (тот же solve, те же линейный/CatBoost, отличается только transformer), поэтому как отдельный финалист он даёт мало разнообразия; (4) сборка выполнена через ~1.5 минуты после локального 256-замера `variant-trans-wide-full` (0.9961); временная близость делает вероятным, что решение собрать этот бленд было информировано 256-диагностикой. Кода/записи об этом нет; порог/веса — train-derived, но если сборка действительно выбрана по 256, оценка такого ансамбля на 256 не будет независимой. Риск переобучения — умеренный/неизвестный; без замера его нельзя оценить.

## Официальный train-замер

**Отсутствует (pending).** Это ограничивает только оценку качества, не допустимость. Если `results/variant-ens-lct-full.train.json` появится, проверить score/F1, tp/fp/tn/fn, один ответ на пример, нулевые токены немодельного решателя, solver seconds, ошибки; сравнить с `variant-ens-lct` и `variant-trans-wide-full` на одном и том же 256-наборе.

## Минимальная дополнительная проверка

Официальный train-замер точного снимка на 256 (те же conditions `ar12-first-f1-v1`), затем — сравнение с `variant-ens-lct` и `variant-trans-wide-full` на этом же наборе и диагностика чувствительности F1 к порогу 0.4888 (только отчёт; порог после 256 не менять). До такого замера кандидат не следует выбирать финалистом по quality-основаниям, хотя admissibility не блокирует отправку.
