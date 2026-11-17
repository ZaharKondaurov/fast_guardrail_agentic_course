# Быстрый гардрейл для потока сообщений: навык, выращенный на арене

Тема: AR-12: Быстрый гардрейл для потока сообщений

## Итог в одной таблице

#### Метрики производительности

| Model | p99 (ms) | throughput (messages/s) 
|---|---|---|
| Baseline Matching    | 0.08 ms | 102062.33 |
| Baseline Catboost    | 7.16 ms | 177.49    |
| Baseline Transformer | 0.18 ms | 6005.88   |

#### Метрики качества

| Model | Train | Public | Private |
|---|---|---|---|
| Baseline Matching    | TPR=87,29%</br> FPR=11,46% | TPR 83,17%</br> FPR 9,58%  | |
| Baseline Catboost    | TPR=99,43%</br> FPR=3,08%  | TPR 96,04%</br> FPR 10,07% | |
| Baseline Transformer | TPR 97,42%</br> FPR 4,73%  | TPR 90,38%</br> FPR 6,63%  | |



Лимит предложенных решений: <N>. Кандидатов в журнале: <M>. AUC кривой: <...>.

## Структура

```
docs/domain.md      обзор домена и классических подходов
docs/arena.md       как устроена арена (или своя методология и сравнение с ареной)
docs/evolution.md   разбор эволюции: кривая, родословная, ключевые мутации, переобучение
docs/solution.md    итоговое решение своими словами
docs/talk.md        план рассказа
docs/adr/           решения: методология, split, метрика, лимит
case/               кейс арены (по cases/template): задача, данные, метрика, кампания, стартовый способ
skill/              итоговый навык: SKILL.md + scripts/ + tests/
results/            выгрузки журнала, замеры, графики
```

## Как воспроизвести

Для локальной работы нужны Python 3.12 и `uv`.

```bash
uv sync --locked
uv run pytest case/metric/tests
```

Команда подготовки данных описана в [case/dataset/README.md](case/dataset/README.md).
Расчёт метрик и условия гейтов описаны в [case/metric/README.md](case/metric/README.md).

```bash
<клонировать арену и подключить кейс>
<сгенерировать данные: python case/dataset/generate.py --split train --seed ...>
<проверить метрику: pytest case/metric/tests>
<подготовка и запуск кампании — по prompts/prepare.md и prompts/start.md арены>
<построить кривую и родословную: ...>
```

## Модели

<Модель исследователей и (если есть) решателя/судьи, версии, AA Intelligence Index, дата.>
