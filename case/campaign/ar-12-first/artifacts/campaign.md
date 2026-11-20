# Кампания AR-12 на очищенных данных

Каталог кампании: `case/campaign/ar-12-first`. Протокол: подготовка, исследования через Hermes с промптами Arena, независимое ревью, train/public-оценки и одна закрытая private-серия.

- **Данные:** `ar12-deduplicated-v2`: 11 676 train, 1 114 public, 1 123 private. Внутри разделов и между ними нет совпадений по ID и тексту после схлопывания пробелов. Исходные тексты сохранены.
- **Оценочная подвыборка:** 256 примеров из train, по 128 каждого класса, seed 42. ID закреплены в `evaluation-ids.txt`. Для обучения остаются 11 420 примеров; оценочную подвыборку нельзя включать в обучение или калибровку. Контрольные суммы источников и производных файлов закреплены в `dataset.lock.json` и проверяются при подготовке.
- **Метрика:** замороженный пакет `metric/`, версия `ar12-f1-v1`, направление maximize. Целевой score — чистый бинарный F1. TPR, FPR, recall, precision, p99, throughput и гейты — диагностика и не меняют score. Без доверенного времени инференса p99/throughput и их гейты равны `null`; Docker wall time не заменяет время инференса.
- **Модель:** `deepseek/deepseek-v4.1-flash-1017` через OpenRouter для оркестратора, исследователей и ревьюеров. Решатель и метрика работают без LLM. Public/private-данные и закрытые результаты доступны только оценщику и оператору; агентам выдаются train, общий public score и статус private.
- **Разрешённые изменения:** алгоритм, признаки, гиперпараметры, пороги и состав ансамбля. Данные, метрика, лимит и условия закрытой серии в ходе поиска не меняются. Таймаут runner — 30 секунд на пример.

## Лимит предложений

18 доступных мест в журнале: три бейзлайна и до 15 новых предложений. Каждую новую версию регистрируют до проверки со снимком, гипотезой, изменениями и родителями. Неудачные предложения тоже расходуют место; технические повторы прежней версии не создают новое предложение. Токены и время учитываются, но не являются дополнительными бюджетами поиска.

## Финализация

Поиск заканчивается после исчерпания лимита либо явной команды пользователя остановиться. Завершаются разрешённые проверки и независимые ревью. Оркестратор выбирает ровно 3 разных финалиста по train/public F1 с учётом ревью, устойчивости и разнообразия, записывает основания в `selection.md` и замораживает единый пакет.

Private-оценщик принимает одну серию и выбирает 1 победителя по F1; при равенстве применяются штатные правила Arena по токенам и времени решателя. Агентам доступен только статус серии. Исследовательский `report.md` содержит доступные train/public-свидетельства и ограничения, без private-баллов. Версия условий и закрытой серии — `ar12-first-f1-v1`.

[Порядок запуска](launch.md)

## Состояние поиска (обновляется оркестратором)

Журнал: `/search/search.sqlite3` (канонический, max_solutions=18). Readiness private проверен 2026-11-18T15:46:29Z: mode=campaign, gate_enabled=true, journal_ready=true, evaluation_enabled=true, max_solutions=18, finalists=3, output_count=1, release_id=ar12-first-f1-v1.

Схема train-валидации: внутренний 12% стратифицированный holdout train (seed 42) для порогов и весов; 5-fold OOF для линейных/CatBoost порогов; 256-подвыборка `ar12-deduplicated-v2-train-holdout-v1` используется только как официальный train-замер Arena (не для обучения/калибровки). Сопоставимый ключ train: split=train, dataset=ar12-deduplicated-v2-train-holdout-v1, conditions=ar12-first-f1-v1.

| # | ID | Семейство | 256 F1 | TPR | FPR | Precision | Гейт FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | matching | baseline словарь | 0.8627 | 0.8594 | 0.1328 | 0.8661 | fail |
| 2 | boosting | baseline CatBoost | 0.9653 | 0.9766 | 0.0469 | 0.9542 | pass |
| 3 | transformer | baseline byte ONNX | 0.9569 | 0.9531 | 0.0391 | 0.9606 | pass |
| 4 | variant-lin-wordchar | linear TF-IDF | 0.9882 | 0.9844 | 0.0078 | 0.9921 | pass |
| 5 | variant-lin-char | char TF-IDF | 0.9922 | 0.9922 | 0.0078 | 0.9922 | pass |
| 6 | variant-lin-robust | deobf linear | 0.9922 | 0.9922 | 0.0078 | 0.9922 | pass |
| 7 | variant-svc | LinearSVC | 0.9922 | 0.9922 | 0.0078 | 0.9922 | pass |
| 8 | variant-cat-tuned | CatBoost | 0.9843 | 0.9766 | 0.0078 | 0.9921 | pass |
| 9 | variant-cat-feat | CatBoost+numeric | 0.9882 | 0.9844 | 0.0078 | 0.9921 | pass |
| 10 | variant-ens-lc | linear+cat | 0.9922 | 0.9922 | 0.0078 | 0.9922 | pass |
| 11 | variant-trans-192 | byte transformer | 0.9528 | 0.9453 | 0.0391 | 0.9603 | pass |
| 12 | variant-trans-wide | byte+word transformer | 0.9921 | 0.9844 | 0.0000 | 1.0000 | pass |
| 13 | variant-ens-ct | cat+transformer | 0.9961 | 0.9922 | 0.0000 | 1.0000 | pass |
| 14 | variant-ens-lct | linear+cat+transformer | 0.9961 | 0.9922 | 0.0000 | 1.0000 | pass |
| 15 | variant-ens-all4 | 2×cat+transformer | 0.9961 | 0.9922 | 0.0000 | 1.0000 | pass |
| 16 | variant-ens-rct | robust+cat+transformer | 0.9961 | 0.9922 | 0.0000 | 1.0000 | pass |
| 17 | variant-trans-wide-full | transformer на 100% train | 0.9961 | 1.0000 | 0.0078 | 0.9922 | pass |
| 18 | variant-ens-lct-full | linear+cat+transformer-full | 0.9961 | 1.0000 | 0.0078 | 0.9922 | pass |

Предложения исчерпаны (18/18); новые решения не регистрируются. Все плановые train- и public-замеры завершены (18 train + 5 public), незавершённых проверок нет. Public-замеры выполняются только для прошедших независимое ревью кандидатов и фиксируются в журнале с split=public, dataset=ar12-deduplicated-v2-public, conditions=ar12-first-f1-v1. Private-серия отправлена 2026-11-18T20:03Z и завершена со статусом `complete` (2026-11-18T21:25Z). Финалисты заморожены: `variant-ens-lct-full`, `variant-lin-char`, `variant-trans-wide-full`; bundle digest `c0cbb0c6a3ae71058aa290ba429b3eadc182aeb52a14903063344b972eacb133`, `finalize` 2026-11-18T20:03:02Z.

### Public-замеры (общий score, 1114 примеров)

| ID | Public F1 | Условия |
| --- | --- | --- |
| variant-ens-lct-full | 0.95755 | split=public, dataset=ar12-deduplicated-v2-public, subset=ar12-public-full-v1, conditions=ar12-first-f1-v1 |
| variant-ens-ct | 0.95737 | те же |
| variant-ens-lct | 0.95737 | те же |
| variant-lin-char | 0.95714 | те же |
| variant-trans-wide-full | 0.95524 | те же |

Public-замеры резко ниже train-256: публичный набор содержит более трудные/иные примеры, поэтому отбор финалистов опирается прежде всего на public и на разнообразие механизмов, а train-256 (почти насыщенный) используется как проверка работоспособности. Private-серия отправлена 2026-11-18T20:03Z и завершена со статусом `complete` (2026-11-18T21:25Z). Финалисты заморожены: `variant-ens-lct-full`, `variant-lin-char`, `variant-trans-wide-full`; bundle digest `c0cbb0c6a3ae71058aa290ba429b3eadc182aeb52a14903063344b972eacb133`, `finalize` 2026-11-18T20:03:02Z.
