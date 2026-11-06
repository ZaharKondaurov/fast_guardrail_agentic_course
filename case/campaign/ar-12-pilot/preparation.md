# Подготовка пилота AR-12

Статус — локальные данные, метрика, стартовые пакеты и train-оценщик готовы. Smoke Arena, Docker-приёмка и реальный smoke OpenRouter прошли 6 ноября 2026. Поиск ожидает утверждения бюджета. [Команды запуска](launch.md) · [Как повторить smoke-проверки](../../deploy/README.md).

## Данные и метрика

- Снимок `ar12-pilot-train-v1` получен из локального `case/dataset/ar12-train/ar12.train.jsonl` с 14 281 уникальным ID. Ревизии исходных внешних наборов для этой копии не подтверждены.
- `case/dataset/ar12-pilot-v1/data/train.jsonl` содержит 14 025 записей для обучения и настройки. `case/dataset/ar12-pilot-v1/data/evaluation.jsonl` содержит 256 записей из [списка ID](../../dataset/ar12-pilot-v1/evaluation-ids.txt), по 128 на класс. Проверено отсутствие пересечений по ID. Оба файла сохраняют исходные `id`, `label` и `class`.
- `case/dataset/ar12-pilot-v1/data/evaluation.arena.jsonl` — вход для train-оценщика. Все 256 строк прошли проверку схемой `evaluator.contracts.Example`. Public и private в пилоте пусты. Каталог `case/dataset/ar12-pilot-v1/data/` хранится локально.
- Пакет [ar12-tpr-v3](metric/metric.json) заморожен в `metric/` и проверен командой `evaluator.prepare verify`. Тесты метрик прошли, 18 из 18.

## Runner и приёмка

- Три baseline переобучены на `case/dataset/ar12-pilot-v1/data/train.jsonl` и упакованы в `.run/candidates/`. Все три пакета возвращают ответ в формате Arena при независимом запуске из другого рабочего каталога.
- Образ runner собран с зависимостями из `uv.lock`. ID принятого образа хранится в `.run/runner-image.txt`. Train-сервис запускается на loopback, runner работает без сети, таймаут на пример — 30 секунд.
- Сквозная приёмка прошла на двух синтетических примерах. Контроль, три baseline и тестовый вариант `variant-matching` завершили `propose → submit → status → result → export` через HTTP и Docker. Контроль вернул score 1. Вариант прошёл локальное обучение, регистрацию с родителем и 123 подставными токенами исследователя, повтор команды и финализацию. Свидетельства сохранены в `.run/acceptance-h9ff9anm/journal.json`. Синтетический журнал изолирован от журнала поиска.
- Штатный `python -m evaluator.smoke` прошёл. На образе пилота также прошли немодельные проверки Arena `check_program`, `check_restored_program` и `check_large_inputs`.

## Исследователь

Пользователь сообщил о доступе к OpenRouter. Предлагаемый фиксированный ID — [deepseek/deepseek-v4.1-flash-1017](https://openrouter.ai/deepseek/deepseek-v4.1-flash-1017). OpenRouter можно использовать для синтетической проверки учёта токенов. Для работы с данными кейса действуют условия кампании о внутреннем endpoint. Внешнему харнесу доступна работа с кодом и синтетическими примерами, если содержимое датасета и выводы с его фрагментами изолированы от модели.

Реальный `openrouter-smoke` на одном синтетическом запросе к `deepseek/deepseek-v4.1-flash-1017` вернул 53 токена за 0,94 секунды. Результат сохранён в `.run/openrouter-smoke.json`. Запись usage по ID кандидата проверена с подставным ответом. Поиск ещё не запускался.
