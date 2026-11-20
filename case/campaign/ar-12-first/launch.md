# Запуск ar-12-first

Все команды выполняются из корня репозитория. Локальные дедуплицированные источники находятся в `case/dataset/ar12-deduplicated-v2`; их восстановление описано в [документации данных](../../dataset/README.md).

## Подготовка

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh prepare
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh setup
```

Prepare создаёт пустой журнал с лимитом 18, обучает три бейзлайна и проверяет runner. Синтетическая приёмка имеет отдельный журнал и не расходует бюджет. Setup экспортирует очищенные public/private и создаёт новое закрытое состояние серии вне репозитория.

## Оценщики

Порты 8765, 8767 и 8766 общие с e2e: предыдущие оценщики нужно остановить. Каждую команду запустите в отдельном терминале и оставьте работать:

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh train
```

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh public
```

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh private
```

## Агенты

С ключом `OPENROUTER_API_KEY` в корневом `.env`:

```bash
systemd-inhibit --what=idle:sleep --mode=block --why="AR-12 campaign running" \
  env AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e-openrouter.sh
```

Повторный запуск продолжает ту же кампанию по журналу и workspace. Лимит 18 включает три бейзлайна; для новых вариантов остаётся 15 мест. Финалистов — 3, победителей — 1. Конфигурация не запускает агентов автоматически.

## Статус

```bash
PYTHONPATH=arena arena/.venv/bin/python -m evaluator.search \
  --journal case/campaign/ar-12-first/.run/search.sqlite3 status
docker logs --tail 100 ar12-agents
```

Журнал появляется после успешного prepare. Документы и board находятся в `.run/agent-workspace/campaign/`. Private-результаты остаются вне репозитория по указателю `.run/closed-dir` и не передаются агентам.
