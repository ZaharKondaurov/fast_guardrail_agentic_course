# Запуск кампании AR-12

Выполните команды из корня репозитория. Если локального `ar12-deduplicated-v2` ещё нет, сначала [воспроизведите очищенный набор](../../dataset/README.md#размеры-после-дедупликации). Исходные файлы, список выбранных ID и производное train-разбиение проверяются по `dataset.lock.json` при подготовке.

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh prepare
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh setup
```

Используются те же порты, что и у `ar-12-e2e`. Перед запуском освободите их. Запустите следующие команды в трёх отдельных терминалах:

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh train
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh public
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e.sh private
```

Затем с настроенным `OPENROUTER_API_KEY` в корневом `.env`:

```bash
AR12_CAMPAIGN=case/campaign/ar-12-first bash case/deploy/e2e-openrouter.sh
```

`runtime.json` задаёт лимит 18, трёх финалистов и одного победителя. Для диагностики train используйте [отчёт F1](../../metric/README.md). Скрипт сохраняет путь закрытой серии в `.run/closed-dir`.

## Мониторинг в MLflow

Запустите локальный сервер MLflow в отдельном терминале из корня репозитория:

```bash
uvx --from mlflow==3.16.0 mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///case/campaign/ar-12-first/.run/mlflow.sqlite3
```

После подготовки кампании установите клиент tracking и запустите монитор в другом терминале:

```bash
(cd arena && uv sync --frozen --extra tracking)
PYTHONPATH=arena arena/.venv/bin/python -m case.deploy.monitor \
  --campaign case/campaign/ar-12-first --tracking-uri http://127.0.0.1:5000
```

Откройте `http://127.0.0.1:5000`, эксперимент `Arena / ar-12-first` и актуальный run. В разделе Tags видны фаза, активные роли, ID делегированных задач и текущих замеров. Счётчики в Metrics обновляются при изменении статуса. Монитор проверяет его каждые три минуты. Открытый ход оркестратора может включать ожидание инструментов или исследователей. Для делегированных вызовов без метки роли используется `unattributed`. Для видимости ролей нужны события от новой версии launcher и Hermes-адаптера. Private передаётся только как статус, без оценок и содержимого выборки.
