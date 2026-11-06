# Отчёт о затратах пилота

Из корня репозитория укажите фактический путь к журналу. Пример предполагает, что зависимости Arena установлены в её `.venv`.

```bash
journal=case/campaign/ar-12-pilot/search.sqlite3
PYTHONPATH=arena arena/.venv/bin/python -m evaluator.search --journal "$journal" export |
  jq '
    def summary: {count: length, mean: (if length == 0 then null else add / length end)};
    . as $journal
    | [
        .candidates[]
        | select(.candidate_id | startswith("variant-"))
        | . as $candidate
        | [
            $journal.measurements[]
            | select(.candidate_id == $candidate.candidate_id
                     and .comparison.split == "train"
                     and (.state == "succeeded" or .state == "failed"))
            | .elapsed_seconds
          ] as $attempts
        | {
            candidate_id,
            preparation_seconds: .resources.seconds,
            research_tokens: .resources.total_tokens,
            tokens_complete: .resources.tokens_complete,
            evaluation_seconds: (if ($attempts | length) == 0 then null else $attempts | add end)
          }
      ] as $rows
    | {
        candidates: $rows,
        preparation_seconds: ([$rows[].preparation_seconds] | summary),
        evaluation_seconds: ([$rows[].evaluation_seconds | select(. != null)] | summary),
        total_seconds: ([$rows[] | select(.evaluation_seconds != null)
                         | .preparation_seconds + .evaluation_seconds] | summary),
        research_tokens: ([$rows[] | select(.tokens_complete) | .research_tokens] | summary)
      }'
```

Выгружайте итог после завершения проверок. Неполный учёт токенов отображается в строке кандидата и исключается из среднего. Кандидат без завершённой проверки входит в среднее время подготовки. Длительность оценки складывается из завершённых попыток, включая неудачные.
