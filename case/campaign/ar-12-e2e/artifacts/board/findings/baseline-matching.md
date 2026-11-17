# baseline-matching — findings

ID `baseline-matching`, предложение 1, родителей нет. Формальная train-проверка `train-baseline-matching-v1`: 256/256 примеров, `score=0.0`; TP 112, FP 15, FN 16, TN 113; TPR 0.8750, FPR 0.1172. Ресурсы: solver 0 токенов / 163.51 с, judge 0 токенов / 24.79 с.

Механизм: словарь 100 частотных n-грамм, исключающих частые безопасные n-граммы; решение по точному совпадению. Подтверждено: высокая скорость и нулевые модельные расходы. Опровергнуто: порог по словарю не удерживает FPR ≤ 5 % на фиксированном наборе; score 0. Ограничение: правило бинарное, без калибровки уверенности. Рекомендация: как baseline для сравнения, не для private.

Ссылки: `/workspace/campaign/results/train-baseline-matching-v1.json`; журнал `/search/search.sqlite3`, measurement `train-baseline-matching-v1`.
