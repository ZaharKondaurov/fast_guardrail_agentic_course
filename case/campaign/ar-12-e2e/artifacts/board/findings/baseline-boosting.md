# baseline-boosting — findings

ID `baseline-boosting`, предложение 2, родителей нет. Формальная train-проверка `train-baseline-boosting-v1`: 256/256 примеров, `score=0.0`; TP 128, FP 7, FN 0, TN 121; TPR 1.0000, FPR 0.0547. Ресурсы: solver 0 токенов / 474.80 с, judge 0 токенов / 24.72 с.

Механизм: word 1-2 + char 3-4 TF-IDF, длины, CatBoost 100 итераций, порог по 5 % FPR на holdout. Подтверждено: TPR 1.0 на фиксированном наборе. Опровергнуто: калибровка ровно на 5 % не оставляет запаса; 7 ложных срабатываний из 128 негативов (5.47 %) дают score 0. Это ключевое наблюдение для вариантов с более консервативным порогом. Ограничение: один замер; TPR-потолок достигнут. Рекомендация: baseline для сравнения; потомок `variant-boost-calibrated` проверяет гипотезу порога.

Ссылки: `/workspace/campaign/results/train-baseline-boosting-v1.json`; журнал `/search/search.sqlite3`, measurement `train-baseline-boosting-v1`.
