# baseline-transformer — findings

ID `baseline-transformer`, предложение 3, родителей нет. Формальная train-проверка `train-baseline-transformer-v1`: 256/256 примеров, `score=0.9296875`; TP 119, FP 6, FN 9, TN 122; TPR 0.9297, FPR 0.0469. Ресурсы: solver 0 токенов / 196.39 с, judge 0 токенов / 24.93 с.

Механизм: байтовый трансформер 128 токенов, динамическая INT8-квантизация ONNX, cutoff по 5 % FPR на holdout. Подтверждено: проходит FPR-гейт и даёт положительный score; самый быстрый из прошедших гейт baseline. Опровергнуто: TPR ниже boosting; часть ошибок — ложные негативы на токсичных и инъекционных примерах. Ограничение: INT8-модель недотренирована (4 эпохи в baseline); public/private не измерялись. Рекомендация: baseline-кандидат для private, если нужен контроль исходного подхода с положительным train-score.

Ссылки: `/workspace/campaign/results/train-baseline-transformer-v1.json`; журнал `/search/search.sqlite3`, measurement `train-baseline-transformer-v1`.
