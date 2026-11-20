# Отбор финалистов AR-12 (кампания ar-12-first)

## Основание завершения поиска

Исчерпан лимит предложений: журнал `/search/search.sqlite3` содержит 18 предложений при `max_solutions=18`, свободных мест 0. Явной команды человека остановить поиск не поступало; остановка в журнале не записана. Все плановые проверки завершены: 18 train-замеров и 5 public-замеров, `status.running` пуст, `completed_measurements=23`. Private-серия до этого момента не отправлялась.

## Условия сравнения

- Train: `split=train`, `dataset=ar12-deduplicated-v2-train-holdout-v1`, `subset` = закреплённая 256-подвыборка (`evaluation-ids.txt`), `conditions=ar12-first-f1-v1`. Официальный замер Arena, 256 примеров, 128/128 классов. Пороги и веса кандидатов настраивались только на train-производных данных (12% holdout seed 42 или 5-fold OOF), не на этой подвыборке.
- Public: `split=public`, `dataset=ar12-deduplicated-v2-public`, `subset=ar12-public-full-v1`, `conditions=ar12-first-f1-v1`, 1 114 примеров. Агенту доступен только общий score; он записан в журнал.
- Private: 1 123 примера, результаты агенту недоступны.

## Доступные оценки

| ID | Семейство | Train 256 F1 | Public F1 | Ревью |
| --- | --- | --- | --- | --- |
| variant-ens-lct-full | linear+CatBoost+byte-transformer (full) | 0.99611 | **0.95755** | допустим |
| variant-ens-ct | CatBoost+byte-transformer | 0.99608 | 0.95737 | допустим |
| variant-ens-lct | linear+CatBoost+byte-transformer | 0.99608 | 0.95737 | допустим |
| variant-lin-char | char-only TF-IDF linear | 0.99219 | 0.95714 | допустим |
| variant-trans-wide-full | byte+hashed-word transformer, full train | 0.99611 | 0.95524 | допустим |
| variant-ens-all4 | 2×CatBoost+transformer | 0.99608 | не измерялся | допустим |
| variant-ens-rct | robust-linear+CatBoost+transformer | 0.99608 | не измерялся | не ревьюился |
| variant-lin-wordchar | word+char TF-IDF linear | 0.98824 | не измерялся | не ревьюился |
| variant-lin-robust / variant-svc | deobf linear / LinearSVC | 0.99219 | не измерялся | не ревьюились |
| variant-cat-tuned / variant-cat-feat | CatBoost | 0.98425 / 0.98824 | не измерялся | не ревьюились |
| variant-trans-wide | byte+word transformer (88% train) | 0.99213 | не измерялся | не ревьюился |
| variant-trans-192 | byte transformer seq192 | 0.95276 | не измерялся | не ревьюился |
| baselines matching / boosting / transformer | словарь / CatBoost / byte ONNX | 0.86275 / 0.96525 / 0.95686 | не измерялись | не ревьюились |

Train-256 почти насыщен: 0.9961 соответствует одному неверному примеру из 256, поэтому различия 0.992–0.996 внутри шума. Public-1114 заметно труднее (все кандидаты 0.955–0.958) и служит основным свидетельством переноса.

## Выбранные финалисты (ровно 3)

### 1. variant-ens-lct-full — digest `d68a3d972b9a20bbe200ee4e43ef5a7ef448424e5822251aff71e3171e14a287`

- Состав: 0.2·linear(wordchar) + 0.4·CatBoost(tuned) + 0.4·byte+word transformer, обученный на всех 11 420 строках; порог 0.4888 и веса перенесены из train-holdout бленда.
- Свидетельства: лучший public F1 0.95755; train 0.99611 (128 TP / 1 FP / 127 TN / 0 FN); на 1371-строчном train holdout бленд дал F1 0.9956 против 0.9939 у лучшего компонента.
- Ревью: `board/reviews/variant-ens-lct-full.md`, verdict «допустим». Отмечено: порог/веса унаследованы от ens-lct, где holdout для full-трансформера является in-sample; это train-derived, не public/256-подгонка, но остаётся риск переноса порога.
- Причина выбора: лучший доступный public-результат и максимальное покрытие трёх механизмов; наиболее вероятный победитель private при сходстве private с public.

### 2. variant-lin-char — digest `4f4e39ef5b51f347b1e02a017a23bb45066b25352eae510e3a5c7e6bbe23ff3c`

- Состав: char-only 2–6 gram TF-IDF + LogisticRegression C=128, точный словарь 344 715 признаков, порог 1.1021 по class-balanced 5-fold OOF.
- Свидетельства: public F1 0.95714; train 0.99219 (127 TP / 1 FP / 127 TN / 1 FN); OOF balanced F1 0.99183. Ревью независимо воспроизвело все 256 ответов и порог из train.
- Ревью: `board/reviews/variant-lin-char.md`, verdict «допустим». Риски: near-duplicate кластеры в train завышают OOF/256; char-only хуже на части toxic; порог оптимизирован под 50/50.
- Причина выбора: лучший single-model public среди линейных, максимально быстрый и механически отличный от ансамблей (символьные n-граммы, без CatBoost/трансформера); полезен как страховка при сдвиге распределения.

### 3. variant-trans-wide-full — digest `575bf6d04504aa838423f13fcf92becb5d36cfc883ec1125136f9ab9176b0d4b`

- Состав: byte+hashed-word Transformer (seq 192, d_model 96, 2 слоя, INT8 ONNX), обучен на всех 11 420 строках, cutoff −0.00162439 перенесён из 88%-модели.
- Свидетельства: train 0.99611 (128 TP / 1 FP / 127 TN / 0 FN, TPR 1.0, FPR 0.0078); public F1 0.95524; локальный 256-замер после заморозки 0.99611; на общем holdout transformer-wide 0.9939 — лучший одиночный механизм.
- Ревью: `board/reviews/variant-trans-wide-full.md`, verdict «допустим». Риски: +0.004 к 88%-версии — это один пример; порог перенесён; финальная эпоха без early stopping; один новый benign FP.
- Причина выбора: независимый от словарей механизм с полным recall на train-256 и лучшей одиночной holdout-точностью; быстрое решение (0 токенов, ~0.9 с/пример в контейнере) даёт преимущество по правилу равенства F1.

## Почему не выбраны остальные

- `variant-ens-ct` и `variant-ens-lct` (public 0.95737) — почти те же CatBoost+transformer ядра, что у ens-lct-full; добавление второго такого кандидата дублировало бы механизм и не увеличивало бы шанс private-победителя.
- `variant-ens-all4` — лучший holdout (0.9962) и train 0.9961, но public не измерялся; по составу близок к ens-ct (два CatBoost-варианта с общими признаками).
- `variant-ens-rct` — train 0.9961, public не измерялся, механизм близок к ens-lct-full (замена линейного компонента на deobf-вариант).
- `variant-lin-robust`, `variant-svc` — train 0.9922, но без public-замера; близки к lin-char по пространству признаков.
- `variant-cat-tuned`, `variant-cat-feat` — train 0.9843/0.9882, ниже линейных и ансамблей.
- `variant-trans-192` — train 0.9528, существенно слабее.
- Бейзлайны — ниже всех финалистов по train; public не измерялся.

## Ресурсы

- Все решатели немодельные: solver_tokens=0 (полный учёт), judge_tokens=0; время решателя по train-замерам: lin-char ≈253 с, ens-lct-full ≈359 с, trans-wide-full ≈227 с на 256 примеров (включая запуск контейнера на пример). Подготовка кандидатов: частичный учёт токенов (`tokens_incomplete=true`) и измеренные секунды в журнале; точное разделение токенов по кандидатам внутри серии исследователя является аллокацией.
- Оценщик: судейские токены 0, время агрегатора ≈25–33 с на замер.

## Ограничения

- Public и private — закрытые разделы; известен только общий public score. Public-разрыв 0.955–0.958 против train-256 0.992–0.996 означает заметный сдвиг распределения; private может отличаться и от public.
- Train-256 насыщен (1–2 ошибки), поэтому не различает верхние кандидаты; отбор опирается на public и разнообразие.
- Пороговые переносы (OOF→full, 88%→100% transformer, holdout→full ensemble) проверены на train-holdout, но не на private.
- Риск переобучения на near-duplicate кластеры train признан в ревью; он ограничивает ожидания, но не делает кандидатов недопустимыми.
- Private-оценки и итоговый выбор оператора агенту не раскрываются.

## Замороженный пакет

Пакет из трёх кандидатов упакован командой `evaluator.client pack` из восстановленных неизменяемых снимков. Digest пакета: `c0cbb0c6a3ae71058aa290ba429b3eadc182aeb52a14903063344b972eacb133`. Журнал зафиксировал `finalize` в 2026-11-18T20:03:02Z с тем же `batch_digest`; новые предложения и проверки запрещены.
