# Pre-registration — `variant-trans-wide-full`

Written before the run; the measurement plan is fixed here so the 256 set cannot
influence any choice.

## Hypothesis

- **Mechanism.** The frozen `variant-trans-wide` was trained on 10 049 of 11 420
  train rows (88%); the remaining 1 371 rows were only used to pick epoch 7 and
  the INT8 cutoff. Adding those 1 371 rows (13.6% more data, including 908
  positives) to training should sharpen the decision boundary on the residual
  error classes (benign false positives, jailbreak/encoding/toxicity false
  negatives) without changing the model family.
- **Expected effect.** Small increase in ranking quality on unseen rows
  (holdout AUC 0.999505 is already near-ceiling, so the headroom is ~0.0005);
  most of the effect should appear as slightly better-calibrated logits, not as
  a large F1 jump. On the 256 diagnostic set the frozen candidate scores
  0.992126 (126/0/128/2); the plausible range for the full-data model is
  0.988–0.996.
- **Scope.** Same byte+hashing-word Transformer, same 8 epochs / lr 3e-3 / batch
  64 / wd 0.02 / pos_weight 1.1 / dropout 0.15 / seed 42. Only the training row
  set changes (10 049 -> 11 420). No new features, no ensembling.
- **Falsification.** The hypothesis is refuted if the full-data model's 256 F1
  is materially below the frozen 0.992126 (i.e. the extra rows, or the missing
  holdout-based epoch selection, hurt more than the added data helps). Because
  the cutoff is transferred unchanged, a calibration shift is a competing
  explanation for any F1 drop; it can be separated by comparing the score
  distributions and by reporting the diagnostic train-side oracle cutoff
  (reporting only — the shipped cutoff stays fixed).

## Fixed conditions

- Train rows: all 11 420 rows of `/workspace/data/train.jsonl`, labels as given.
- Cutoff: `-0.00162439`, taken from
  `candidates/variant-trans-wide/meta.json`; the exact float in that candidate's
  `model.json` is `-0.0016243904829025269` (same value to 8 decimals).
- No epoch selection: 8 epochs, final-epoch weights are packaged. The frozen
  candidate kept epoch 7, but epoch 8 had the identical holdout F1 (0.993939)
  and AUC 0.999496 vs 0.999498, so this is not a material change.
- `/workspace/data/evaluation.arena.jsonl` is read exactly once, after the
  package is frozen, by `run_eval.py` (one fresh process per example). It never
  feeds training, cutoff, epoch or hyper-parameter choice.
- Export: dynamic INT8 (QInt8, MatMul+Gemm), onnxruntime + numpy solver.
