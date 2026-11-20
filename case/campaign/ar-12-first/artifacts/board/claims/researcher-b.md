# claims — researcher-b (AR-12 F1, variants cat-tuned / cat-feat / ens-lc)

Role: researcher (variant family: CatBoost + ensemble). Data: `/workspace/data/train.jsonl`
(11 420 labeled, 7 564 positive / 3 856 negative). The 256-example `evaluation.arena.jsonl`
is used **only** as a final local measurement, never for training, feature selection,
threshold search or calibration.

Shared reference point: the frozen `variant-lin-wordchar` candidate (word 1-2 + char 3-5
TF-IDF, logistic C=4) has 5-fold OOF F1 **0.99169** at threshold 0.5636 on the same
`StratifiedKFold(5, shuffle, seed=42)` split (artefact `work/oof_lin.npy`). Any claim of
improvement below is measured against that.

## C1 — variant-cat-tuned

**Mechanism.** Harmful prompts are lexical *and* compositional: an obfuscation marker
(leetspeak, escapes, `SYSTEM` casing) matters only in combination with an instruction verb.
An additive linear model cannot represent those conjunctions; symmetric gradient-boosted
trees over word 1-3 and char 3-5 n-grams can split on both and therefore separate the
residual adversarial/encoding positives from benign look-alikes.

**Expected effect.** OOF F1 above the linear reference (0.9917), with the gain concentrated
in the `adversarial` / `encoding` / `jailbreak` classes. A threshold chosen directly for F1
on OOF (not the FPR<=5% rule) should add another small amount because the class balance is
66/34 and precision/recall are not symmetric.

**Scope.** Applies to every input; no example-specific rules. Only the model class, feature
space and threshold change relative to the shared baseline pipeline.

**Minimal change.** Same TF-IDF construction, CatBoost instead of logistic regression,
`boosting_type=Plain`, `border_count=16`, per-feature caps, threshold from OOF.

**Falsified if.** OOF F1 at the OOF-optimal threshold is <= 0.9917, or the apparent gain
disappears when the same threshold is applied to a held-out fold model (i.e. it is only
threshold drift, not ranking quality). Secondary check: OOF AUC must also exceed the linear
model's 0.99912.

## C2 — variant-cat-feat

**Mechanism.** Some signals are global statistics rather than n-grams: total length, digit /
uppercase / punctuation / control-character ratios, count of instruction-control words,
character repetition, escape-sequence density, long base64-like runs, leetspeak ratio. TF-IDF
normalises these away (L2 per document), so the tree model never sees "this text is 90 %
digits" or "this text repeats one token 40 times".

**Expected effect.** Better recall on `encoding`, `token_smuggling` and long `adversarial`
inputs at equal precision, i.e. OOF F1 above C1. If the numeric block adds nothing the
result is a measured negative and C1 stays the CatBoost representative.

**Falsified if.** OOF F1 <= C1's OOF F1.

## C3 — variant-ens-lc

**Mechanism.** The linear model and the tree model make different errors: the linear model
is strong on lexical evidence but blind to conjunctions, the tree model can overfit rare
n-grams and is weaker on very short inputs. Averaging their probabilities reduces variance
on the boundary examples where either alone flips.

**Expected effect.** OOF F1 >= max(component OOF F1), with fewer threshold-adjacent errors.
The 50/50 probability average is preferred a priori; a weighted/logit variant is only kept
if it beats 50/50 by more than 2e-4 OOF F1 (below that the difference is one example).

**Falsified if.** Blended OOF F1 < max(p_lin OOF F1, p_cat OOF F1).

## Shared risks / limits

- Threshold selected on 5-fold OOF is optimistically biased (11 420 points, ~1 example =
  0.0001 F1); the 256-example measurement has a binomial standard error around 0.004-0.008
  near F1 0.995, so 256-set differences below ~0.01 are not evidence.
- Train and the 256 subsample come from the same source; the 256 set is a final local
  measurement only, not an independent validation of the method.
- CatBoost training is CPU-bound; configs were chosen under a wall-clock budget, so the
  hyperparameter search is deliberately coarse and not exhaustive.
- No external data, no pretrained weights, no synthetic examples.
