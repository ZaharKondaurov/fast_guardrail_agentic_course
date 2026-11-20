# researcher-a — claims (AR-12, campaign ar-12-first)

Role: researcher. Workdir: `/workspace/campaign/work/researcher-a`. Candidates: `variant-lin-char`,
`variant-lin-robust`, `variant-svc`. All selection/tuning below uses only 5-fold stratified OOF on
`/workspace/data/train.jsonl` (11420 rows, seed 42). The 256-row `evaluation.arena.jsonl` is used
only once per candidate, after freezing, as a local measurement.

## Shared method

- Features: deterministic text views; per view word unigrams/bigrams and/or character n-grams over
  whitespace-collapsed NFKC-lowercased text, char stream padded with one space. TF-IDF with
  sublinear TF and per-view L2 normalisation; multi-view vectors are concatenated and scaled by
  1/sqrt(non-empty views), so a clean text seen through two identical views keeps single-view geometry.
- Model: scikit-learn linear classifier on the fold-train vocabulary/IDF only (no leakage).
- Threshold: the exported solver compares the raw decision score against one scalar. The threshold is
  selected on pooled OOF scores by **class-balanced F1** (positive weight 1/(2*n_pos), negative
  1/(2*n_neg)). Reason: train prevalence is 66.2% positive while campaign.md fixes the local 256
  measurement split at 128/128; the balanced criterion corrects the prior shift without touching any
  held-out label. Plain (unweighted) OOF F1 at its own optimum is also reported for reference.
- Model/feature choice rule: highest class-balanced OOF F1, tie-break by plain OOF F1, then by number
  of features (smaller package). The 256 split is never used to choose features, C, alpha or threshold.

## variant-lin-char — character-only n-grams (2–6), TF-IDF + logistic regression

- Mechanism: prompt attacks in this task are often carried by surface form (delimiters, leet digits,
  homoglyph scripts, role-play scaffolding, token-smuggling markers) rather than by word identity.
  Character 2–6 grams model those sub-word patterns and are insensitive to word segmentation and to
  intra-word typos that word features miss.
- Expected effect: competitive OOF F1 (target ≥0.985), with an error profile shifted toward semantic /
  toxicity confusions and away from obfuscation and spelling errors, i.e. useful diversity for the
  finalist set.
- Scope: full train distribution; one model, one feature block; threshold as above.
- Falsified if: balanced OOF F1 is materially below the word+char baseline (<0.98) or its error set is
  a superset of the baseline's (no diversity value).
- Sweep: char ranges 2–5 / 2–6 / 3–6, min_df 2, C ∈ {2, 4, 8}; choose on OOF.

## variant-lin-robust — normalized + deterministic deobfuscated views, word+char, logistic regression

- Mechanism: encoding attacks in train hide intent behind leet substitution, Greek/Cyrillic homoglyphs,
  zero-width/format characters, backslash escaping and repeated-character padding. A deterministic
  deobfuscated copy (`NFKC → drop Cf/Cc/Cs/Co/Cn → strip backslash before non-ASCII → homoglyph fold →
  leet fold → collapse 3+ repeated chars → collapse whitespace/lower`) restores the hidden words, so
  word+char TF-IDF over that view can match attack phrases that the raw view cannot. Keeping the raw
  view as a second block preserves the obfuscation signal itself (obfuscated benign "control" prompts
  are labelled 1 in train, so erasing obfuscation alone would hurt).
- Expected effect: recall gain on encoding / token_smuggling / obfuscated control examples at a small
  precision cost on clean text; balanced OOF F1 at or above the single-view baseline.
- Scope: same classifier family, doubled feature space; threshold as above.
- Falsified if: balanced OOF F1 ≤ single-view OOF F1 on the same folds, or obfuscated-class recall does
  not improve.
- Sweep: C ∈ {2, 4, 8}, char 3–5 word 1–2 on both views, min_df 2.

## variant-svc — linear SVM / modified_huber SGD on word+char TF-IDF, OOF score threshold

- Mechanism: a max-margin (or Huberised hinge) objective weights only boundary examples, unlike the
  logistic log-loss which is driven by all points; on this near-separable task it should land on a
  different boundary and produce a different error set, and its margin threshold is selected on OOF
  scores rather than assumed at 0.
- Expected effect: balanced OOF F1 within ~0.003 of logistic regression, error set differing on the
  benign/edge-case vs jailbreak/toxic boundary; useful as a mechanism-diverse finalist.
- Scope: same word+char TF-IDF block as the baseline; LinearSVC C ∈ {0.25, 0.5, 1, 2} and
  SGDClassifier(modified_huber) alpha ∈ {1e-4, 1e-5}; pick by OOF.
- Falsified if: balanced OOF F1 is materially lower (<0.98) and its errors are a subset of the
  logistic regression error set (then it adds no diversity).

## Risks / notes

- Many train rows are near-duplicate clusters that differ only by trailing attack text or by label;
  a linear bag-of-ngrams model partly memorises cluster-specific phrases. OOF is grouped only by
  random fold, not by cluster, so OOF F1 is optimistic for truly unseen phrasings. All three
  candidates share this limitation; it is reported in findings.
- The 256 measurement split is drawn from the same train pool (128/128), so its F1 is a local estimate,
  not an independent-distribution guarantee.
