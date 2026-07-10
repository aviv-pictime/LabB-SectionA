# Section A — Milestone Summary

_Last updated: 2026-07-10._ Condensed view of the active-learning journey. Full experiment log with rationale for every variant is in `process.md`.

**Goal:** maximize mean F1(Left) on a fixed RandomForest. 500 free labels + ≤5,000 oracle queries, ≤60s/seed, seeds {1,2,3}. Gate at 0.55 for guaranteed points.

---

## The milestones that WORKED (the path to best)

| Step | Change | Mean F1 | Gain | Why it worked |
|---|---|---|---|---|
| Baseline | Train on 500 free labels only | 0.4068 | — | Reference. Confirms severe class imbalance (~15% Left). |
| Random 5,000 | Spend full budget on random IDs | 0.5481 | +0.141 | "More data" alone clears most of the gap. Floor for any smart method. |
| Uncertainty | RF scorer, query top-500 by \|p−0.5\| each iter | 0.5846 | +0.036 | Querying near the decision boundary beats random. |
| Minority pseudo | Add pool rows with P(Left) ≥ 0.7 as Left=1 (minority-only) | 0.6222 | +0.038 | Biggest single jump. Over-predicting Left is safe for F1(Left); threshold must sit in RF's real proba range (caps ~0.9), not a nominal 0.95. |
| QBC queries | Committee RF + LR + HGB; query by std of P(Left) | 0.6251 | +0.003 | Cross-family disagreement finds more informative points than one model. |
| Consensus pseudo | Pseudo gated by committee *mean* P(Left) ≥ 0.7 | 0.6364 | +0.011 | Agreement across 3 families filters single-model overconfidence. Disagreement→queries, agreement→pseudo. |
| Minority upweight 3× | Duplicate real Left rows 3× in training | 0.6562 | +0.020 | Pushes training to ~69% Left → shifts the RF's fixed 0.5 threshold toward high recall. |
| Stratified queries | Reserve 20% of each batch for highest-P(Left) minority hunt | **0.6575** | +0.0013 | Oracles +26% more real Left (36%→45%). Small F1 gain (saturated) but fixes the weak seed; mechanism verified, not noise. |

**Final: mean F1 = 0.6575**, all seeds 0.655–0.660, ~13s/seed. +0.25 over baseline, +0.10 over the 0.55 gate.

**Unifying theme:** every win was class-imbalance handling. Because the RF's decision threshold is fixed at 0.5 and unmodifiable, the only lever is training-data composition — and F1(Left) wants the model biased toward predicting Left. Minority pseudo, minority upweight, and stratified minority-hunt all push in that one direction.

---

## The milestones that DID NOT work (and why)

| Idea | Result | Why it failed |
|---|---|---|
| Symmetric pseudo (add confident Stayed too) | −0.010 | Wrong pseudo-Stayed labels cut Left recall; confident Stayed is low-info; undoes the minority skew. |
| Pseudo threshold 0.8 / 0.95 | −0.02 to flat | Committee-mean rarely reaches 0.8 (RF proba caps ~0.9) → pseudo pool empties. |
| Pseudo threshold 0.6 | −0.003 | Too permissive → noisy pseudo-Left. (0.7 is the peak.) |
| Downweight pseudo (real 2×/pseudo 1×; pseudo 2×) | −0.006 to −0.009 | The 3:1 real:pseudo ratio is already optimal; shifting either way loses signal or amplifies noise. |
| Minority upweight 4× | −0.008 | Over-skews → model over-predicts Left → precision collapses. (3× is the peak.) |
| Target-ratio oversampling (fractional) | −0.008 | Stochastic resampling adds variance; deterministic integer 3× is cleaner. |
| k-means batch diversity | ~0 | 500 clusters over 2,000 points fragments; QBC already spreads picks. |
| Density-aware (prefer sparse) | −0.005 | Decision boundary lives in the *dense* core, not the sparse halo. |
| Information Density (prefer dense) | −0.008 | Any geometric prior contaminates the QBC signal; PCA space ≠ decision-relevant space. |
| Query by Bagging (100 RF trees) | −0.012 | Trees in one RF are correlated → their disagreement is training noise, not real ambiguity. |
| Bigger committee (+KNN / +MLP / +ExtraTrees, 4–6 members) | −0.003 to −0.012 | Extra members dilute rather than sharpen; MLP's polarized probas dominate std. |
| SVM-RBF committee member | −0.002 + 57s | No signal gain, and runtime nearly hit the 60s cap. |
| Alt disagreement measures (BALD, KL, vote entropy) | −0.005 to −0.010 | Std of P(Left) across 3 members is already best; others over-weight extremes or drop magnitude. |
| Warm-start (random first batch) | −0.011 | Free initial 500 is already diverse; QBC's first iter beats random. |
| Iteration-persistent pseudo | −0.004 | Requiring a 2-iter streak discards too many still-correct pseudo-Left. |
| Decouple final vs scorer upweight | −0.002 to −0.004 | Final model also peaks at 3× independently. |
| Post-hoc self-training | −0.001 to −0.003 | Final model is 69%-Left-biased → inflated P(Left) → noisy pseudo. |
| Committee calibration (isotonic) | −0.034 + slow | Calibrates toward the artificial 69%-Left prior — wrong direction. |
| Hyperparameter sweeps (batch, LR C, HGB lr, RF n_est, pseudo start/cap) | all ≤ 0 | Every knob already at its optimum. |

**Meta-lesson:** ~40 variants tested; everything outside the imbalance theme was neutral or negative. The strategy sits at a robust local optimum. The one subtle call — a +0.0013 "win" that first looked like noise — was kept only after a mechanistic check proved the effect was real (measured Left-yield jump), not the score alone.
