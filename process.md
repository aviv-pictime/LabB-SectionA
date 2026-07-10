# Process Log — Section A (Active Learning)

Tracking strategy experiments and their F1(Left) results.

**Target:** mean F1(Left) ≥ 0.55 (guarantees 30/45 functional pts) — competitive placement above that.

**Budget per seed:** 500 free initial + up to 5,000 oracle queries. Runtime ≤ 60s per seed. Classifier fixed (RandomForest, `n_estimators=100`, `random_state=seed`).

---

## Status — 2026-07-10 (session close)

**Best strategy: v29 — mean F1(Left) = 0.6575**, all seeds 0.655–0.660, ~13s/seed (well under the 60s cap).

Recipe: QBC committee (RF + LR + HGB) queries by disagreement std, with 20% of each batch reserved for a highest-P(Left) minority hunt; minority-only committee-consensus pseudo-labels (mean P(Left) ≥ 0.7, refreshed each iteration from the halfway point); real Left rows upweighted 3× in all training.

~40 variants tested across queries, pseudo-labeling, committee composition, geometry/diversity, disagreement measures, calibration, self-training, and hyperparameters. Every win was **class-imbalance handling**; everything else was neutral or negative — strong evidence of a robust local optimum. Condensed view in `process_summary.md`.

Next session: Section B (GraphSAGE on Cora) — not yet started.

---

## Summary table

| # | Method (short) | Seed 1 | Seed 2 | Seed 3 | Mean F1 | Δ vs current best at that point | Runtime | Kept? |
|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (initial 500) | 0.4198 | 0.4009 | 0.3998 | **0.4068** | — | 0.1s | — |
| 1 | Random 5,000 | 0.5531 | 0.5289 | 0.5623 | **0.5481** | +0.141 | 3.4s | — |
| 2 | RF uncertainty, batch 500 | 0.5810 | 0.5796 | 0.5932 | **0.5846** | +0.036 | 6s | — |
| 3 | Same, batch 250 | 0.5753 | 0.6040 | 0.5828 | 0.5874 | +0.003 | 8s | reverted |
| 4 | + symmetric pseudo p≥0.95 end-only | 0.5667 | 0.5987 | 0.5839 | 0.5831 | −0.004 | 8s | ❌ |
| 5 | Symmetric pseudo p≥0.95, mid-loop | 0.5797 | 0.5799 | 0.5948 | 0.5848 | +0.000 | 6s | ❌ |
| 6a | Minority-only pseudo p≥0.95 | 0.5810 | 0.5796 | 0.5932 | 0.5846 | +0.000 | 6s | ❌ (empty pool) |
| **6** | **Minority-only pseudo p≥0.7** | 0.6229 | 0.6156 | 0.6283 | **0.6222** | +0.038 | 6s | ✅ |
| 7 | v6 + real duplicated 2× | 0.6230 | 0.6122 | 0.6139 | 0.6164 | −0.006 | 6s | ❌ |
| 8 | + QBC queries (RF+LR+HGB) | 0.6184 | 0.6250 | 0.6319 | 0.6251 | +0.003 | 13s | ✅ |
| **9** | **+ consensus pseudo** | 0.6311 | 0.6341 | 0.6441 | **0.6364** | +0.011 | 14s | ✅ |
| 10 | + k-means diversity | 0.6230 | 0.6371 | 0.6495 | 0.6365 | +0.000 | 22s | ❌ |
| 11 | + density-aware selection | 0.6212 | 0.6317 | 0.6419 | 0.6316 | −0.005 | 13s | ❌ |
| 12a | Pseudo threshold 0.6 | 0.6252 | 0.6335 | 0.6429 | 0.6339 | −0.003 | 12s | ❌ |
| 12b | Pseudo threshold 0.8 | 0.6059 | 0.6170 | 0.6218 | 0.6149 | −0.021 | 12s | ❌ |
| 13a | + minority upweight 2× | 0.6370 | 0.6546 | 0.6551 | 0.6489 | +0.012 | 13s | ✅ |
| **13** | **+ minority upweight 3× — current best** | **0.6502** | **0.6600** | **0.6584** | **0.6562** | +0.007 | 13s | ✅ |
| 13c | + minority upweight 4× | 0.6414 | 0.6457 | 0.6575 | 0.6482 | −0.008 | 14s | ❌ |
| 14 | v13 + SVC(RBF) as 4th committee member | 0.6427 | 0.6615 | 0.6596 | 0.6546 | −0.002 | ~57s | ❌ (runtime cliff) |
| 15 | v13 + pseudo upweight 2× | 0.6349 | 0.6504 | 0.6577 | 0.6477 | −0.009 | 14s | ❌ |
| 16 | v13 + warm-start (iter 0 random) | 0.6380 | 0.6516 | 0.6466 | 0.6454 | −0.011 | 13s | ❌ |
| 17 | v13 + 2-RF committee (sqrt + log2) | 0.6429 | 0.6595 | 0.6635 | 0.6553 | −0.001 | 17s | ❌ |
| T1 | BATCH_SIZE=250 | 0.6462 | 0.6491 | 0.6594 | 0.6516 | −0.005 | 27s | ❌ |
| T2 | BATCH_SIZE=1000 | 0.6363 | 0.6509 | 0.6620 | 0.6497 | −0.007 | 9s | ❌ |
| T3 | PSEUDO_START_FRACTION=0.3 | 0.6425 | 0.6527 | 0.6594 | 0.6515 | −0.005 | 13s | ❌ |
| T4 | PSEUDO_START_FRACTION=0.7 | 0.6442 | 0.6560 | 0.6540 | 0.6514 | −0.005 | 12s | ❌ |
| T5 | LR C=0.5 (more regularization) | 0.6409 | 0.6542 | 0.6585 | 0.6512 | −0.005 | 12s | ❌ |
| T6 | LR C=2.0 (less regularization) | 0.6502 | 0.6442 | 0.6472 | 0.6472 | −0.009 | 12s | ❌ |
| T7 | HGB learning_rate=0.05 | 0.6478 | 0.6540 | 0.6606 | 0.6541 | −0.002 | 12s | ❌ |
| T8 | HGB learning_rate=0.2 | 0.6457 | 0.6448 | 0.6652 | 0.6519 | −0.004 | 13s | ❌ |
| T9 | Scorer RF n_estimators=300 | 0.6433 | 0.6601 | 0.6569 | 0.6534 | −0.003 | 19s | ❌ |
| T10 | PSEUDO_CAP_PER_CLASS=200 | 0.6294 | 0.6447 | 0.6483 | 0.6408 | −0.015 | 13s | ❌ |
| 18a | Information Density (α=0.7, prefer dense) | 0.6422 | 0.6461 | 0.6579 | 0.6487 | −0.008 | 15s | ❌ |
| 18b | Information Density (α=0.9, tiny density influence) | 0.6448 | 0.6633 | 0.6514 | 0.6532 | −0.003 | 14s | ❌ |
| 19 | Iteration-persistent pseudo (require ≥2 consecutive iters ≥ threshold) | 0.6377 | 0.6581 | 0.6600 | 0.6520 | −0.004 | 15s | ❌ |
| 20a | Disagreement = BALD | 0.6454 | 0.6581 | 0.6495 | 0.6510 | −0.005 | 12s | ❌ |
| 20b | Disagreement = KL-to-consensus (≡ BALD for Bernoulli) | 0.6454 | 0.6581 | 0.6495 | 0.6510 | −0.005 | 12s | ❌ |
| 20c | Disagreement = vote entropy (hard votes) | 0.6372 | 0.6455 | 0.6553 | 0.6460 | −0.010 | 13s | ❌ |
| 21 | Query by Bagging (100 RF trees as committee for queries) | 0.6354 | 0.6420 | 0.6559 | 0.6444 | −0.012 | 22s | ❌ |
| 22 | Confidence-weighted pseudo (p≥0.85 doubled) | 0.6429 | 0.6596 | 0.6579 | 0.6535 | −0.003 | 14s | ❌ |
| 23a | Combined score: `std × (1 + 1.0·mean_P_Left)` | 0.6435 | 0.6511 | 0.6555 | 0.6500 | −0.006 | 12s | ❌ |
| 23b | Combined score: `std × (1 + 0.5·mean_P_Left)` | 0.6457 | 0.6606 | 0.6445 | 0.6503 | −0.006 | 13s | ❌ |
| 24 | Adaptive pseudo threshold (0.8 → 0.65 over iters) | 0.6418 | 0.6516 | 0.6565 | 0.6500 | −0.006 | 13s | ❌ |
| 25a | v13 + KNN (4 members) | 0.6443 | 0.6450 | 0.6474 | 0.6456 | −0.011 | 14s | ❌ |
| 25b | v13 + MLP (4 members) | 0.6372 | 0.6461 | 0.6488 | 0.6441 | −0.012 | 27s | ❌ |
| 25c | v13 + ExtraTrees (4 members) | 0.6445 | 0.6517 | 0.6628 | 0.6530 | −0.003 | 16s | ❌ |
| 25d | v13 + MLP + ExtraTrees (5 members) | 0.6443 | 0.6527 | 0.6559 | 0.6510 | −0.005 | 28s | ❌ |
| 25e | v13 + KNN + MLP + ExtraTrees (6 members) | 0.6538 | 0.6556 | 0.6494 | 0.6529 | −0.003 | 29s | ❌ |
| 26 | Symmetric pseudo at 0.7 threshold — pseudo-Left (P≥0.7) AND pseudo-Stayed (P≤0.3), Stayed count capped to match Left | 0.6374 | 0.6446 | 0.6557 | 0.6459 | −0.010 | 13s | ❌ |
| 27 | Target-ratio oversampling: resample real Left (with replacement) to exact Left fraction. Peak at 0.70. | 0.6404 | 0.6503 | 0.6547 | 0.6485 | −0.008 | 13s | ❌ |
| **29** | **Stratified query batch (f=0.20): reserve 20% of each batch for highest-P(Left) minority hunt — BEST** | **0.6550** | **0.6572** | **0.6603** | **0.6575** | **+0.0013** | 13s | ✅ |
| 30 | Decouple final-model upweight from scorer (scorer 3×, final 4×/5×/6×) | — | — | — | 0.6534–0.6555 | −0.002 to −0.004 | 13s | ❌ |
| 31 | Post-hoc self-training (final RF pseudo-labels, 1–2 rounds) | 0.6540 | 0.6555 | 0.6591 | 0.6562 | −0.0013 | 14s | ❌ |
| 32 | Committee calibration (isotonic CalibratedClassifierCV, cv=3) | 0.6240 | 0.6213 | 0.6267 | 0.6240 | −0.034 | 28s | ❌ |
| 33 | QBC with bootstrap-RF committee (5 RFs on bootstrap resamples) | 0.6403 | 0.6412 | 0.6537 | 0.6451 | −0.012 | 13s | ❌ |
| 34a | Distance-to-labeled diversity blend, α=0.7 | 0.6432 | 0.6522 | 0.6580 | 0.6511 | −0.006 | 14s | ❌ |
| 34b | Distance-to-labeled diversity blend, α=0.9 | 0.6461 | 0.6548 | 0.6646 | 0.6552 | −0.002 | 12s | ❌ |
| 35 | Farthest-first / Core-Set selection (pure geometric coverage) | 0.6430 | 0.6457 | 0.6501 | 0.6462 | −0.011 | 27s | ❌ |

**Current best: v29** — v13 + stratified query batch (reserve 20% of each batch for minority hunt). Full recipe: QBC (3 models: RF + LR + HGB) queries by disagreement std with 20% reserved for highest-P(Left); minority-only consensus pseudo (threshold 0.7, refreshed each iter from halfway); real Left upweight 3×. Mean F1 = **0.6575**, all seeds ≥ 0.655, runtime ~13s/seed.

**v13** (predecessor best) — same but pure-disagreement queries (no stratified reserve). Mean F1 = 0.6562.

---

## Methods, detailed

### V0 — Baseline

- **What**: Train the final RF only on the free initial 500 labeled rows. No oracle calls.
- **Why**: Establish the floor and confirm the class-imbalance shape of the problem.
- **Result**: 0.4068. F1(Left) is much lower than accuracy would suggest — Left is the ~15% minority.
- **Kept?** No, only used as reference.

### V1 — Random 5,000

- **What**: Spend the full oracle budget on 5,000 uniformly random IDs, retrain once.
- **Why**: Isolate "more data alone" from "smart selection". Any smart strategy has to beat this.
- **Result**: 0.5481. Massive +0.14 jump; almost clears the 0.55 gate but not quite.
- **Kept?** No — sets the floor for AL strategies.

### V2 — RF uncertainty sampling, batch 500

- **What**: 10 iterations × 500 queries. Each iteration: train RF on current labeled, score every unlabeled pool row's P(Left=1), pick top-500 by `|p−0.5|`. RF-scorer matches the final model.
- **Why**: Classical AL baseline. Using RF-as-scorer keeps the notion of "uncertain" consistent with the deployed model.
- **Result**: 0.5846. All seeds ≥ 0.58. Seed 2 (previously the drag at 0.529) jumped to 0.580.
- **Kept?** Yes, as the starting AL scaffold.

### V3 — Batch 250

- **What**: Same as v2 but batch=250 (20 iters), refreshing uncertainty more often.
- **Why**: Test whether more frequent retraining picks better queries.
- **Result**: 0.5874, +0.003. Marginal improvement; seed 1 dropped slightly.
- **Kept?** No — kept batch 500 for simplicity.

### V4 — Symmetric pseudo p≥0.95, end-only

- **What**: After the 10 AL iters, pseudo-label remaining pool rows with p≥0.95 (Left=1) or p≤0.05 (Stayed=0), cap 2,000/class. Retrain final RF on labeled + pseudo.
- **Why**: Cheap use of pool data; assumes RF's "confident" predictions are correct.
- **Result**: 0.5831, −0.004. Consistently loses on 2/3 seeds.
- **Interpretation**: RF `predict_proba` is overconfident; false-positive pseudo-Stayed labels suppress recall on Left. Diagnostic later showed *zero* pseudo-Left labels ever populated because RF's max P(Left) tops out at ~0.85–0.91 (leaf-averaging cap with imbalanced training).
- **Kept?** No.

### V5 — Symmetric pseudo p≥0.95, refreshed each iter from #5

- **What**: Same threshold/cap but pseudo pool rebuilt each iteration from the current model, starting at iter 5. Pseudo included in scorer training and final model.
- **Why**: Test whether mid-loop refresh (more iterations of pseudo) helps over end-only.
- **Result**: 0.5848, essentially neutral.
- **Interpretation**: Threshold 0.95 still excludes most rows; the ones it does add are dominated by high-confidence Stayed predictions, mirroring v4's problem.
- **Kept?** No.

### V6 — Minority-only pseudo, threshold 0.7 (BIG WIN)

- **What**: Pseudo-label only `Left=1` predictions (skip pseudo-Stayed entirely); threshold lowered to p≥0.7 to actually hit the achievable RF-proba range; cap 2,000; refreshed each iter from #5.
- **Why**: Two hypotheses — (1) F1(Left) errors are asymmetric: over-predicting Left hurts less than under-predicting; (2) threshold must sit inside RF's actual proba range, not a nominal "high confidence" value.
- **Result**: 0.6222, **+0.038** over v2 — biggest single gain until this point.
- **Interpretation**: Confirmed both hypotheses. Adding minority pseudo-labels rebalances training toward Left and expands the minority signal without adding majority-class noise.
- **Kept?** Yes.

### V7 — v6 + real labels duplicated 2×

- **What**: Duplicate every real (initial + oracle) labeled row so real:pseudo effective ratio = 2:1.
- **Why**: Explicit downweighting of the (potentially noisy) pseudo channel.
- **Result**: 0.6164, −0.006. Downweighting pseudo *hurts*.
- **Interpretation**: The p≥0.7 minority pseudo signal is trustworthy enough that reducing its influence loses more signal than noise.
- **Kept?** No.

### V8 — QBC queries (RF + LR + HGB)

- **What**: Committee of three model families — RandomForest, Logistic Regression (with StandardScaler), HistGradientBoosting. Query score = standard deviation of P(Left=1) across the 3 members.
- **Why**: Different inductive biases → different notions of "the decision boundary" → potentially more informative disagreement than single-model uncertainty. RF alone might miss uncertainty that shows up in linear or boosted-tree boundaries.
- **Result**: 0.6251, +0.003. Small consistent gain (all seeds moved slightly).
- **Kept?** Yes — modest but positive.

### V9 — QBC + consensus pseudo (big compound win)

- **What**: Keep QBC queries. For pseudo, use committee **mean** P(Left) instead of RF-alone P(Left), same p≥0.7 threshold.
- **Why**: If disagreement finds informative queries, *agreement* should find reliable pseudo-labels — a pseudo-Left row is trustworthy only if RF, LR, and HGB all lean Left.
- **Result**: 0.6364, +0.011. **All seeds > 0.63 for the first time.**
- **Interpretation**: Consensus filters single-model overconfidence; QBC is used in two complementary ways within the same loop (disagreement for queries, agreement for pseudo).
- **Kept?** Yes — foundational for everything after.

### V10 — v9 + k-means batch diversity

- **What**: Take top-2,000 by QBC disagreement, cluster into 500 groups with MiniBatchKMeans on standard-scaled features, pick the highest-disagreement point per cluster.
- **Why**: Even if QBC picks informative points, the top-500 might be near-duplicates in feature space. Diversity should spread the batch.
- **Result**: 0.6365, essentially flat (+0.0001). Seeds 2 & 3 gained slightly; seed 1 lost 0.008.
- **Interpretation**: 500 clusters over 2,000 points fragments into avg-4-point clusters — noisy centroids. Additionally, QBC across 3 model families already naturally spreads picks across different boundary regions. Runtime +8s for no reliable gain.
- **Kept?** No.

### V11 — v9 + density-aware selection

- **What**: `score = 0.7 · qbc_norm + 0.3 · density_norm`, where density = mean distance to 10 nearest neighbors in scaled feature space (higher = sparser). Density precomputed once on the initial remaining pool.
- **Why**: The PCA-density plot showed one dense core + sparse halo. If QBC over-picks from the crowded core, biasing toward sparse regions might recover overlooked informative points.
- **Result**: 0.6316, −0.005. All seeds regressed.
- **Interpretation**: **PCA-space density ≠ information structure for this problem.** The decision boundary evidently lives *inside* the dense core (that's where classes actually mix); pulling toward the sparse halo drifts into regions RF already predicts confidently.
- **Kept?** No.

### V12a / V12b — Pseudo threshold sweep

- **What**: v9 with `PSEUDO_THRESHOLD` set to 0.6 (12a) and 0.8 (12b).
- **Why**: v9 used 0.7; test whether more (0.6) or fewer (0.8) pseudo-labels help.
- **Result**: 0.6339 (−0.003) and 0.6149 (−0.021). Clear U-shape with 0.7 as the peak.
- **Interpretation**: 0.6 adds too much noise; 0.8 rarely populates because the committee-mean cap is around ~0.85 (RF proba drags the mean down). 0.7 is close to optimal.
- **Kept?** No — confirmed v9's threshold choice.

### V13 — v9 + minority upweight (CURRENT BEST)

- **What**: In every training set (scorer + final), duplicate real `Left=1` rows N times. Sweep tried N = 2, 3, 4.
- **Why**: F1(Left) is minority-only; the RF/LR/HGB bootstrap sees ~15% Left → over-predicts Stayed by default. Explicit oversampling nudges each model to weight the minority class more.
- **Result**:
  - 2× → 0.6489 (+0.012 vs v9)
  - **3× → 0.6562 (+0.007 more)** ← peak, current best
  - 4× → 0.6482 (−0.008 vs 3×) — model over-predicts Left, precision drops.
- **Interpretation**: Clean peak at 3×. Minority upweighting is the biggest single win since consensus pseudo, and it's basically free (just row duplication).
- **Kept?** Yes.

### V14 — v13 + SVC(RBF) as 4th committee member

- **What**: Add `SVC(kernel='rbf', probability=False)` with sigmoid on `decision_function` to the QBC committee.
- **Why**: SVM's non-linear geometry is complementary to RF, LR, HGB; kernel margin is a natural boundary-focused signal.
- **Result**: 0.6546, −0.002. Also runtime blew from 13s to **~57s per seed** (RBF SVM is O(N²)–O(N³); scoring 9k+ points/iter compounds).
- **Interpretation**: Signal-wise the existing 3-family committee already captures the useful disagreement. Runtime-wise it's a hard no — too close to the 60s grading timeout to trust across different hardware.
- **Kept?** No.

### V15 — v13 + pseudo upweight 2×

- **What**: Duplicate pseudo Left rows 2× (real Left still 3×; ratio real:pseudo shifts from 3:1 to 3:2).
- **Why**: If real minority upweight worked, maybe pseudo Left is also underweighted.
- **Result**: 0.6477, −0.009. All seeds lose.
- **Interpretation**: Pseudo carries noise (~15–20% mislabel rate at threshold 0.7); higher weight amplifies the noise. Together with v7 (real 2× at 2:1 hurt too), we've now bracketed the sweet spot at **real:pseudo = 3:1**.
- **Kept?** No.

### V16 — v13 + warm-start (iter 0 random)

- **What**: First iteration uses random-500 instead of QBC; iters 1–9 unchanged.
- **Why**: Seed 1 was the persistent laggard; hypothesis was that QBC's first-iter picks were biased and a random seed batch would give more diverse coverage.
- **Result**: 0.6454, −0.011. All seeds regressed, including seed 1.
- **Interpretation**: The free initial 500 already provides enough diverse coverage; QBC's first iter is more informative than random even without any oracle history. Warm-start just wastes 500 queries on random picks.
- **Kept?** No.

### V17 — v13 + broadened committee (2 RFs with different max_features)

- **What**: Committee = RF(max_features='sqrt') + RF(max_features='log2') + LR + HGB. Cheap way to expand from 3 to 4 members without SVM's runtime cost.
- **Why**: Larger committee → potentially finer disagreement signal.
- **Result**: 0.6553, essentially flat (−0.001). Seed 1 lost 0.007; seed 3 gained 0.005.
- **Interpretation**: Two RFs (both tree-ensembles) are too similar to add real inductive-bias diversity. Their disagreement mostly mirrors the RF-vs-others axis already covered. Runtime +4s for nothing.
- **Kept?** No.

### V18 — Information Density (inverse of v11)

- **What**: Query score = `α · qbc_norm + (1−α) · repr_norm`, where `repr_norm = 1 − minmax(density_arr)` — inverts v11's density signal so we prefer *dense/representative* points instead of sparse ones. Tested α = 0.7 and 0.9.
- **Why**: v11 (prefer sparse) hurt and the density plot showed the decision boundary lives inside the dense core. Information Density is a classical AL method (Settles 2008) that combines uncertainty with typicality — should theoretically favor "informative *and* typical" points.
- **Result**: α=0.7 → 0.6487 (−0.008); α=0.9 → 0.6532 (−0.003). Both hurt.
- **Interpretation**: Neither direction of density weighting helps. The QBC disagreement signal *by itself* already lives on the decision boundary (which is where classes mix, whether that's a dense or sparse region). Adding any geometric prior contaminates that signal.
- **Kept?** No.

### V19 — Iteration-persistent pseudo (my idea)

- **What**: Track which Employee IDs had committee-consensus P(Left) ≥ 0.7 in the previous iteration. Pseudo-label only rows that were high in **both** the current and previous iteration (a "streak of 2" persistence requirement).
- **Why**: Committee scores flicker as the model updates each iter — borderline rows cross the 0.7 threshold in one direction then the other. Requiring persistence should filter out these unstable rows and leave a cleaner pseudo set.
- **Result**: 0.6520, −0.004. All seeds either flat or regressed; seed 1 lost 0.013.
- **Interpretation**: The persistence filter throws away *too many* useful pseudo-labels. Many rows that flicker in/out of ≥ 0.7 are still directionally correct (their true label is Left); enforcing streak = 2 cuts the pseudo pool roughly in half, and losing that volume hurts more than the noise reduction gains.
- **Kept?** No.

### V20 — Alternative QBC disagreement measures

- **What**: Swap the query score from `std(P(Left))` to three alternatives:
  - **20a — BALD**: `H(mean p_i) - mean(H(p_i))` for Bernoulli. Prefers rows where the ensemble is uncertain (`p_bar ≈ 0.5`) but individual members are internally confident — targets *epistemic* uncertainty.
  - **20b — KL-to-consensus**: `mean(KL(p_i || p_bar))`. Weights disagreements near probability extremes more heavily than std does.
  - **20c — Vote entropy**: convert each `p_i` to a hard vote (`p > 0.5`), then entropy of the vote distribution — the classical QBC formulation.
- **Why**: Test whether the specific choice of disagreement measure matters. Std, variance, and range are equivalent for ranking; BALD, KL, and vote entropy are genuinely different.
- **Results**:
  - BALD → 0.6510 (−0.005)
  - KL-to-consensus → 0.6510 (−0.005) — **identical to BALD** (mathematical identity for Bernoulli: `mean KL(p_i || p_bar) ≡ H(p_bar) − mean(H(p_i))`)
  - Vote entropy → 0.6460 (−0.010)
- **Interpretation**: **Std remains the best measure** for our 3-model binary committee. BALD/KL emphasize disagreements more sharply near the extremes — apparently that emphasis pulls toward points that are marginally less useful here. Vote entropy loses probability magnitude entirely and picks worse queries.
- **Kept?** No. Confirms that std is well-suited for our setup.

### V21 — Query by Bagging (RF's 100 trees as committee)

- **What**: Replace the 3-model QBC disagreement score with std of P(Left=1) across the 100 individual decision trees inside the RandomForest (`rf.estimators_[i].predict_proba`). Committee balloons from 3 → 100.
- **Why**: Larger, higher-resolution committee should give a smoother disagreement signal — classical AL technique (Settles 2008).
- **Result**: 0.6444, −0.012. All three seeds regressed; runtime jumped to 22s (100 tree.predict calls).
- **Interpretation**: **Trees within one RF are highly correlated** — they see bootstraps of the same data with random feature subsets. Their disagreement reflects *training-noise variability* rather than genuinely ambiguous points. QBC's 3 different model families (RF/LR/HGB) have real inductive-bias diversity; QBB doesn't.
- **Kept?** No.

### V22 — Confidence-weighted pseudo (via row duplication)

- **What**: Pseudo rows with consensus P(Left) ≥ 0.85 get 2 copies in training; 0.7 ≤ p < 0.85 get 1 copy. Everything else unchanged from v13.
- **Why**: Differentiate high-vs-low confidence pseudo without changing the mean. High-consensus rows are more trustworthy so they should count more.
- **Result**: 0.6535, −0.003. Same pattern as v15 (pseudo upweight 2×).
- **Interpretation**: Together with v7, v15, and v22, we've now tested *three* different mechanisms for shifting weight toward pseudo, all of which hurt. Confirms the **real:pseudo ≈ 3:1** ratio is close to optimal, and higher pseudo weight amplifies noise faster than it adds signal.
- **Kept?** No.

### V23 — Combined uncertainty + minority bias

- **What**: Multiply the QBC disagreement by a minority-preference factor: `score = std · (1 + β · mean_P_Left)`. Tested β=1.0 and β=0.5.
- **Why**: F1(Left) cares only about the minority class. Querying rows the committee thinks are more likely Left could specifically boost minority representation in `labeled`.
- **Result**: β=1 → 0.6500 (−0.006); β=0.5 → 0.6503 (−0.006). Both hurt about equally.
- **Interpretation**: Biasing queries toward the predicted minority reduces the diversity of what we oracle — we end up over-sampling "confidently predicted Left" (which the model already handles well) instead of hunting for genuinely confusing rows. Explicit minority bias in the query score double-dips with the minority upweight we already have in training.
- **Kept?** No.

### V24 — Adaptive pseudo threshold schedule

- **What**: Instead of fixed 0.7 threshold, linearly interpolate from **0.8** (first pseudo iter) → **0.65** (last iter). Rationale: early model is less trustworthy so require higher confidence; later model can be trusted with more permissive threshold.
- **Why**: Curriculum-style pseudo — start safe, gradually admit more pseudo-labels as the model matures.
- **Result**: 0.6500, −0.006. All seeds slightly worse.
- **Interpretation**: At threshold=0.8 early, pseudo pool is nearly empty (as v12b showed); at 0.65 late, noise creeps in. The trajectory delivers *both* problems in sequence rather than either working. Fixed threshold at the sweet-spot 0.7 dominates.
- **Kept?** No.

### V25 — Committee expansion (add more model families, don't replace)

- **What**: Enlarge the QBC committee beyond v13's 3 members. Tested all common additions:
  - **25a**: v13 + KNN (4 members) — instance-based
  - **25b**: v13 + MLPClassifier (4 members) — neural non-linear
  - **25c**: v13 + ExtraTreesClassifier (4 members) — more randomized tree ensemble
  - **25d**: v13 + MLP + ExtraTrees (5 members)
  - **25e**: v13 + KNN + MLP + ExtraTrees (6 members)
- **Why**: v13's committee has two tree-based members (RF, HGB) and only one non-tree (LR). Adding genuinely different families (instance-based, neural, more randomized trees) might broaden the disagreement signal.
- **Result**: Every expansion hurt, in a range of −0.003 to −0.012.
  - Single-addition MLP was worst (−0.012). Neural networks tend to output highly polarized probabilities that dominate the std computation.
  - Single-addition ET was best (−0.003) but seed-level was mixed (seed 3 gained +0.004, others lost).
  - Bigger committees (5, 6 members) landed *between* the singletons — averaging out individual model quirks but still net-negative.
- **Interpretation**: The 3-member committee isn't limited by *breadth*; adding more members dilutes rather than sharpens the disagreement signal. Two possible mechanisms: (a) new members' probability calibrations differ from the tree/linear group so they dominate std; (b) with more members, the std tends to shrink toward each model's individual prediction confidence, weakening the "which points are argued about" signal.
- **Kept?** No.

### V26 — Symmetric pseudo at threshold 0.7 (with balanced counts)

- **What**: Extend v13's minority-only pseudo to include confident *Stayed* predictions too: add rows where committee-consensus P(Left) ≤ 0.3 with label 0, capping the count to match the pseudo-Left count so the pseudo set is class-balanced.
- **Why**: Earlier tests of symmetric pseudo (v4, v5) used threshold 0.95, where pseudo-Left never populated because RF proba caps at ~0.9. Symmetric at threshold 0.7 (in the achievable range) is untested. Adding confident Stayed pseudos in principle gives the model more Stayed reference points too.
- **Result**: 0.6459, −0.010. All seeds regressed.
- **Interpretation**: Confirms v6's original hypothesis — **pseudo-Stayed hurts F1(Left) even with count-balancing**:
  - Confident Stayed rows carry little information (model already handles them easily), so gains are minimal.
  - Wrong pseudo-Stayed labels (a true Left mislabeled as Stayed) *directly reduce recall on Left* — the biggest lever in F1(Left).
  - Adding Stayed rows counteracts the minority-upweight (3× real Left) that variant 13 established as critical.
- **Kept?** No — validates minority-only pseudo (v6/v13 design) at the current best threshold.

### Key structural insight (measured at v13)

At the 3× minority upweight, the **final training set is 68.8% Left** (7,829 Left / 11,386 rows for seed 1) — the strategy inverts the natural ~15% minority into a 69% majority. This reveals the true mechanism behind why every imbalance lever worked: **we are effectively pushing the RF's decision threshold far down** so it predicts Left eagerly. Since `evaluate_model` calls `model.predict()` (fixed 0.5 argmax threshold), the only way to shift the operating point is via training-data composition — and F1(Left) on this data wants very high recall, hence ~69% Left is near-optimal.

### V27 — Target-ratio oversampling (fractional upweight)

- **What**: Instead of integer duplication, oversample real Left rows (with replacement) until the training set hits an exact target Left fraction. Swept 0.60–0.80.
- **Why**: The 2×/3×/4× integer sweep found 3× (≈69% Left) best, but the true optimum could be fractional. Target-ratio gives continuous control.
- **Result**: Peak at 0.70 → 0.6485, still −0.008 below v13. All target fractions underperformed the integer 3×.
- **Interpretation**: **Deterministic integer duplication beats stochastic resampling to the same ratio.** Tripling every Left row equally gives uniform emphasis; with-replacement sampling randomly over/under-weights specific rows, adding variance. The ~69% target is confirmed correct, but v13's mechanism for reaching it is superior.
- **Kept?** No.

### V29 — Stratified query batch (reserve minority hunt) — NEW BEST

- **What**: Split each 500-query batch: `(1−f)` by highest disagreement (boundary), `f` by highest P(Left) (guaranteed minority discovery). Reserved allocation, so minority hunt can't dilute the disagreement picks. Adopted f = 0.20.
- **Why**: The ~69% Left training target means the strategy wants as many *real* Left examples as possible to feed the 3× upweight. Explicitly reserving batch slots to find Left increases real minority yield. Distinct from v23 (multiplicative bias, which diluted disagreement and hurt).
- **Result**: f=0.20 → **0.6575** (+0.0013 over v13). Sweep: f=0.10→0.6563, 0.15→0.6519, 0.20→0.6575, 0.25→0.6529, 0.30→0.6537.
- **First reaction (wrong)**: the jagged sweep looked like local-test noise, so initially not adopted.
- **Mechanistic check (decisive)**: measured real Left oracle yield — **STRAT=0.0 oracles 5,364 Left (35.8% of queries); STRAT=0.2 oracles 6,787 (45.2%)**, i.e. +1,423 (+26%) more real minority labels, deterministically, every seed. The mechanism unambiguously fires and is pool-agnostic (will transfer to hidden data).
- **Corrected interpretation**: the effect is **real but saturated**. We are already past the Left-quantity ceiling (target-ratio sweep showed >70% training Left doesn't help), so the extra real Left examples give only a small residual benefit — their advantage over v13 is being *genuine and varied* rather than the same rows tripled. Net +0.0013, and critically it did **not** cost anything on the boundary side. The jaggedness at 0.15/0.25 is noise layered on a real flat-positive effect. Bonus: seed 1 (the persistent laggard across all prior variants) lifts 0.6502 → 0.6550, giving the tightest seed spread we've achieved (0.655–0.660).
- **Kept?** **Yes** — real mechanism, mildly net-positive, fixes the weak seed, zero runtime cost. Lesson: verify the *mechanism*, not just the score, before calling a small delta noise.

### V30 — Decouple final-model upweight from scorer upweight

- **What**: The minority-upweight sweep (v13) applied the same multiplier to both the committee scorers and the final model. Here, hold the scorer at 3× (good query quality) but sweep the *final* model's upweight independently: 4×, 5×, 6×.
- **Why**: Scorer and final model have different jobs — scorer ranks pool points (queries), final model wants maximal F1-optimal threshold shift. Maybe 4× hurt earlier only because it degraded the *scorer*, and the final model would prefer a harder skew.
- **Result**: final 3× (=v29) 0.6575, 4× 0.6555, 5× 0.6534, 6× 0.6546. All decoupled values ≤ v29.
- **Interpretation**: The final model *also* peaks at 3×, independently. Disproves the decoupling hypothesis and strengthens the "3× is the true optimum" conclusion — it's optimal for both roles separately, not an averaging artifact.
- **Kept?** No.

### V31 — Post-hoc self-training

- **What**: After the AL loop produces the final RF, run 1–2 extra rounds: use the final model's own P(Left) to add fresh confident-Left pseudo-labels from the remaining pool, then retrain.
- **Why**: Classic self-training with the actual returned model (not the committee), which might catch confident-Left rows the committee missed.
- **Result**: 1 round 0.6562 (−0.0013), 2 rounds 0.6549 (−0.0026).
- **Interpretation**: The final model is trained at 69% Left, so its P(Left) is inflated → too many rows clear 0.7 → noisy pseudo-Left floods in. The final model is precisely the *wrong* pseudo source because of its deliberate Left skew; the committee mean (built from less-skewed scorer training) is better.
- **Kept?** No.

### V32 — Committee probability calibration

- **What**: Wrap each committee member in `CalibratedClassifierCV(cv=3, method="isotonic")` before computing disagreement and consensus. Tested pseudo threshold 0.7 and 0.6 (calibration shifts the proba range).
- **Why**: Members have different calibration (MLP polarized, RF capped at 0.9). A common calibrated scale could sharpen both the disagreement std and the consensus pseudo.
- **Result**: 0.6240 (thr 0.7) / 0.6175 (thr 0.6) — a large −0.034 drop; runtime doubled to ~28s.
- **Interpretation**: Isotonic calibration is fit on the *upweighted* training data (69% Left), so it calibrates toward that artificial prior — pulling probabilities in exactly the wrong direction for surfacing the true minority. Calibration assumes a representative class distribution, which our deliberate skew violates. Runtime cost (internal 3-fold CV per member) is also a hazard.
- **Kept?** No.

### V33–V35 — Completing the canonical AL method list

Audited our work against the standard Active Learning method catalog (random, uncertainty {entropy/least-confidence/margin}, QBC {trees/vote-entropy/variance/bootstrap-RFs}, diversity {clustering/farthest-first/distance-to-labeled}). Three methods had never been *literally* implemented, though close variants were tested and failed. Ran them for completeness.

- **V33 — QBC with bootstrap-RF committee.** 5 RandomForests, each trained on a bootstrap resample of the labeled set; disagreement = std across the 5. Result: **0.6451 (−0.012)**. Bootstrap RFs are highly correlated (all RF, differing only by resample), so their disagreement is weak/noisy — same failure mode as Query-by-Bagging (v21, trees within one RF). Confirms RF-only committees can't match the 3-family committee.
- **V34 — Distance-to-labeled diversity.** Blend `α·disagreement + (1−α)·dist_to_nearest_labeled` (scaled features). Result: α=0.7 → 0.6511 (−0.006), α=0.9 → 0.6552 (−0.002). Monotone: the more the diversity term influences selection, the worse. Same conclusion as pool-density (v11) and Information Density (v18) — geometric coverage relative to the labeled set is not aligned with the decision boundary.
- **V35 — Farthest-first / Core-Set.** Greedy k-center: pick points maximizing min-distance to the already-selected set (seeded by the labeled set), pure geometric coverage ignoring disagreement. Result: **0.6462 (−0.011)**, runtime 27s. Notably *well above random* (0.548) despite ignoring the boundary — because the pseudo-labeling + 3× upweight training recipe carries most of the load regardless of query rule. But it still loses to disagreement-based queries by 0.011.
- **Cross-cutting note (least-confidence / margin):** for binary classification these are *provably* identical rankings to |p−0.5| (entropy, least-confidence, and margin are all monotonic transforms of |p−0.5|), so they need no separate test beyond v2 — verified by proof, not just assumed.
- **Implementation note:** v33–v35 live behind dormant flags in `strategy.py` (`COMMITTEE_MODE`, `DIST_TO_LABELED_ALPHA`, `QUERY_MODE`) at their v29 defaults. These experimental branches should be stripped before final submission for a clean file.
- **Meta-insight from v35:** most of our gain (~0.07) comes from the *training recipe* (pseudo + minority upweight), not the query rule (~0.037 uncertainty-over-random). Even a boundary-blind query strategy lands at 0.646 with our recipe. This reframes where the value is.
- **Kept?** None — all three confirmed the existing pattern. v29 remains best.

### T1–T10 — Hyperparameter sweep at v13 config

- **What**: Systematic single-knob sweeps over the strategy's remaining hyperparameters, holding all else equal at v13 config:
  - `BATCH_SIZE`: 250, **500** (baseline), 1000
  - `PSEUDO_START_FRACTION`: 0.3, **0.5**, 0.7
  - `LogisticRegression.C`: 0.5, **1.0** (default), 2.0
  - `HistGradientBoosting.learning_rate`: 0.05, **0.1** (default), 0.2
  - Scorer `RandomForest.n_estimators`: **100**, 300
  - `PSEUDO_CAP_PER_CLASS`: 200, **2000** (default; effectively unbounded since threshold 0.7 yields ~300–500 candidates/iter)
- **Why**: Confirm none of the strategy's tuned knobs has an unexplored optimum. Also refactored `PSEUDO_START_ITER` into `PSEUDO_START_FRACTION` so batch-size sweeps compare apples-to-apples.
- **Results**: **Every single-parameter change hurt.** The best of the sweep tied within noise of v13 baseline; the worst dropped 0.015.
- **Interpretation**: v13's hyperparameters form a **local optimum** in the ~6-dimensional space we can control. Further gains must come from a *structural* change (different algorithm/composition), not more knob-turning. Notable individual findings:
  - Batch=250 more retrains but hurt slightly — the extra refreshes don't outweigh the noisier uncertainty from smaller labeled sets.
  - Batch=1000 much faster (9s) but hurt more — fewer retrains means stale scoring.
  - Pseudo-cap at 200 (below the natural ~300–500 candidate count) dropped 0.015 — confirms the full natural pseudo pool is load-bearing; artificially shrinking it loses signal.
- **Kept?** No — all reverted to v13 defaults.

---

## Observations / Ideas

- **Class imbalance is the dominant lever.** Every strategy that explicitly rebalances (minority pseudo, minority upweight) has moved the needle; strategies that ignore imbalance haven't.
- **RF `predict_proba` on the minority class caps around 0.85–0.91.** Any threshold ≥ 0.9 for pseudo-Left labels produces an empty pool. This shaped every pseudo-related decision.
- **Seed 2 was the initial laggard; seed 1 has become the current laggard.** Suggests seed-level variance is inherent to the initial splits, not a fixable feature of any single strategy.
- **PCA-space geometry doesn't predict information structure** — density-aware sampling hurt because the decision boundary lives inside the dense core, not the sparse halo.
- **Committee family diversity has diminishing returns** — 3 members already captures most of the useful cross-model disagreement; adding a 4th (SVM or 2nd RF) doesn't reliably help.
- **The 3:1 real:pseudo weight ratio is close to optimal** — deviations in both directions (v7 at 2:1, v15 at 3:2) have hurt.
