# Project B — Section A: Active Learning

Active-learning strategy for a binary attrition classifier (predict whether an employee **Left** or **Stayed**). The final classifier is a fixed `RandomForestClassifier` provided by the course framework; the only lever is **which employees to label** — 500 free initial labels plus up to 5,000 oracle queries — to maximize **F1 on the Left class**.

## Result

Local mean **F1(Left) = 0.6575** across seeds 1–3 (all seeds 0.655–0.660), ~13s/seed (limit: 60s). Clears the 0.55 guaranteed-points gate by +0.10.

## Approach (final strategy)

- **Query by Committee** — RandomForest + Logistic Regression + HistGradientBoosting; each iteration queries the pool points with the highest disagreement (std of P(Left)) across the three families.
- **Stratified batches** — 80% of each 500-query batch by disagreement (boundary), 20% reserved for highest-P(Left) rows (guaranteed minority discovery). Raises real-Left oracle yield from ~36% to ~45%.
- **Consensus pseudo-labeling** — pool rows where the committee *mean* P(Left) ≥ 0.7 are added as Left=1 (minority-only), refreshed each iteration from the halfway point.
- **Minority upweighting** — real Left rows duplicated 3× in training, pushing the training set to ~69% Left. Since the RF's decision threshold is fixed at 0.5, skewing the data is the only way to bias the model toward the recall-heavy operating point F1(Left) rewards.

## Methods explained

### Concepts you need first
- **Precision / Recall / F1 (for "Left")** — *precision* = of those we predicted Left, how many really Left; *recall* = of those who really Left, how many we caught; *F1* = their harmonic mean (high only when both are). We maximize F1(Left).
- **Class balance** — "Left" is the minority: **~1 in 3** (measured 31.5% in the initial labels, 33.3% in the test set), outnumbered ~2:1 by "Stayed." A model that leans "Stayed" gets easy accuracy but misses leavers — hence F1, not accuracy.
- **Oracle** — the paid labeling service; each unique ID costs one of our 5,000. **Pool** — the ~14,900 unlabeled records. **Seed** — controls the initial 500 + randomness; graded on 3 seeds, averaged.
- **`predict_proba` / P(Left)** — the forest is 100 trees; P(Left) is the fraction voting "Left."
- **Fixed 0.5 threshold** — the model predicts "Left" when P(Left) > 0.5; the framework's `.predict()` uses 0.5 and we can't change it. This constraint drives the whole strategy.
- **Pseudo-label** — a label we *guess* from the model (free) instead of buying from the oracle. **Committee (QBC)** — several different models; where they disagree is where the data is genuinely ambiguous.

### Methods that worked (what each is, why it helped)
- **Uncertainty sampling** — *what:* each round, train the model, then buy labels for the rows whose P(Left) is closest to 0.5 (the ones it's most torn about). *Why:* those sit on the decision boundary — labeling them sharpens exactly the part the model is confused about. (For two classes, "least confident," "smallest margin," and "highest entropy" are all the same ranking as |P−0.5|.)
- **Minority pseudo-labeling** — *what:* add unlabeled rows the model is confident are Left (P ≥ 0.7) to training as Left, for free; Left-only. *Why:* free extra minority examples. The 0.7 threshold (not 0.95) is deliberate — RF probabilities for a rare class cap around 0.85–0.9, so 0.95 would select nothing. Left-only because a wrong "Stayed" guess makes the model miss a real leaver (kills recall), while a wrong "Left" guess is mostly harmless.
- **Query by Committee (QBC)** — *what:* train three different model families — **RandomForest** (many trees), **Logistic Regression** (a linear model), **HistGradientBoosting** (boosted trees) — and query the rows where their P(Left) values disagree most (standard deviation across the three). *Why:* disagreement across genuinely different model shapes is a stronger "worth labeling" signal than one model's self-doubt.
- **Consensus pseudo-labeling** — *what:* use the same committee, but pseudo-label a row Left only when the committee's *mean* P(Left) ≥ 0.7 (all three agree). *Why:* disagreement picks what to *ask* the oracle; agreement picks what to *trust* for free — requiring all three to agree filters out single-model overconfidence.
- **Minority upweighting (3×)** — *what:* duplicate every real Left row three times in training. *Why (the core idea):* we can't move the fixed 0.5 line, but we can change the data. Tripling Left pushes the training mix from the ~1/3 base rate to **~69% Left** (measured 68.8%), so the trees vote "Left" more readily and more rows cross 0.5 — moving the operating point through data instead of threshold. Swept: 2× < 3× > 4× (past 3×, precision collapses).
- **Stratified queries** — *what:* reserve 20% of each batch for the highest-P(Left) rows (deliberately hunting real leavers), 80% for disagreement. *Why:* feeds the 3× upweight more *real* Left examples — raised real-Left oracle yield 36% → 45%. Small F1 gain (already saturated) but it fixed the weakest seed; kept only after verifying the yield jump was real, not noise.

### Methods that did NOT work (what each is, why it failed)
- **Diversity sampling** — *what:* pick a spread-out, representative set instead of near-duplicates. Flavors: **k-means** (cluster the pool, label one per cluster), **density** (prefer sparse or dense neighborhoods), **farthest-first / Core-Set** (greedily pick points far from all chosen so far), **distance-to-labeled** (prefer rows unlike what we have). *Why it failed:* all measure "spread" by feature-space geometry, but the decision boundary sits inside the *dense* core, not the sparse regions — so spreading out buys easy points the model already gets right.
- **Bigger / different committees** — *what:* **Query-by-Bagging** (use the RF's 100 individual trees as members), **bootstrap-RFs** (5 RFs on resampled data), or add **SVM** (max-margin classifier), **KNN** (majority vote of nearest neighbors), **MLP** (small neural net), **ExtraTrees** (more-randomized forest). *Why it failed:* trees/RFs on the same data are too correlated — their "disagreement" is just noise; adding an SVM/KNN/MLP added nothing or hurt (a *likely* cause for the MLP, not separately verified, is that neural nets output polarized 0/1 probabilities that dominate the disagreement score); the SVM nearly blew the 60s limit. Three different-by-design families already capture the useful disagreement.
- **Probability calibration** — *what:* post-process scores (Platt/isotonic) so a stated 0.7 means a true 70%. *Why it failed (worst result):* calibration tunes toward the *training* class distribution — but we deliberately skewed that to 69% Left, so calibration undoes our main lever.
- **Symmetric pseudo-labeling** — *what:* also add confident "Stayed" guesses. *Why it failed:* confident-Stayed rows teach nothing and a wrong one makes the model miss a leaver; it also dilutes the minority skew.
- **Alternative disagreement measures** — **BALD** (ensemble entropy minus mean member entropy — isolates model uncertainty), **KL divergence** (distance of each member from consensus; same ranking as BALD for two classes), **vote entropy** (spread of hard votes). *Why they failed:* plain std keeps the full probability information; the others over-weight extremes or discard magnitudes.

The full experiment history (~43 variants, in order, with rationale) is in **[process.md](process.md)**; the deep-dive study guide (concepts + every method) is in **[process_summary.md](process_summary.md)**.

## Files

| File | Role |
|---|---|
| `strategy.py` | The submission — implements `run_active_learning(seed)`. **The only file I authored.** |
| `run.py`, `utils.py`, `evaluation.py` | Fixed course framework (unmodified). |
| `process.md` | Full experiment log. |
| `process_summary.md` | Condensed milestone summary. |

## Running

```bash
python evaluation.py
```

Requires the course-provided `data/` directory and `constants.yaml`, which are **not** included here (course-owned data / instructor config). Place them in the project root before running.

---

_Technion coursework. Shared privately for personal version history — not for redistribution._
