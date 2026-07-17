# Section A — Milestone Summary & Method Guide

_Last updated: 2026-07-10._ A learn-it-from-scratch guide to the active-learning strategy: the concepts, what each method does and why, and what failed. The blow-by-blow experiment log (every variant, in order) is in `process.md`.

**Goal:** maximize the mean F1 of the **"Left"** class on a **fixed** RandomForest. We get 500 free labels + up to 5,000 paid oracle queries, must finish in ≤60s per seed, and are scored on 3 seeds {1,2,3}. Clearing **F1 ≥ 0.55** guarantees the base points; higher is competitive.

---

## 1. The problem in plain words

We are predicting whether an employee **Left** (1) or **Stayed** (0). We have ~14,900 employee records but almost none are labeled. Getting a label costs budget (the "oracle"). **Active learning** is the art of choosing *which* records to spend our label budget on so the final model is as good as possible.

Three things make this hard:

1. **The classifier is frozen.** We must return the framework's RandomForest with fixed settings. We can't tune it, change its threshold, or swap models. **The only thing we control is the training data** — which rows go in, and with what labels.
2. **"Left" is the minority (~1 in 3 — measured 31.5% in the initial labels, 33.3% in the test set).** It's outnumbered ~2:1 by "Stayed," so a lazy model that leans toward "Stayed" scores decent accuracy while catching too few leavers. That's why we're scored on **F1 of the Left class**, not accuracy (see §2).
3. **Tight budget and time.** 5,000 labels out of ~14,900, and 60 seconds per run.

---

## 2. Key concepts (glossary)

- **Pool** — the ~14,900 unlabeled employee records we can choose from.
- **Oracle** — the labeling service. `call_oracle(ids)` returns true labels for those IDs and charges budget. Each *unique* ID costs one of our 5,000; re-asking the same ID is free.
- **Seed** — a random setting that picks the initial 500 labeled rows and all randomness. We're graded on 3 seeds and the scores are averaged, so the strategy has to be robust, not lucky.
- **Precision, Recall, F1 (for "Left")**
  - **Precision** = of everyone we *predicted* Left, what fraction actually Left. (Are our "Left" calls trustworthy?)
  - **Recall** = of everyone who *actually* Left, what fraction we caught. (Did we find the leavers?)
  - **F1** = the harmonic mean of the two — it's high only when *both* are high. This is what we maximize.
- **Class imbalance** — one class (Stayed) hugely outnumbers the other (Left). Models default to the majority, which kills recall on the minority. Fighting this imbalance turned out to be the whole game (§5).
- **`predict_proba` / P(Left)** — the RandomForest is 100 decision trees; P(Left) is the fraction of trees voting "Left." So P(Left)=0.7 means 70 of 100 trees said Left.
- **Decision threshold (the fixed 0.5)** — the model predicts "Left" when P(Left) > 0.5. The framework calls `.predict()`, which uses 0.5 and **we cannot change it**. This constraint is the key to everything (§5, "Why it works").
- **Pseudo-labeling** — instead of *paying* the oracle, we let the model *guess* labels for confident unlabeled rows and add those guesses to training. Free, but a wrong guess adds noise.
- **Committee / Query by Committee (QBC)** — train several *different* models; where they **disagree** is where the data is genuinely ambiguous and most worth labeling.
- **Uncertainty** — a single model is most "unsure" about a row when P(Left) is near 0.5 (a coin-flip). Those rows sit on the decision boundary.

---

## 3. The path that WORKED (quick table)

| Step | What we changed | Mean F1 | Gain |
|---|---|---|---|
| Baseline | Train on the 500 free labels only | 0.4068 | — |
| Random 5,000 | Spend the whole budget on random IDs | 0.5481 | +0.141 |
| Uncertainty | Query the 500 rows the model is least sure about, each round | 0.5846 | +0.036 |
| Minority pseudo | Also add confident "Left" guesses (free) to training | 0.6222 | +0.038 |
| QBC queries | Query where 3 different models disagree most | 0.6251 | +0.003 |
| Consensus pseudo | Only trust "Left" guesses when all 3 models agree | 0.6364 | +0.011 |
| Minority upweight 3× | Duplicate real "Left" rows 3× in training | 0.6562 | +0.020 |
| Stratified queries | Reserve 20% of each batch to hunt real "Left" | **0.6575** | +0.0013 |

**Final: 0.6575 mean**, all seeds 0.655–0.660, ~13s/seed. That's +0.25 over the baseline and +0.10 over the 0.55 gate.

---

## 4. The working methods, explained in depth

### Baseline — 0.407
**What:** just train the RandomForest on the 500 free labeled rows and return it. No oracle spending.
**Why we do it:** it's the reference point. It also confirms the pain — F1 is only 0.41 because with just 500 rows (~1/3 of them Left) the model barely learns to separate the minority.

### Random 5,000 — 0.548  (+0.141)
**What:** pick 5,000 random pool IDs, buy their labels, add them, train once.
**Why it matters:** this isolates "more data" from "smart data." It jumps +0.14 — so *most* of the achievable gain is just having more labels at all. This becomes the floor: any clever strategy has to beat 0.548 to justify itself.

### Uncertainty sampling — 0.585  (+0.036)
**What (mechanics):** work in 10 rounds of 500. Each round: (1) train an RF on everything labeled so far; (2) score every unlabeled row's P(Left); (3) pick the 500 rows with P(Left) closest to **0.5** — the ones the model is most torn about; (4) buy their labels; (5) repeat.
**Intuition:** a row where the model already says "95% Stayed" teaches it little. A row at 50/50 sits right on the decision boundary — labeling it sharpens exactly the part of the boundary the model is confused about. Spending budget there is more efficient than random.
**Note (binary trick):** "closest to 0.5," "least confident," and "smallest margin" and "highest entropy" are the *same ranking* for two classes — they're all just |P(Left) − 0.5|. So we didn't need to test them separately.

### Minority pseudo-labeling — 0.622  (+0.038, the biggest single jump)
**What (mechanics):** partway through, look at the still-unlabeled pool. Any row the model is confident is a leaver — **P(Left) ≥ 0.7** — gets added to the training set **labeled Left, for free** (no oracle charge). We only do this for the **minority (Left)**, never for Stayed. The pseudo-labels are refreshed each round as the model improves.
**Why 0.7 and not 0.95?** RandomForest probabilities for a *rare* class saturate low — its P(Left) tops out around 0.85–0.9, never 0.95. So a "confident" threshold of 0.95 would select **zero** rows (we checked). The threshold has to live in the model's *actual* probability range. This was a real diagnostic finding, not a guess.
**Why minority-only?** F1(Left) punishes *missing* leavers (low recall) more than the occasional false alarm. A wrong pseudo-"Left" nudges the model to over-predict Left (mostly harmless, sometimes helpful). A wrong pseudo-"Stayed" would nudge it to *miss* leavers — directly hurting the score. So we add Left guesses and never Stayed guesses.
**Net effect:** free extra minority examples that push the model to find more leavers.

### QBC queries (Query by Committee) — 0.625  (+0.003)
**What (mechanics):** instead of one RF judging uncertainty, train **three different model families** on the same data:
- **RandomForest** (many decision trees),
- **Logistic Regression** (a linear model),
- **HistGradientBoosting** (boosted trees).
For each pool row, get P(Left) from all three. The query score is the **standard deviation** across the three — i.e., *how much they disagree*. Buy the 500 they disagree about most.
**Intuition:** a single model's "uncertainty" only reflects *its own* blind spots. When three genuinely different model types all agree, the row is easy. When they *split* (say 0.9 / 0.2 / 0.5), the row is genuinely ambiguous — at least one model is wrong, and buying the true label settles the argument and teaches all of them. Disagreement across *different* model shapes is a stronger "this is worth labeling" signal than one model's self-doubt.

### Consensus pseudo-labeling — 0.636  (+0.011)
**What (mechanics):** reuse the same 3-model committee, but now for the *pseudo-labels* use the **mean** P(Left) across the three. A row is pseudo-labeled Left only if the committee's **average** confidence ≥ 0.7 — i.e., all three lean Left.
**The elegant part:** we use the committee two opposite ways at once. **Disagreement** (std) picks what to *ask the oracle* (the ambiguous rows). **Agreement** (mean) picks what to *trust as a free pseudo-label* (the rows all models are sure about). Requiring all three to agree filters out cases where one model was overconfident on its own — cleaner free labels, fewer mistakes injected.

### Minority upweight 3× — 0.656  (+0.020, the deepest idea)
**What (mechanics):** in the final training set, include every *real* "Left" row **three times** (duplicate it). Simple as that.
**Why it works — the core insight of the whole project:** remember the model predicts "Left" only when P(Left) > 0.5, and **we can't move that 0.5 line.** But P(Left) itself depends on how often the trees saw "Left" during training. The representative rate is ~1 in 3 (33% in the test set); tripling the real Left rows pushes our training mix all the way to **~69% Left** (we measured 68.8%). Now the trees vote "Left" far more readily, so more rows cross the 0.5 line. In effect, **we moved the operating point by changing the data instead of the threshold** — exactly what F1 on the minority wants (catch more leavers).
**Why 3× and not more?** We swept it: 2× → 0.649, 3× → 0.656, 4× → 0.648. Past ~3× the model over-predicts Left and *precision* collapses (too many false alarms), so F1 drops. 3× is the sweet spot.

### Stratified queries — 0.6575  (+0.0013, small but real)
**What (mechanics):** split each 500-query batch: **80% (400)** picked by committee disagreement (as before), and **20% (100)** reserved for the rows with the highest P(Left) — deliberately going out to *buy real leaver labels*.
**Why:** the 3× upweight is hungry for *real* Left examples to triple. Pure disagreement-picking oracled about 36% Left; adding the reserved minority hunt raised that to **45%** — 26% more genuine leavers bought, every seed. The F1 only moved +0.0013 because we were already *saturated* on Left quantity (past ~70% training-Left doesn't help), but the extra real (not duplicated) leavers gave a small, genuine lift and, notably, fixed our weakest seed.
**Lesson embedded here:** the +0.0013 first looked like random noise. We only kept it after *measuring the mechanism* (the 36%→45% yield jump, identical across seeds) — proving it was a real, data-independent effect, not luck on the local test.

---

## 5. The one-sentence "why it all works"

The RandomForest's decision line is frozen at P(Left) = 0.5 and we can't touch it. So every winning move — minority pseudo-labels, 3× upweighting, minority-hunt queries — does the same thing a different way: **it feeds the model more "Left" so the model becomes more willing to predict "Left,"** which is exactly what F1 on the rare class rewards. That's why the theme of everything that worked is *class-imbalance handling*, and everything unrelated to it did nothing.

---

## 6. The ideas that DID NOT work — each concept explained, then why it failed

We tested ~43 variants. Below, each named method gets a plain **"What it is"** (the general ML concept, independent of this project) followed by **"Why it failed here."**

### 6a. Diversity / coverage sampling
**The shared idea:** don't waste budget on near-duplicate rows; pick a *spread-out* set that covers the feature space. Four flavors:

- **k-means clustering** — *What it is:* an algorithm that groups points into *k* clusters, each summarized by a center (centroid). Every point joins its nearest center; centers are recomputed and points reassigned, repeatedly, until stable. In active learning you cluster the pool and label one representative per cluster to get variety.
- **Density-based sampling** — *What it is:* estimate how crowded each point's neighborhood is (e.g., the average distance to its 10 nearest neighbors — small distance = dense/typical region, large = sparse/unusual). Then prefer either the sparse points (novel) or the dense points (representative).
- **Farthest-first / Core-Set (k-center)** — *What it is:* a greedy "spread" method. Start from the labeled set, then repeatedly add the pool point whose distance to the *nearest already-chosen* point is the largest. The result is a set that blankets the space so nothing is too far from a chosen point.
- **Distance-to-labeled** — *What it is:* score each unlabeled row by its distance to the closest *already-labeled* row, and prefer the far ones — i.e., rows that look unlike anything we've labeled yet.

**Why they all failed here:** every one of these measures "spread" as *geometric distance between feature vectors*. We plotted the data (PCA) and saw one dense core plus a sparse halo — but the **decision boundary, where Left and Stayed actually mix, sits inside the dense core**, not out in the empty regions. So spreading queries outward just buys easy, far-away points the model already classifies correctly. Geometry ≠ where the model is confused. Query-by-Committee already lands right on the boundary; bolting a geometric prior on top only drags picks *off* it.

### 6b. Different / bigger committees
**The shared idea:** if 3 disagreeing models is good, more (or more varied) models should sharpen the disagreement signal.

- **Bootstrap** — *What it is:* a resampling trick: draw N rows *with replacement* from your N training rows to make a slightly different training copy. Repeat to get many varied datasets.
- **Query-by-Bagging (QBB)** — *What it is:* build the committee from bootstraps of *one* model type. We used the RandomForest's own 100 trees as 100 committee members and measured their vote spread.
- **Bootstrap-RFs** — *What it is:* same idea one level up — 5 whole RandomForests, each trained on a different bootstrap of the labeled data.
- **SVM (Support Vector Machine)** — *What it is:* a classifier that draws the boundary maximizing the *margin* (gap) between the two classes. With an "RBF kernel" it can bend that boundary into curves. Powerful but slow — training cost grows roughly with the square (or worse) of the number of rows.
- **KNN (k-Nearest Neighbors)** — *What it is:* the simplest classifier — to label a point, look at its *k* closest neighbors and take the majority vote. No real training; purely distance-based.
- **MLP (Multi-Layer Perceptron)** — *What it is:* a basic neural network — layers of weighted sums passed through nonlinear "activation" functions, trained by gradient descent. Flexible, but tends to output very *polarized* probabilities (things near 0 or 1).
- **ExtraTrees (Extremely Randomized Trees)** — *What it is:* like a RandomForest but the tree splits are chosen even more randomly, producing more *decorrelated* trees.

**Why they failed:** trees inside one forest — and RFs trained on resamples of the same data — are **highly correlated**, so their "disagreement" is just training-noise jitter, not genuine ambiguity (QBB and bootstrap-RFs both lost ~0.012 — measured). Adding an SVM/KNN/MLP either added nothing or hurt (all measured); the RBF-SVM also nearly blew the 60-second limit. A *likely* cause for the MLP (not separately verified in this run) is that neural nets tend to output polarized 0/1 probabilities that dominate the standard-deviation. The three *different-by-design* families we already use (trees / linear / boosted) capture the useful disagreement; extra members dilute rather than sharpen it.

### 6c. Probability calibration  (−0.034, the worst result)
**What it is:** raw classifier scores aren't always "honest" — a model might say 0.7 for things that are actually right only 60% of the time. **Calibration** (e.g. *Platt scaling* or *isotonic regression*, via `CalibratedClassifierCV`) is a post-processing step that fits a correction on held-out data so a stated 0.7 really means ~70%.
**Why it failed badly:** calibration tunes probabilities to match the *training* class distribution — but we **deliberately skewed** that distribution to 69% Left. So calibration "fixes" the model right back toward the very skew we engineered on purpose, cancelling our main lever. It fights the thing that makes the strategy work (and it was ~2× slower).

### 6d. Symmetric pseudo-labeling  (−0.010)
**What it is:** our winning pseudo-labeling only adds confident *Left* guesses. The "symmetric" version also adds confident *Stayed* guesses, for balance.
**Why it failed:** confident-Stayed rows are ones the model already gets easily — they teach it nothing. And any *wrong* Stayed guess trains the model to miss a real leaver, hurting recall directly. It also dilutes the minority skew we're building. The asymmetry (Left-only) is deliberate and correct.

### 6e. Alternative disagreement measures  (−0.005 to −0.010)
Different math for scoring "how much does the committee disagree about this row." We use **standard deviation** of the three P(Left) values. Alternatives tried:

- **BALD (Bayesian Active Learning by Disagreement)** — *What it is:* an acquisition score = (entropy of the *average* prediction) − (average of each member's *own* entropy). It's high when the ensemble as a whole is unsure but each individual member is confident — i.e., it isolates *model* uncertainty (they confidently disagree).
- **KL divergence** — *What it is:* a standard measure of how different two probability distributions are. Here: the average divergence of each member's prediction from the committee consensus. (For our binary case this is mathematically the same ranking as BALD — and it scored identically.)
- **Vote entropy** — *What it is:* turn each member's probability into a hard yes/no vote, then measure the entropy (spread) of the vote tally. Maximal when the vote is evenly split.

**Why they failed:** for a 3-model binary committee, plain std is already the cleanest signal. BALD/KL over-weight disagreements that sit near the probability extremes; vote entropy throws away the actual magnitudes entirely (0.51 and 0.99 both collapse to "vote Left"). Std keeps the full information and beat all of them.

### 6f. Smaller ideas and tuning (each defined, all neutral-to-negative)
- **Target-ratio oversampling** — *What it is:* instead of duplicating minority rows an integer number of times, resample them *with replacement* to hit an exact target class fraction (e.g. "make it exactly 65% Left"). *Failed* because random resampling adds variance; deterministic integer 3× duplication is cleaner.
- **Warm-start** — *What it is:* use plain random sampling for the first round(s) before switching to the smart query strategy, to seed variety. *Failed* because the free initial 500 rows are already varied; the committee's very first picks beat random.
- **Iteration-persistent pseudo** — *What it is:* only accept a pseudo-label if the model stays confident about that row across *two consecutive* rounds (a stability filter). *Failed* because it discards too many still-correct pseudo-labels — the volume lost hurt more than the noise removed.
- **Post-hoc self-training** — *What it is:* after building the final model, use *it* to pseudo-label more pool rows, retrain, and repeat. *Failed* because the final model is 69%-Left-skewed, so its probabilities are inflated → it floods training with noisy "Left" guesses.
- **Decoupled upweight** — *What it is:* use a different minority-upweight factor for the intermediate committee (which picks queries) than for the final returned model. *Failed* because the final model independently peaks at the same 3×.
- **Threshold / amount sweeps** — pseudo threshold: 0.6 (too noisy) < **0.7** > 0.8/0.95 (selects nothing, because RF minority probabilities cap ~0.9). Upweight: 2× < **3×** > 4× (past 3× the model over-predicts Left and precision collapses). Real:pseudo weight ratio **3:1** is optimal in both directions. Batch size, learning rates, tree counts, pseudo cap — every single-knob change was flat or negative. It's a genuine local optimum.

---

## 7. Big-picture lessons (for the talk)

1. **This is an imbalance problem wearing an active-learning costume.** Every gain came from making the frozen model more willing to predict the rare class; everything else was noise. Recognizing that early is what focused the search.
2. **The training recipe beats the query rule — by about 2×.** Smarter querying (uncertainty over random) bought ~+0.037. The composition machinery (pseudo-labels + 3× upweight) bought ~+0.07. We even proved it: a query strategy that *ignores* the boundary entirely (farthest-first) still reaches 0.646 once our recipe is applied. **Where the labels go matters less than how you compose the training set once you have them.**
3. **Verify the mechanism, not just the score.** The stratified-query win (+0.0013) looked like local-test noise. We kept it only after measuring that it deterministically oracled 26% more real leavers on every seed — a real, transferable mechanism — and explaining why the F1 barely moved (saturation). That discipline is what keeps a result from being an overfit fluke.
