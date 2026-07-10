# Project B — Section A: Active Learning

Active-learning strategy for a binary attrition classifier (predict whether an employee **Left** or **Stayed**). The final classifier is a fixed `RandomForestClassifier` provided by the course framework; the only lever is **which employees to label** — 500 free initial labels plus up to 5,000 oracle queries — to maximize **F1 on the Left class**.

## Result

Local mean **F1(Left) = 0.6575** across seeds 1–3 (all seeds 0.655–0.660), ~13s/seed (limit: 60s). Clears the 0.55 guaranteed-points gate by +0.10.

## Approach (final strategy)

- **Query by Committee** — RandomForest + Logistic Regression + HistGradientBoosting; each iteration queries the pool points with the highest disagreement (std of P(Left)) across the three families.
- **Stratified batches** — 80% of each 500-query batch by disagreement (boundary), 20% reserved for highest-P(Left) rows (guaranteed minority discovery). Raises real-Left oracle yield from ~36% to ~45%.
- **Consensus pseudo-labeling** — pool rows where the committee *mean* P(Left) ≥ 0.7 are added as Left=1 (minority-only), refreshed each iteration from the halfway point.
- **Minority upweighting** — real Left rows duplicated 3× in training, pushing the training set to ~69% Left. Since the RF's decision threshold is fixed at 0.5, skewing the data is the only way to bias the model toward the recall-heavy operating point F1(Left) rewards.

The full experiment history (~40 variants, what worked and what didn't, with rationale) is in **[process.md](process.md)**; a condensed milestone view is in **[process_summary.md](process_summary.md)**.

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
