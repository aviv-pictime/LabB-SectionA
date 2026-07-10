# Active Learning for Employee Attrition (Section A)

Active Learning for **binary employee-attrition classification** (predict whether an
employee will **Left** = 1 or **Stayed** = 0). Starting from a small labeled set, the
strategy iteratively queries a labeling oracle under a fixed budget and returns a
trained `RandomForestClassifier`. The metric is **F1 of the positive `Left` class**.

> Only `strategy.py` is graded/submitted. `run.py`, `utils.py`, `evaluation.py`, and
> `constants.yaml` are the fixed framework and must not be modified.

## Results

Official evaluation (`python evaluation.py`, mean over seeds 1–3):

| Strategy | Mean F1(Left) | Notes |
|---|---|---|
| Initial 500 labels only | 0.407 | no active learning |
| Random querying (5,000) | 0.564 | baseline |
| `posprob` (highest P(Left)) | 0.625 | strong single strategy, preserved as baseline |
| **`pos_diverse` + ratio = 0.80 (default)** | **≈ 0.650** | diversity within positives + class-ratio control |

Per-seed for the default: 0.642 / 0.656 / 0.652. Runtime 24–34 s/seed (limit 60 s),
deterministic across `PYTHONHASHSEED`.

## Problem setup

- Pool: ~14,900 unlabeled employees. Initial labeled set: 500/seed (free).
- Oracle budget: **≤ 5,000** unique Employee IDs. Runtime: **≤ 60 s/seed**.
- Model: framework-fixed `RandomForestClassifier` (100 trees). The 0.5 decision
  threshold and hyperparameters cannot be changed.
- Allowed imports in `strategy.py`: `numpy, pandas, sklearn, scipy, collections,
  warnings, typing, utils`.

## The approach (short version)

1. **Iterative pool-based loop:** train → score the pool → query a batch → retrain.
2. **`pos_diverse` acquisition:** query the highest predicted `P(Left)` candidates
   (maximises true-positive yield, the main driver of F1 for the minority class),
   but shortlist ~3×batch and spread the batch across feature-space clusters so
   budget isn't spent on near-duplicate positives.
3. **Class-ratio control:** the RF is under-confident on `Left` (only ~30 % of true
   positives reach P ≥ 0.5), so recall limits F1. The final model is trained on the
   acquired **true** labels oversampled to ~0.80 positive prevalence, which
   compensates the fixed 0.5 threshold. The ratio is chosen by unbiased internal
   validation (`final_pos_ratio="auto"`); 0.80 is the validated plateau value.

Full reasoning, analysis, and experiment tables are in [`PROCESS.md`](PROCESS.md);
pipeline/config details are in [`WORKFLOW.md`](WORKFLOW.md).

## Repository layout

| File | Role |
|---|---|
| `strategy.py` | **The submission.** Modular AL pipeline + all selectable strategies. |
| `run.py`, `utils.py`, `evaluation.py`, `constants.yaml` | Fixed framework (do not modify). |
| `experiments.py` | Dev-only harness to compare strategies over seeds. |
| `analysis.py` | Dev-only analysis (yield curve, per-iteration trajectory, error/FN study). |
| `data/` | Course-provided local data (not committed; see below). |

## Running

```bash
# Official self-evaluation (requires the course data/ folder)
python evaluation.py

# Compare strategies (dev)
python experiments.py posprob,posdiv_r080,ratio_auto
python experiments.py all

# Analysis (dev)
python analysis.py yield        # positive-yield curve vs P(Left)
python analysis.py trajectory   # per-iteration dynamics
python analysis.py errors       # false-negative characterization
```

Dependencies: `numpy pandas scikit-learn scipy pyyaml joblib`.

## Data & privacy

The `data/` folder and the assignment PDF are course-provided and are **not committed**
(see `.gitignore`). In particular `data/.pool_labels.pkl` is an internal oracle file
that must never be read directly by the strategy. To run locally, place the provided
`data/` folder next to the code.
