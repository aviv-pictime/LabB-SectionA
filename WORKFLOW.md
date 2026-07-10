# Workflow & Design

How `strategy.py` is organised, how the pieces fit together, and how to reproduce
every experiment.

## Pipeline overview

`run_active_learning(seed)` → `run_pipeline(seed, DEFAULT_CONFIG)`. The orchestrator
only wires helpers together; every meaningful step is its own function.

```
load data ─► encode pool by ID ─► [optional] reserve random validation
   │
   ▼
iterate until budget spent:
   fit RF ─► score candidates (strategy) ─► select batch ─► call_oracle ─► merge ─► retrain
   │
   ▼
[optional] conservative pseudo-labeling
   │
   ▼
final training with class-ratio control ─► return model
```

Key correctness choices:

- **Encoding via the framework.** `encode_features_by_id` routes the pool through the
  public `prepare_xy` (adding a dummy target) so features always match the fixed
  `feature_columns` used for training and evaluation.
- **Determinism.** Candidates are ordered with `sorted(ids, key=int)` every iteration,
  so tie-breaking among equal `P(Left)` does not depend on `PYTHONHASHSEED`. Verified:
  identical F1 across hash seeds.
- **Budget/runtime.** `time` and `dataclasses` are not allowed imports, so the budget
  is controlled by batch size / iteration count (not wall-clock), and config is a
  plain class. Total oracle queries default to `utils.MAX_LABELED`.

## Central configuration (`StrategyConfig`)

All knobs live in one place so experiments never require pipeline edits:

| Field | Meaning |
|---|---|
| `strategy` | selection strategy (see below) |
| `batch_size` | queries per iteration (default 500; smaller = more adaptive) |
| `max_queries` | total oracle budget (default = framework budget) |
| `uncertainty_measure` | `entropy` / `least_confidence` / `margin` (equivalent in binary) |
| `hybrid_proportions` | quota weights for `hybrid*` strategies |
| `shortlist_mult` | shortlist size = `shortlist_mult × batch` for diversity |
| `final_pos_ratio` | `None` (natural) / float (oversample) / `"auto"` (validated) |
| `ratio_grid`, `val_reserve` | candidate ratios / optional reserved validation set |
| `use_confidence_regions`, `oof_splits`, `target_reliability_*`, `min_region_count` | confidence-region analysis |
| `use_pseudo_labels`, `pseudo_*` | conservative pseudo-labeling (off by default) |

Defaults:

```python
BASELINE_CONFIG = StrategyConfig(strategy="posprob", batch_size=500)
DEFAULT_CONFIG  = StrategyConfig(strategy="pos_diverse", batch_size=500, final_pos_ratio=0.80)
```

## Selectable strategies

| Name | Idea |
|---|---|
| `random` | uniform sampling (baseline) |
| `uncertainty` | highest predictive entropy / margin / least-confidence |
| `posprob` | highest predicted `P(Left)` (F1-oriented; preserved baseline) |
| `pos_diverse` | **default** – top-`P(Left)` shortlist, then diverse across clusters |
| `hybrid_pu`, `hybrid_pur` | posprob + uncertainty (+ random exploration) |
| `ambiguous` | uncertainty restricted to the empirical ambiguous band |
| `committee` | RF tree-vote disagreement (vote entropy/variance) |
| `diversity`, `density` | generic diversity / representativeness (uncertainty-based) |
| staged | per-iteration schedule via `staged_schedule` |

## Class-ratio control (the main F1 lever)

Because the RF and the 0.5 threshold are fixed, the training-set composition is the
lever. `resample_to_ratio` oversamples acquired **true** positives to a target
prevalence; `select_final_ratio` chooses that ratio by:

1. training candidate models on the *acquired* set (resampled to each grid ratio),
2. scoring **F1(Left) on the held-out initial 500** (a representative, unbiased
   sample of the pool — unlike the acquired negatives, which are a hard subset),
3. picking the **neighbour-smoothed** best ratio (the small val set is noisy; the
   true curve is a broad plateau).

This keeps ratio selection off the test set. `"auto"` performs this at run time;
`0.80` hardcodes the validated plateau value for speed and stability.

## Reproducing experiments

```bash
python experiments.py <name[,name...]>   # e.g. posprob,posdiv_r080,ratio_auto
python experiments.py all
python analysis.py yield|trajectory|errors|all
```

`experiments.py` reports, per strategy: F1 per seed, mean, std, test precision/recall,
runtime, oracle usage, queried positive-rate, final labeled ratio, selected final
ratio, iterations, and pseudo-label count. `analysis.py` produces the mechanistic
figures/tables behind the design decisions.
