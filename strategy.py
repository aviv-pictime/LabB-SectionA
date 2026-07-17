"""
Student implementation file — submit this file only.

Active-learning strategy for the fixed RandomForest attrition classifier, aimed at
maximizing F1 of the minority "Left" class. Because the classifier and its fixed 0.5
decision threshold cannot be changed, the only lever is the training data — so the
whole strategy is built around the class imbalance.

Pipeline (per seed) — 10 rounds of 500 oracle queries (5,000 total):
  1. Build the training set = current real labels + confident pseudo-labels, with the
     real "Left" rows up-weighted 3x.
  2. Train a 3-model committee (RandomForest + LogisticRegression +
     HistGradientBoosting) and score every remaining pool row.
  3. Query 500 rows: 80% with the highest committee disagreement (std of P(Left)),
     20% with the highest committee-mean P(Left) — a deliberate "minority hunt" that
     accumulates real "Left" labels feeding the up-weighting in later rounds.
  4. From the halfway round on, refresh minority-only pseudo-labels: pool rows whose
     committee-mean P(Left) >= 0.7 are added as free "Left" labels (capped).
Finally, train the fixed RandomForest on all real labels + pseudo-labels (real "Left"
up-weighted 3x) and return it. Runs well under the 60s/seed limit.

The committee is used two ways at once: disagreement (std) selects which rows to query;
agreement (mean) selects which rows to trust as free pseudo-labels. Variants that were
tried and dropped (diversity/geometry sampling, larger committees, calibration,
symmetric pseudo-labels, alternative disagreement measures, ...) are documented in
process_summary.md.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import call_oracle, load_initial_labeled, load_pool, prepare_xy, train_model


ID_COLUMN = "Employee ID"
TARGET_COLUMN = "Attrition"

ORACLE_BUDGET = 5000          # total unique oracle queries allowed
BATCH_SIZE = 500              # queries per round -> ORACLE_BUDGET // BATCH_SIZE = 10 rounds
STRAT_LEFT_FRACTION = 0.2     # fraction of each batch reserved for the "minority hunt"
PSEUDO_START_FRACTION = 0.5   # pseudo-labels activate at this fraction of the rounds
PSEUDO_THRESHOLD = 0.7        # committee-mean P(Left) required to pseudo-label a row "Left"
PSEUDO_CAP_PER_CLASS = 2000   # max pseudo-labels added in a round
MINORITY_UPWEIGHT = 3         # duplicate each real "Left" row this many times in training


def _prep_pool_X(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Encode an unlabeled pool DataFrame into the fixed feature space."""
    tmp = pool_df.copy()
    tmp[TARGET_COLUMN] = 0  # dummy label so prepare_xy runs; the label is discarded
    X_pool, _, _ = prepare_xy(tmp)
    return X_pool


def _proba_left(model, X: pd.DataFrame) -> np.ndarray:
    """P(Left=1) from a fitted classifier's predict_proba."""
    proba = model.predict_proba(X)
    return proba[:, 1] if model.classes_[1] == 1 else proba[:, 0]


def _train_committee(training_set: pd.DataFrame, seed: int) -> list:
    """Three inductive-bias families: RandomForest + LogisticRegression (scaled) +
    HistGradientBoosting. Their agreement/disagreement drives querying and pseudo-labeling."""
    X, y, _ = prepare_xy(training_set)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(X, y)
    lr = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, random_state=seed),
    ).fit(X, y)
    hgb = HistGradientBoostingClassifier(random_state=seed).fit(X, y)
    return [rf, lr, hgb]


def _committee_proba_matrix(committee: list, pool_df: pd.DataFrame) -> np.ndarray:
    """(n_models, n_rows) matrix of P(Left=1) for each committee member over the pool."""
    X_pool = _prep_pool_X(pool_df)
    return np.array([_proba_left(m, X_pool) for m in committee])


def _upweight_minority(training_set: pd.DataFrame, labeled: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Append (factor - 1) extra copies of the real "Left" rows (taken from `labeled`),
    pushing the training set toward the minority class to shift the fixed-threshold model
    toward higher recall. Pseudo-labels are NOT up-weighted (only real labels are)."""
    if factor <= 1:
        return training_set
    real_minority = labeled[labeled[TARGET_COLUMN] == 1]
    if len(real_minority) == 0:
        return training_set
    return pd.concat([training_set] + [real_minority] * (factor - 1), ignore_index=True)


def _build_pseudo_pool(consensus_left: np.ndarray, remaining: pd.DataFrame) -> pd.DataFrame:
    """Minority-only pseudo-labels: remaining rows whose committee-mean P(Left) is at
    least PSEUDO_THRESHOLD, most-confident first, capped at PSEUDO_CAP_PER_CLASS, all
    labeled "Left". Returns an empty (correctly-columned) frame when none qualify."""
    empty = pd.DataFrame(columns=list(remaining.columns) + [TARGET_COLUMN])
    if len(remaining) == 0:
        return empty
    pos_idx = np.where(consensus_left >= PSEUDO_THRESHOLD)[0]
    pos_idx = pos_idx[np.argsort(-consensus_left[pos_idx])][:PSEUDO_CAP_PER_CLASS]
    if len(pos_idx) == 0:
        return empty
    rows = remaining.iloc[pos_idx].copy()
    rows[TARGET_COLUMN] = 1
    return rows


def run_active_learning(seed: int):
    """
    Run the active-learning strategy for one seed and return the trained RandomForest.
    See the module docstring for the full pipeline; the steps below are numbered to match.
    """
    labeled = load_initial_labeled(seed)
    pool = load_pool()
    labeled_ids = set(labeled[ID_COLUMN].astype(str))
    remaining = pool.loc[~pool[ID_COLUMN].astype(str).isin(labeled_ids)].reset_index(drop=True)

    n_iterations = ORACLE_BUDGET // BATCH_SIZE
    pseudo_start_iter = max(1, int(round(n_iterations * PSEUDO_START_FRACTION)))
    n_minority_hunt = int(BATCH_SIZE * STRAT_LEFT_FRACTION)
    n_disagree = BATCH_SIZE - n_minority_hunt
    pseudo_pool = pd.DataFrame(columns=list(labeled.columns))

    for iteration in range(n_iterations):
        # 1. Training set = real labels + confident pseudo-labels, real "Left" up-weighted.
        if len(pseudo_pool):
            training_set = pd.concat([labeled, pseudo_pool], ignore_index=True)
        else:
            training_set = labeled
        training_set = _upweight_minority(training_set, labeled, MINORITY_UPWEIGHT)

        # 2. Committee scores every remaining pool row.
        committee = _train_committee(training_set, seed)
        proba_matrix = _committee_proba_matrix(committee, remaining)  # (3, n_remaining)
        disagreement = proba_matrix.std(axis=0)
        mean_p_left = proba_matrix.mean(axis=0)

        # 3. Stratified query: n_disagree by disagreement + n_minority_hunt by highest P(Left).
        disagree_idx = np.argsort(-disagreement)[:n_disagree]
        already_chosen = np.zeros(len(remaining), dtype=bool)
        already_chosen[disagree_idx] = True
        hunt_idx = [i for i in np.argsort(-mean_p_left) if not already_chosen[i]][:n_minority_hunt]
        top_idx = np.concatenate([disagree_idx, np.array(hunt_idx, dtype=int)])

        # 4. Query the oracle; move the newly labeled rows out of the pool.
        chosen_ids = remaining.iloc[top_idx][ID_COLUMN].astype(str).tolist()
        new_labeled = call_oracle(chosen_ids)
        labeled = pd.concat([labeled, new_labeled], ignore_index=True)
        remaining = remaining.drop(index=remaining.index[top_idx]).reset_index(drop=True)

        # 5. From the halfway round on, refresh minority-only consensus pseudo-labels.
        if iteration + 1 >= pseudo_start_iter:
            consensus_left = _committee_proba_matrix(committee, remaining).mean(axis=0)
            pseudo_pool = _build_pseudo_pool(consensus_left, remaining)

    # Final model: all real labels + pseudo-labels, real "Left" up-weighted, fixed RF.
    if len(pseudo_pool):
        final_training = pd.concat([labeled, pseudo_pool], ignore_index=True)
    else:
        final_training = labeled
    final_training = _upweight_minority(final_training, labeled, MINORITY_UPWEIGHT)

    X_train, y_train, train_ids = prepare_xy(final_training)
    return train_model(X_train, y_train, train_ids, seed=seed)
