"""
Student implementation file — submit this file only.

Implement run_active_learning(seed) to run your active learning strategy and return
a trained RandomForestClassifier. You may add helper functions in this file only.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

from utils import (
    call_oracle,
    get_oracle_usage,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    train_model,
)

# Import all allowed libraries in advance
import warnings
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import sparse, stats
from scipy.spatial import distance

import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMN = "Employee ID"
TARGET_COLUMN = "Attrition"
ORACLE_BUDGET = 5000
BATCH_SIZE = 500
PSEUDO_THRESHOLD = 0.7
PSEUDO_CAP_PER_CLASS = 2000
PSEUDO_START_FRACTION = 0.5  # fraction of iterations before pseudo-labels activate
DENSITY_KNN = 10       # k for mean-k-NN density estimator
DENSITY_ALPHA = 0.7    # (unused in current strategy; kept for reference)
MINORITY_UPWEIGHT = 3  # duplicate real Left=1 rows this many times in SCORER training
FINAL_MINORITY_UPWEIGHT = 3  # duplicate real Left=1 rows in FINAL model (decoupling tested: 3x best, same as scorer)
PSEUDO_UPWEIGHT = 1    # duplicate pseudo Left rows this many times in training
POST_SELFTRAIN_ROUNDS = 0  # post-hoc self-training (tested 1-2 rounds: hurts; final RF is too Left-biased, adds noisy pseudo)
CALIBRATE_COMMITTEE = False  # tested: hurts badly (0.624 vs 0.658) + 2x slower; isotonic on 69%-Left-skewed data mis-calibrates
COMMITTEE_MODE = "families"  # "families" (RF+LR+HGB) or "bootstrap_rf" (N RFs on bootstrap resamples)
N_BOOTSTRAP_RF = 5           # committee size when COMMITTEE_MODE == "bootstrap_rf"
DIST_TO_LABELED_ALPHA = 1.0  # 1.0 = pure disagreement; <1 blends distance-to-labeled diversity
QUERY_MODE = "qbc"           # "qbc" or "farthest_first" (greedy k-center / Core-Set)
WARMSTART_ITERS = 0    # first N iterations use random sampling instead of QBC (0 = disabled)
TARGET_LEFT_FRACTION = None  # if set, oversample real Left to this fraction instead of integer upweight (tested: worse than integer 3x)
STRAT_LEFT_FRACTION = 0.2     # fraction of each batch reserved for highest-P(Left) minority hunt (raises real Left yield 36% -> 45%)


def _prep_pool_X(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Encode an unlabeled pool DataFrame into the fixed feature space."""
    tmp = pool_df.copy()
    tmp[TARGET_COLUMN] = 0  # dummy label so prepare_xy works
    X_pool, _, _ = prepare_xy(tmp)
    return X_pool


def _proba_left(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(Left=1). Uses predict_proba when available; otherwise
    falls back to sigmoid(decision_function) — needed for SVC(probability=False)."""
    try:
        proba = model.predict_proba(X)
        if model.classes_[1] == 1:
            return proba[:, 1]
        return proba[:, 0]
    except AttributeError:
        decision = model.decision_function(X)
        if model.classes_[1] == 1:
            return 1.0 / (1.0 + np.exp(-decision))
        return 1.0 / (1.0 + np.exp(decision))


def _score_pool(scorer, pool_df: pd.DataFrame) -> np.ndarray:
    """Return P(Left=1) for each row in an unlabeled pool DataFrame."""
    return _proba_left(scorer, _prep_pool_X(pool_df))


def _train_committee(training_set: pd.DataFrame, seed: int) -> list:
    """RF + LR (scaled) + HistGradientBoosting — three inductive-bias families.
    If CALIBRATE_COMMITTEE, wrap each in isotonic CalibratedClassifierCV(cv=3).
    If COMMITTEE_MODE == 'bootstrap_rf', return N RFs on bootstrap resamples."""
    X, y, _ = prepare_xy(training_set)

    if COMMITTEE_MODE == "bootstrap_rf":
        members = []
        n = len(X)
        for i in range(N_BOOTSTRAP_RF):
            rng = np.random.default_rng(seed * 1000 + i)
            idx = rng.integers(0, n, size=n)  # bootstrap resample (with replacement)
            Xi = X.iloc[idx]
            yi = y[idx]
            members.append(
                RandomForestClassifier(n_estimators=100, random_state=seed + i, n_jobs=-1).fit(Xi, yi)
            )
        return members

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    lr = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    hgb = HistGradientBoostingClassifier(random_state=seed)
    if CALIBRATE_COMMITTEE:
        rf = CalibratedClassifierCV(rf, cv=3, method="isotonic")
        lr = CalibratedClassifierCV(lr, cv=3, method="isotonic")
        hgb = CalibratedClassifierCV(hgb, cv=3, method="isotonic")
    return [rf.fit(X, y), lr.fit(X, y), hgb.fit(X, y)]


def _committee_proba_matrix(committee: list, pool_df: pd.DataFrame) -> np.ndarray:
    """Return (n_models, n_samples) matrix of P(Left=1)."""
    X_pool = _prep_pool_X(pool_df)
    return np.array([_proba_left(m, X_pool) for m in committee])


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    """Element-wise Bernoulli entropy H(p) = -p log(p) - (1-p) log(1-p)."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _bald_score(proba_matrix: np.ndarray) -> np.ndarray:
    """BALD = H(mean_p) - mean(H(p_i)). High when the ensemble is uncertain
    but individual members are internally confident (epistemic uncertainty)."""
    p_bar = proba_matrix.mean(axis=0)
    h_mean = _bernoulli_entropy(p_bar)
    mean_h = _bernoulli_entropy(proba_matrix).mean(axis=0)
    return h_mean - mean_h


def _vote_entropy(proba_matrix: np.ndarray) -> np.ndarray:
    """Hard-vote entropy: convert each proba to a Left=1 vote (p > 0.5),
    then compute the entropy of the vote distribution across committee members."""
    votes = (proba_matrix > 0.5).astype(float)
    vote_p1 = votes.mean(axis=0)  # fraction voting Left=1
    return _bernoulli_entropy(vote_p1)


def _kl_to_consensus(proba_matrix: np.ndarray) -> np.ndarray:
    """Mean KL divergence of each member's Bernoulli distribution to the consensus.
    Weights disagreements near 0/1 more strongly than std does."""
    p_bar = proba_matrix.mean(axis=0)
    p = np.clip(proba_matrix, 1e-9, 1.0 - 1e-9)
    p_bar_c = np.clip(p_bar, 1e-9, 1.0 - 1e-9)
    kl_terms = p * np.log(p / p_bar_c) + (1.0 - p) * np.log((1.0 - p) / (1.0 - p_bar_c))
    return kl_terms.mean(axis=0)


def _qbb_disagreement(rf: RandomForestClassifier, X_pool: pd.DataFrame) -> np.ndarray:
    """Query by Bagging: std of P(Left=1) across the RF's individual trees (100-member committee)."""
    X_np = X_pool.to_numpy() if hasattr(X_pool, "to_numpy") else np.asarray(X_pool)
    n = X_np.shape[0]
    probas = np.zeros((len(rf.estimators_), n))
    for i, tree in enumerate(rf.estimators_):
        p = tree.predict_proba(X_np)
        classes = tree.classes_
        if len(classes) == 2:
            idx1 = int(classes[1] == 1)
            probas[i] = p[:, idx1] if classes[idx1] == 1 else p[:, 1 - idx1]
        elif len(classes) == 1:
            probas[i] = float(classes[0] == 1)
    return probas.std(axis=0)


def _mean_knn_distance(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to k nearest neighbors in scaled feature space.
    Larger value = sparser local neighborhood."""
    scaled = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(scaled)
    dists, _ = nn.kneighbors(scaled)
    return dists[:, 1:].mean(axis=1)


def _minmax_norm(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo + 1e-12)


def _dist_to_nearest_labeled(remaining: pd.DataFrame, labeled: pd.DataFrame) -> np.ndarray:
    """Distance from each remaining row to its nearest labeled row (scaled features).
    Larger = farther from what we've already labeled = more novel coverage."""
    Xr = _prep_pool_X(remaining).to_numpy().astype(float)
    Xl = _prep_pool_X(labeled).to_numpy().astype(float)
    scaler = StandardScaler().fit(np.vstack([Xr, Xl]))
    Xr_s = scaler.transform(Xr)
    Xl_s = scaler.transform(Xl)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xl_s)
    dists, _ = nn.kneighbors(Xr_s)
    return dists[:, 0]


def _farthest_first_select(remaining: pd.DataFrame, labeled: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Greedy k-center / Core-Set: iteratively pick the remaining point whose
    minimum distance to the already-selected set (seeded by the labeled set) is
    largest. Returns positional indices into `remaining`."""
    Xr = _prep_pool_X(remaining).to_numpy().astype(float)
    Xl = _prep_pool_X(labeled).to_numpy().astype(float)
    scaler = StandardScaler().fit(np.vstack([Xr, Xl]))
    Xr_s = scaler.transform(Xr)
    Xl_s = scaler.transform(Xl)

    # min distance from each remaining point to the current selected set (start = labeled)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xl_s)
    min_d, _ = nn.kneighbors(Xr_s)
    min_d = min_d[:, 0]

    selected = np.empty(batch_size, dtype=int)
    for k in range(batch_size):
        i = int(np.argmax(min_d))
        selected[k] = i
        # update min distance with the newly selected point
        d_new = np.linalg.norm(Xr_s - Xr_s[i], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[i] = -np.inf  # never reselect
    return selected


def _diverse_select(candidate_X: np.ndarray, candidate_scores: np.ndarray, batch_size: int, seed: int) -> np.ndarray:
    """
    Cluster candidates into `batch_size` k-means groups (in scaled feature space);
    pick the highest-scoring point per cluster. Backfills by descending score if
    any clusters are empty. Returns positional indices into the candidate arrays.
    """
    n = len(candidate_X)
    if n <= batch_size:
        return np.arange(n)

    scaled = StandardScaler().fit_transform(candidate_X)
    km = MiniBatchKMeans(
        n_clusters=batch_size,
        random_state=seed,
        n_init=3,
        batch_size=1024,
    )
    cluster_labels = km.fit_predict(scaled)

    selected: list[int] = []
    used: set[int] = set()
    for c in range(batch_size):
        positions = np.where(cluster_labels == c)[0]
        if len(positions) == 0:
            continue
        best = int(positions[np.argmax(candidate_scores[positions])])
        selected.append(best)
        used.add(best)

    if len(selected) < batch_size:
        for pos in np.argsort(-candidate_scores):
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)
                if len(selected) == batch_size:
                    break

    return np.array(selected)


def _apply_upweights(
    training_set: pd.DataFrame,
    labeled: pd.DataFrame,
    pseudo_pool: pd.DataFrame,
    minority_factor: int,
    pseudo_factor: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Add extra copies of real Left rows and (separately) pseudo Left rows.

    If TARGET_LEFT_FRACTION is set, oversample real Left rows (with replacement)
    to reach that exact class fraction instead of using the integer multiplier.
    """
    if TARGET_LEFT_FRACTION is not None:
        n_total = len(training_set)
        n_left = int((training_set[TARGET_COLUMN] == 1).sum())
        n_stay = n_total - n_left
        # target: n_left_final / (n_left_final + n_stay) = f  =>  n_left_final = f*n_stay/(1-f)
        f = TARGET_LEFT_FRACTION
        target_left = int(round(f * n_stay / (1.0 - f)))
        extra_needed = target_left - n_left
        if extra_needed <= 0:
            return training_set
        real_minority = labeled[labeled[TARGET_COLUMN] == 1]
        if len(real_minority) == 0:
            return training_set
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(real_minority), size=extra_needed)
        extra_rows = real_minority.iloc[idx]
        return pd.concat([training_set, extra_rows], ignore_index=True)

    parts = [training_set]
    if minority_factor > 1:
        real_minority = labeled[labeled[TARGET_COLUMN] == 1]
        if len(real_minority) > 0:
            parts.extend([real_minority] * (minority_factor - 1))
    if pseudo_factor > 1 and pseudo_pool is not None and len(pseudo_pool) > 0:
        parts.extend([pseudo_pool] * (pseudo_factor - 1))
    if len(parts) == 1:
        return training_set
    return pd.concat(parts, ignore_index=True)


def _build_pseudo_pool(scores: np.ndarray, remaining: pd.DataFrame) -> pd.DataFrame:
    """Return confident Left=1 rows given a per-row P(Left) score vector (minority-only pseudo)."""
    if len(remaining) == 0:
        return pd.DataFrame(columns=list(remaining.columns) + [TARGET_COLUMN])

    pos_idx = np.where(scores >= PSEUDO_THRESHOLD)[0]
    pos_idx = pos_idx[np.argsort(-scores[pos_idx])][:PSEUDO_CAP_PER_CLASS]

    if len(pos_idx) == 0:
        return pd.DataFrame(columns=list(remaining.columns) + [TARGET_COLUMN])
    rows = remaining.iloc[pos_idx].copy()
    rows[TARGET_COLUMN] = 1
    return rows


def run_active_learning(seed: int):
    """
    Iterative uncertainty sampling with RF as scorer.
    From iteration PSEUDO_START_ITER onward, augment training with confident
    pseudo-labels (p >= PSEUDO_THRESHOLD, cap PSEUDO_CAP_PER_CLASS per class);
    pseudo-labels are refreshed each iteration.
    """
    labeled = load_initial_labeled(seed)
    pool = load_pool()

    labeled_ids = set(labeled[ID_COLUMN].astype(str))
    remaining = pool.loc[~pool[ID_COLUMN].astype(str).isin(labeled_ids)].reset_index(drop=True)

    n_iterations = ORACLE_BUDGET // BATCH_SIZE
    pseudo_start_iter = max(1, int(round(n_iterations * PSEUDO_START_FRACTION)))
    pseudo_pool = pd.DataFrame(columns=list(labeled.columns))

    for iteration in range(n_iterations):
        if iteration < WARMSTART_ITERS:
            # Warm-start: random sampling for the first N batches to seed diverse coverage
            # before we have enough labels for the committee to score reliably.
            rng = np.random.default_rng(seed + iteration)
            top_idx = rng.choice(len(remaining), size=BATCH_SIZE, replace=False)
            committee = None  # not built during warm-start
        else:
            if len(pseudo_pool):
                training_set = pd.concat([labeled, pseudo_pool], ignore_index=True)
            else:
                training_set = labeled
            training_set = _apply_upweights(
                training_set, labeled, pseudo_pool, MINORITY_UPWEIGHT, PSEUDO_UPWEIGHT, seed=seed
            )

            committee = _train_committee(training_set, seed)
            proba_matrix = _committee_proba_matrix(committee, remaining)
            disagreement = proba_matrix.std(axis=0)

            if QUERY_MODE == "farthest_first":
                # Pure Core-Set: ignore disagreement, pick a geometrically diverse batch.
                top_idx = _farthest_first_select(remaining, labeled, BATCH_SIZE)
                chosen_ids = remaining.iloc[top_idx][ID_COLUMN].astype(str).tolist()
                new_labeled = call_oracle(chosen_ids)
                labeled = pd.concat([labeled, new_labeled], ignore_index=True)
                remaining = remaining.drop(index=remaining.index[top_idx]).reset_index(drop=True)
                if iteration + 1 >= pseudo_start_iter and committee is not None:
                    consensus_left = _committee_proba_matrix(committee, remaining).mean(axis=0)
                    pseudo_pool = _build_pseudo_pool(consensus_left, remaining)
                continue

            if DIST_TO_LABELED_ALPHA < 1.0:
                # Blend distance-to-labeled diversity into the disagreement score.
                dist = _dist_to_nearest_labeled(remaining, labeled)
                disagreement = (
                    DIST_TO_LABELED_ALPHA * _minmax_norm(disagreement)
                    + (1.0 - DIST_TO_LABELED_ALPHA) * _minmax_norm(dist)
                )

            if STRAT_LEFT_FRACTION > 0.0:
                # Reserve part of the batch for highest-P(Left) rows (minority hunt),
                # the rest for highest-disagreement rows (boundary).
                n_left_hunt = int(BATCH_SIZE * STRAT_LEFT_FRACTION)
                n_disagree = BATCH_SIZE - n_left_hunt
                mean_p_left = proba_matrix.mean(axis=0)
                disagree_idx = np.argsort(-disagreement)[:n_disagree]
                chosen_mask = np.zeros(len(remaining), dtype=bool)
                chosen_mask[disagree_idx] = True
                # From the not-yet-chosen rows, take the highest P(Left).
                remaining_order = np.argsort(-mean_p_left)
                left_hunt_idx = [i for i in remaining_order if not chosen_mask[i]][:n_left_hunt]
                top_idx = np.concatenate([disagree_idx, np.array(left_hunt_idx, dtype=int)])
            else:
                top_idx = np.argsort(-disagreement)[:BATCH_SIZE]

        chosen_ids = remaining.iloc[top_idx][ID_COLUMN].astype(str).tolist()

        new_labeled = call_oracle(chosen_ids)
        labeled = pd.concat([labeled, new_labeled], ignore_index=True)
        remaining = remaining.drop(index=remaining.index[top_idx]).reset_index(drop=True)

        # Refresh pseudo-labels via committee consensus (mean P(Left) across members).
        # Skipped during warm-start (no committee).
        if iteration + 1 >= pseudo_start_iter and committee is not None:
            proba_matrix_remaining = _committee_proba_matrix(committee, remaining)
            consensus_left = proba_matrix_remaining.mean(axis=0)
            pseudo_pool = _build_pseudo_pool(consensus_left, remaining)

    if len(pseudo_pool):
        final_training = pd.concat([labeled, pseudo_pool], ignore_index=True)
    else:
        final_training = labeled
    final_training = _apply_upweights(
        final_training, labeled, pseudo_pool, FINAL_MINORITY_UPWEIGHT, PSEUDO_UPWEIGHT, seed=seed
    )

    X_train, y_train, train_ids = prepare_xy(final_training)
    model = train_model(X_train, y_train, train_ids, seed=seed)

    # Optional post-hoc self-training: use the final model's own predictions to
    # add fresh confident-Left pseudo-labels, then retrain. Uses the actual returned
    # model (not the committee) as the pseudo source.
    for _ in range(POST_SELFTRAIN_ROUNDS):
        if len(remaining) == 0:
            break
        proba_left = _score_pool(model, remaining)
        extra_pseudo = _build_pseudo_pool(proba_left, remaining)
        if len(extra_pseudo) == 0:
            break
        pseudo_pool = pd.concat([pseudo_pool, extra_pseudo], ignore_index=True)
        extra_ids = set(extra_pseudo[ID_COLUMN].astype(str))
        remaining = remaining.loc[~remaining[ID_COLUMN].astype(str).isin(extra_ids)].reset_index(drop=True)

        final_training = pd.concat([labeled, pseudo_pool], ignore_index=True)
        final_training = _apply_upweights(
            final_training, labeled, pseudo_pool, FINAL_MINORITY_UPWEIGHT, PSEUDO_UPWEIGHT, seed=seed
        )
        X_train, y_train, train_ids = prepare_xy(final_training)
        model = train_model(X_train, y_train, train_ids, seed=seed)

    return model
