"""
Dev-only analysis for Section A (NOT submitted). Studies *why* positive-
probability sampling works and where F1(Left) is limited, to motivate improved
positive-oriented Active Learning strategies. Uses only pool data + oracle
labels on samples we would query anyway; never uses the test set for decisions.

Usage:
    python analysis.py yield        # honest positive-yield curve vs P(Left)
    python analysis.py trajectory   # per-iteration posprob dynamics
    python analysis.py errors       # false-negative characterization
    python analysis.py all
"""

from __future__ import annotations

import sys
import time
from typing import List

import numpy as np
import pandas as pd

import strategy as S
import utils

SEEDS = [1, 2, 3]
ID = "Employee ID"
TGT = "Attrition"


def _labels01(rows: pd.DataFrame) -> np.ndarray:
    y = rows[TGT]
    return (y == "Left").astype(int).to_numpy() if y.dtype == object else y.astype(int).to_numpy()


# ---------------------------------------------------------------------------
# 1) Honest positive-yield curve vs predicted P(Left)
# ---------------------------------------------------------------------------
def yield_curve(seed: int, probe: int = 3000) -> None:
    """Train on init 500, then label a RANDOM probe of the pool to measure the
    true positive rate as a function of predicted P(Left). Random probe => the
    yield/error estimates are unbiased for the pool distribution."""
    utils.reset_oracle()
    utils.set_active_seed(seed)
    labeled, pool = S.load_labeled_and_pool(seed)
    X_pool = S.encode_features_by_id(pool)
    model = S.fit_model(labeled, seed)

    init_ids = set(labeled[ID].astype(str))
    cand = sorted(set(pool[ID].astype(str)) - init_ids, key=int)
    rng = np.random.RandomState(seed)
    probe_ids = list(rng.choice(cand, size=probe, replace=False))
    rows = utils.call_oracle(probe_ids)
    y = _labels01(rows)
    p = S.predict_pos_proba(model, X_pool.loc[[str(i) for i in rows[ID].astype(str)]])

    print(f"\n== seed {seed}: yield curve (initial model, random probe n={probe}) ==")
    edges = np.arange(0.0, 1.0001, 0.1)
    print(f"{'band':>12} {'count':>6} {'true_pos_rate':>13} {'cum_pos_from_top':>16}")
    order = np.argsort(-p)
    cum = 0
    # cumulative positives if we selected the top-k by P(Left)
    cumtable = {}
    for k in (500, 1000, 2000, 3000):
        kk = min(k, len(p))
        cumtable[k] = int(y[order[:kk]].sum())
    for b in range(len(edges) - 1):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < len(edges) - 2 else p <= 1.0)
        c = int(m.sum())
        tpr = float(y[m].mean()) if c else float("nan")
        print(f"  [{edges[b]:.1f},{edges[b+1]:.1f}) {c:6d} {tpr:13.3f}")
    print("  cumulative true-positives among top-k by P(Left):")
    for k, v in cumtable.items():
        kk = min(k, len(p)); print(f"     top {k:5d}: {v} positives  (yield {v/kk:.3f})")
    # recall view: where do the TRUE positives sit in probability?
    pos_p = p[y == 1]
    print(f"  TRUE positives: n={len(pos_p)}  frac with P>=0.5={np.mean(pos_p>=0.5):.3f} "
          f"(these are recalled)  median P={np.median(pos_p):.3f}")
    print(f"  FALSE negatives (true Left, P<0.5): n={int(np.sum((y==1)&(p<0.5)))}  "
          f"their P range ~[{p[(y==1)&(p<0.5)].min():.2f},{p[(y==1)&(p<0.5)].max():.2f}]")


# ---------------------------------------------------------------------------
# 2) Per-iteration posprob trajectory
# ---------------------------------------------------------------------------
def _oof_prf(labeled: pd.DataFrame, seed: int, n_splits: int = 3):
    """Out-of-fold precision/recall/F1 for Left on the current labeled set."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    oof = S.out_of_fold_proba(labeled, seed, n_splits)
    _, y, _ = utils.prepare_xy(labeled)
    pred = (oof >= 0.5).astype(int)
    return (precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0))


def trajectory(seed: int, batch: int = 500, budget: int = 5000) -> None:
    """Run posprob with instrumentation; report per-iteration dynamics."""
    utils.reset_oracle()
    utils.set_active_seed(seed)
    labeled, pool = S.load_labeled_and_pool(seed)
    X_pool = S.encode_features_by_id(pool)
    cand = set(pool[ID].astype(str)) - set(labeled[ID].astype(str))

    print(f"\n== seed {seed}: posprob trajectory (batch={batch}) ==")
    print(f"{'it':>2} {'sel_P_min':>9} {'sel_P_mean':>10} {'batch_TPR':>9} "
          f"{'lab_pos%':>8} {'rem_P>0.5%':>10} {'oofP':>6} {'oofR':>6} {'oofF1':>6}")
    spent = 0
    it = 0
    while spent < budget and cand:
        model = S.fit_model(labeled, seed)
        cids = sorted(cand, key=int)
        p = S.predict_pos_proba(model, X_pool.loc[cids])
        k = min(batch, budget - spent)
        idx = np.argsort(-p)[:k]
        picks = [cids[i] for i in idx]
        sel_p = p[idx]
        rem_hi = float(np.mean(p >= 0.5))
        rows = utils.call_oracle(picks)
        yb = _labels01(rows)
        labeled = S.merge_labeled(labeled, rows)
        cand -= set(picks)
        spent += len(picks)
        it += 1
        _, yl, _ = utils.prepare_xy(labeled)
        lab_pos = float(yl.mean())
        if it % 2 == 1 or it <= 2:  # OOF is costly; sample iterations
            pr, rc, f1 = _oof_prf(labeled, seed)
        else:
            pr = rc = f1 = float("nan")
        print(f"{it:2d} {sel_p.min():9.3f} {sel_p.mean():10.3f} {yb.mean():9.3f} "
              f"{lab_pos:8.3f} {rem_hi:10.3f} {pr:6.3f} {rc:6.3f} {f1:6.3f}")


# ---------------------------------------------------------------------------
# 3) False-negative characterization
# ---------------------------------------------------------------------------
def errors(seed: int, warm_queries: int = 2500) -> None:
    """Build a posprob labeled set, then characterize OOF false negatives."""
    utils.reset_oracle()
    utils.set_active_seed(seed)
    labeled, pool = S.load_labeled_and_pool(seed)
    X_pool = S.encode_features_by_id(pool)
    cand = set(pool[ID].astype(str)) - set(labeled[ID].astype(str))
    spent = 0
    while spent < warm_queries and cand:
        model = S.fit_model(labeled, seed)
        cids = sorted(cand, key=int)
        p = S.predict_pos_proba(model, X_pool.loc[cids])
        idx = np.argsort(-p)[:500]
        picks = [cids[i] for i in idx]
        rows = utils.call_oracle(picks)
        labeled = S.merge_labeled(labeled, rows)
        cand -= set(picks); spent += len(picks)

    oof = S.out_of_fold_proba(labeled, seed, 5)
    X, y, ids = utils.prepare_xy(labeled)
    pred = (oof >= 0.5).astype(int)
    fn = (y == 1) & (pred == 0)
    tp = (y == 1) & (pred == 1)
    tn = (y == 0) & (pred == 0)
    print(f"\n== seed {seed}: error analysis on posprob labeled set (n={len(y)}) ==")
    print(f"  labeled pos ratio={y.mean():.3f}")
    print(f"  OOF: TP={int(tp.sum())} FN={int(fn.sum())} recall={tp.sum()/max(1,(y==1).sum()):.3f}")
    print(f"  FN predicted-P: mean={oof[fn].mean():.3f} "
          f"[{oof[fn].min():.2f},{oof[fn].max():.2f}]  (all <0.5 by definition)")
    # feature contrast: which features separate FN from TP (both true Left)?
    Xf = X.reset_index(drop=True)
    diff = (Xf[fn].mean() - Xf[tp].mean())
    diff = diff.reindex(diff.abs().sort_values(ascending=False).index)
    print("  top features where FN (missed Left) differ from TP (caught Left):")
    for name, val in diff.head(8).items():
        print(f"     {name:35s} FN-TP mean diff = {val:+.3f}")


def _resample_to_ratio(X, y, ids, r, rng, mode="oversample"):
    """Return (X,y,ids) resampled so positive ratio ~= r (undersample or oversample)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if mode == "undersample":
        # keep all of the class that must shrink less; undersample the other
        # target: npos/(npos+nneg)=r
        npos, nneg = len(pos), len(neg)
        if npos / (npos + nneg) < r:      # need fewer negatives
            keep_neg = int(npos * (1 - r) / r)
            neg = rng.choice(neg, size=min(nneg, keep_neg), replace=False)
        else:                              # need fewer positives
            keep_pos = int(nneg * r / (1 - r))
            pos = rng.choice(pos, size=min(npos, keep_pos), replace=False)
        idx = np.concatenate([pos, neg])
    else:  # oversample the minority-vs-target class with duplicates
        npos, nneg = len(pos), len(neg)
        if npos / (npos + nneg) < r:       # need more positives
            target_pos = int(nneg * r / (1 - r))
            extra = rng.choice(pos, size=max(0, target_pos - npos), replace=True)
            idx = np.concatenate([pos, neg, extra])
        else:                              # need more negatives
            target_neg = int(npos * (1 - r) / r)
            extra = rng.choice(neg, size=max(0, target_neg - nneg), replace=True)
            idx = np.concatenate([pos, neg, extra])
    rng.shuffle(idx)
    Xr = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    return Xr, y[idx], np.asarray(ids)[idx]


def ratio_scan(seed: int, budget: int = 5000, mode: str = "oversample") -> None:
    """Acquire via posprob, then EXPLORE final-training positive ratios vs test F1.
    Exploratory only (final ratio will be chosen via internal validation)."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    utils.reset_oracle(); utils.set_active_seed(seed)
    labeled, pool = S.load_labeled_and_pool(seed)
    X_pool = S.encode_features_by_id(pool)
    cand = set(pool[ID].astype(str)) - set(labeled[ID].astype(str))
    spent = 0
    while spent < budget and cand:
        model = S.fit_model(labeled, seed)
        cids = sorted(cand, key=int)
        p = S.predict_pos_proba(model, X_pool.loc[cids])
        idx = np.argsort(-p)[:min(500, budget - spent)]
        picks = [cids[i] for i in idx]
        labeled = S.merge_labeled(labeled, utils.call_oracle(picks))
        cand -= set(picks); spent += len(picks)

    X, y, ids = utils.prepare_xy(labeled)
    test = utils.load_test(seed); Xt, yt, _ = utils.prepare_xy(test)
    rng = np.random.RandomState(seed)
    nat = y.mean()
    print(f"\n== seed {seed}: final-training ratio scan (mode={mode}, natural={nat:.3f}) ==")
    for r in [None, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
        if r is None:
            Xr, yr, ir = X, y, ids
            tag = f"natural({nat:.2f})"
        else:
            Xr, yr, ir = _resample_to_ratio(X, y, ids, r, np.random.RandomState(seed), mode)
            tag = f"r={r:.2f}"
        m = utils.train_model(Xr, yr, ir, seed=seed)
        pred = m.predict(Xt)
        f1 = f1_score(yt, pred, pos_label=1)
        pr = precision_score(yt, pred, pos_label=1, zero_division=0)
        rc = recall_score(yt, pred, pos_label=1, zero_division=0)
        print(f"   {tag:14s} n={len(yr):5d} pos={int(yr.sum()):5d}  "
              f"TEST P={pr:.3f} R={rc:.3f} F1={f1:.4f}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which.startswith("ratio"):
        mode = "undersample" if "under" in which else "oversample"
        for seed in SEEDS:
            ratio_scan(seed, mode=mode)
        print("\n[ratio scan done]")
        return
    t0 = time.perf_counter()
    for seed in SEEDS:
        if which in ("yield", "all"):
            yield_curve(seed)
        if which in ("trajectory", "all"):
            trajectory(seed)
        if which in ("errors", "all"):
            errors(seed)
    print(f"\n[analysis done in {time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
