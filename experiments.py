"""
Dev-only experiment harness for Section A (NOT submitted, does not modify the
framework). Runs Active Learning strategies from strategy.py across seeds and
prints the reporting metrics requested for the project (F1 per seed, mean F1,
runtime, oracle usage, positive rate among queried samples, confidence-region
statistics, etc.).

Usage:
    python experiments.py <experiment_name>
where <experiment_name> is one of the keys in EXPERIMENTS, or "all".
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np

import strategy as S
import utils


SEEDS = [1, 2, 3]


def run_strategy(name: str, config: S.StrategyConfig) -> Dict:
    """Run one config over all seeds; return aggregated metrics."""
    from sklearn.metrics import precision_score, recall_score
    per_seed = []
    for seed in SEEDS:
        utils.reset_oracle()
        utils.set_active_seed(seed)
        t0 = time.perf_counter()
        model = S.run_pipeline(seed, config)
        elapsed = time.perf_counter() - t0
        f1 = utils.evaluate_model(model, seed)
        test = utils.load_test(seed)
        Xt, yt, _ = utils.prepare_xy(test)
        pred = model.predict(Xt)
        usage = utils.get_oracle_usage()
        st = dict(S.LAST_RUN_STATS)
        qtot = st.get("queried_total", 0) or 1
        per_seed.append(
            {
                "seed": seed,
                "f1": f1,
                "test_p": precision_score(yt, pred, pos_label=1, zero_division=0),
                "test_r": recall_score(yt, pred, pos_label=1, zero_division=0),
                "runtime": elapsed,
                "queried": usage["unique_queried"],
                "queried_pos": st.get("queried_pos", 0),
                "pos_rate": st.get("queried_pos", 0) / qtot,
                "lab_ratio": st.get("labeled_pos_ratio", float("nan")),
                "final_ratio": st.get("final_ratio", None),
                "n_iter": st.get("n_iterations", 0),
                "pseudo": st.get("pseudo_count", 0),
            }
        )
    return {"name": name, "config": config, "per_seed": per_seed}


def print_result(res: Dict) -> None:
    ps = res["per_seed"]
    cfg = res["config"]
    f1s = [r["f1"] for r in ps]
    print(f"\n=== {res['name']} ===")
    print(f"    batch_size={cfg.batch_size}  strategy={cfg.strategy}"
          f"  hybrid={cfg.hybrid_proportions}  pseudo={cfg.use_pseudo_labels}")
    for r in ps:
        print(
            f"    seed{r['seed']}: F1={r['f1']:.4f} P={r['test_p']:.3f} R={r['test_r']:.3f}  "
            f"t={r['runtime']:.1f}s  qpos_rate={r['pos_rate']:.3f} "
            f"lab_ratio={r['lab_ratio']:.3f} final_r={r['final_ratio']}  "
            f"iters={r['n_iter']}  pseudo={r['pseudo']}"
        )
    print(f"    MEAN F1={np.mean(f1s):.4f}  (std={np.std(f1s):.4f})  "
          f"mean_t={np.mean([r['runtime'] for r in ps]):.1f}s")


def confidence_analysis() -> None:
    """Inspect predicted-probability reliability from OOF on the INITIAL set,
    and the empirically determined confidence boundaries per seed."""
    print("\n########## Confidence-region / reliability analysis (OOF on init 500) ##########")
    for seed in SEEDS:
        utils.reset_oracle()
        utils.set_active_seed(seed)
        labeled = utils.load_initial_labeled(seed)
        _, y, _ = utils.prepare_xy(labeled)
        oof = S.out_of_fold_proba(labeled, seed, n_splits=5)
        tab = S.reliability_by_bin(y, oof, n_bins=10)
        print(f"\n-- seed {seed}: OOF reliability table (init 500) --")
        print(tab.to_string(index=False,
              formatters={"bin_low": "{:.1f}".format, "bin_high": "{:.1f}".format,
                          "pos_rate": "{:.3f}".format, "mean_p": "{:.3f}".format}))
        for tr in (0.85, 0.90, 0.95):
            t_low, t_high = S.determine_confidence_boundaries(y, oof, tr, tr, 50)
            rep = S.region_report(y, oof, t_low, t_high)
            print(f"   target={tr:.2f}: t_low={rep['t_low']:.3f} t_high={rep['t_high']:.3f} "
                  f"| Stayed n={rep['n_stayed']} acc={rep['acc_stayed']} "
                  f"| Amb n={rep['n_ambiguous']} "
                  f"| Left n={rep['n_left']} acc={rep['acc_left']}")


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
def _cfg(**kw) -> S.StrategyConfig:
    return S.StrategyConfig(**kw)


EXPERIMENTS = {
    # 1-3: baselines reproduced in the modular pipeline
    "random": lambda: run_strategy("random (b500)", _cfg(strategy="random", batch_size=500)),
    "uncertainty": lambda: run_strategy("uncertainty (b500)", _cfg(strategy="uncertainty", batch_size=500)),
    "posprob": lambda: run_strategy("posprob (b500)", _cfg(strategy="posprob", batch_size=500)),
    # 4-5: simple hybrids
    "hybrid_pu": lambda: run_strategy("hybrid_pu 50/50", _cfg(strategy="hybrid_pu", batch_size=500)),
    "hybrid_pur": lambda: run_strategy("hybrid_pur 50/35/15", _cfg(strategy="hybrid_pur", batch_size=500)),
    # 6-8: confidence-region driven
    "ambiguous": lambda: run_strategy("ambiguous band", _cfg(strategy="ambiguous", batch_size=500)),
    "staged": lambda: run_strategy(
        "staged posprob->ambiguous",
        _cfg(strategy="posprob", batch_size=500,
             staged_schedule=[(0.5, "posprob"), (1.0, "ambiguous")])),
    # 9-11: advanced
    "committee": lambda: run_strategy("committee vote-entropy", _cfg(strategy="committee", batch_size=500)),
    "diversity": lambda: run_strategy("diversity(unc+kmeans)", _cfg(strategy="diversity", batch_size=500)),
    "density": lambda: run_strategy("density(unc*repr)", _cfg(strategy="density", batch_size=500)),
    "pseudo": lambda: run_strategy(
        "posprob + pseudo",
        _cfg(strategy="posprob", batch_size=500, use_pseudo_labels=True)),
    # batch-size sweep on the current best single strategy
    "batch250": lambda: run_strategy("posprob b250", _cfg(strategy="posprob", batch_size=250)),
    "batch1000": lambda: run_strategy("posprob b1000", _cfg(strategy="posprob", batch_size=1000)),
    # class-ratio control on the final model (new investigation)
    "ratio_auto": lambda: run_strategy(
        "posprob + ratio=auto", _cfg(strategy="posprob", batch_size=500, final_pos_ratio="auto")),
    "ratio080": lambda: run_strategy(
        "posprob + ratio=0.80", _cfg(strategy="posprob", batch_size=500, final_pos_ratio=0.80)),
    "ratio085": lambda: run_strategy(
        "posprob + ratio=0.85", _cfg(strategy="posprob", batch_size=500, final_pos_ratio=0.85)),
    "ratio075": lambda: run_strategy(
        "posprob + ratio=0.75", _cfg(strategy="posprob", batch_size=500, final_pos_ratio=0.75)),
    "ratio_auto_val": lambda: run_strategy(
        "posprob + ratio=auto + val1000",
        _cfg(strategy="posprob", batch_size=500, final_pos_ratio="auto", val_reserve=1000)),
    "ratio_auto_val1500": lambda: run_strategy(
        "posprob + ratio=auto + val1500",
        _cfg(strategy="posprob", batch_size=500, final_pos_ratio="auto", val_reserve=1500)),
    "ratio_auto_free": lambda: run_strategy(
        "posprob + ratio=auto (init-val)",
        _cfg(strategy="posprob", batch_size=500, final_pos_ratio="auto")),
    # positive-diverse acquisition (diversity only within top-P candidates)
    "posdiv": lambda: run_strategy(
        "pos_diverse (shortlist x3)", _cfg(strategy="pos_diverse", batch_size=500)),
    "posdiv_r080": lambda: run_strategy(
        "pos_diverse + ratio=0.80",
        _cfg(strategy="pos_diverse", batch_size=500, final_pos_ratio=0.80)),
    "posdiv_r075_x5": lambda: run_strategy(
        "pos_diverse x5 + ratio=0.75",
        _cfg(strategy="pos_diverse", batch_size=500, final_pos_ratio=0.75, shortlist_mult=5)),
    "posdiv_auto": lambda: run_strategy(
        "pos_diverse + ratio=auto (init-val)",
        _cfg(strategy="pos_diverse", batch_size=500, final_pos_ratio="auto")),
    "posprob_auto": lambda: run_strategy(
        "posprob + ratio=auto (init-val,smoothed)",
        _cfg(strategy="posprob", batch_size=500, final_pos_ratio="auto")),
    "posdiv_b1000_auto": lambda: run_strategy(
        "pos_diverse b1000 + ratio=auto",
        _cfg(strategy="pos_diverse", batch_size=1000, final_pos_ratio="auto")),
    "posdiv_b1000_r080": lambda: run_strategy(
        "pos_diverse b1000 + ratio=0.80",
        _cfg(strategy="pos_diverse", batch_size=1000, final_pos_ratio=0.80)),
    "posdiv_b750_auto": lambda: run_strategy(
        "pos_diverse b750 + ratio=auto",
        _cfg(strategy="pos_diverse", batch_size=750, final_pos_ratio="auto")),
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "confidence":
        confidence_analysis()
        return
    names = list(EXPERIMENTS) if which == "all" else which.split(",")
    results = []
    for n in names:
        if n not in EXPERIMENTS:
            print(f"[skip] unknown experiment: {n}")
            continue
        res = EXPERIMENTS[n]()
        print_result(res)
        results.append(res)
    if len(results) > 1:
        print("\n########## SUMMARY ##########")
        rows = sorted(results, key=lambda r: -np.mean([x["f1"] for x in r["per_seed"]]))
        for r in rows:
            f1s = [x["f1"] for x in r["per_seed"]]
            print(f"  {np.mean(f1s):.4f}  (std {np.std(f1s):.4f})  {r['name']}")


if __name__ == "__main__":
    main()
