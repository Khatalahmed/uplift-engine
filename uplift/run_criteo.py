"""
run_criteo.py

WHAT : The same pipeline as run_synthetic.py, on the real Criteo trial
       (~14M users, ~85% treated, outcome = site visit).
WHY  : Synthetic data proved the methods recover a planted answer. Real
       data has no answer key - only the Qini curve can judge the models -
       but it is where the methods have to work.
FLOW : SQL over all rows: arm sizes, SRM, balance, average effect
       -> fold 0 (~10%): train models
       -> folds 5-9 (~50%): CUPED effect + Qini / uplift@k for every model
       -> results/criteo_report.json
LOGIC:
  - Folds are frozen in the Parquet file (data.to_parquet), so train and
    evaluation rows never overlap and every run sees the same split.
  - Criteo pooled SEVERAL tests, so the chance of treatment varies with
    the features (measured: AUC ~0.51 predicting treatment). The naive
    difference in means is then confounded, and CUPED is invalid (it
    assumes balanced features). The average effect is estimated with
    AIPW, using propensity / outcome models fit on the train fold only.
    The X-learner and causal forest learn e(x) for the same reason.
  - The planned treated share is not published precisely; SRM is run
    against 0.85 and reported, not used as a hard gate.

USAGE
    python -m uplift.data download && python -m uplift.data parquet
    python -m uplift.run_criteo
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .data import FEATURES, PARQUET, arm_summary, feature_balance, load_folds
from .evaluate import qini_at_k_intervals, qini_coefficient, qini_curve, uplift_at_k
from .experiment import aipw, diff_in_means, diff_in_proportions, srm_check
from .learners import CausalForest, RiskModel, SLearner, TLearner, XLearner

OUTCOME = "visit"
TRAIN_FOLDS = [0]
EVAL_FOLDS = [5, 6, 7, 8, 9]
BUDGETS = [0.1, 0.2, 0.3]
N_BOOT = 200          # Poisson-bootstrap resamples of the evaluation rows
RESULTS = Path("results")


def _log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def run(seed: int = 7, skip_forest: bool = False) -> dict:
    t0 = time.time()

    # --- 1. experiment checks over ALL rows, in SQL ---
    arms = arm_summary().set_index("treatment")
    n_t, n_c = int(arms.loc[1, "n"]), int(arms.loc[0, "n"])
    report = {
        "n": n_t + n_c,
        "srm_vs_0.85": srm_check(counts=(n_t, n_c), expected_treat_frac=0.85),
        "imbalanced_features": feature_balance().query("imbalanced")["feature"].tolist(),
        "ate_visit": diff_in_proportions(int(arms.loc[1, "visits"]), n_t, int(arms.loc[0, "visits"]), n_c),
        "ate_conversion": diff_in_proportions(int(arms.loc[1, "conversions"]), n_t,
                                              int(arms.loc[0, "conversions"]), n_c),
    }
    _log("experiment checks done", t0)

    # --- 2. train on one fold ---
    tr = load_folds(TRAIN_FOLDS)
    _log(f"train rows {len(tr):,}", t0)
    t_learner = TLearner(seed)
    models = [SLearner(seed), t_learner, XLearner(seed, fit_propensity=True), RiskModel(seed)]
    if not skip_forest:
        models.append(CausalForest(seed, fit_propensity=True))
    for m in models:
        m.fit(tr[FEATURES], tr["treatment"], tr[OUTCOME])
        _log(f"fit {m.name}", t0)

    # pooled tests -> treatment chance depends on features; learn it
    propensity = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    propensity.fit(tr[FEATURES], tr["treatment"])
    del tr

    # --- 3. judge on folds the models never saw ---
    ev = load_folds(EVAL_FOLDS)
    y, t = ev[OUTCOME].to_numpy(), ev["treatment"].to_numpy()
    _log(f"eval rows {len(ev):,}", t0)

    e = propensity.predict_proba(ev[FEATURES])[:, 1]
    report["eval_n"] = len(ev)
    report["treatment_predictability_auc"] = float(roc_auc_score(t, e))
    report["propensity_pct_1_50_99"] = np.percentile(e, [1, 50, 99]).tolist()
    report["ate_visit_naive_eval"] = diff_in_means(y, t)
    report["ate_visit_aipw_eval"] = aipw(y, t, t_learner.m1.predict_proba(ev[FEATURES].to_numpy())[:, 1],
                                         t_learner.m0.predict_proba(ev[FEATURES].to_numpy())[:, 1], e)

    scores = {"random": np.random.default_rng([seed, 2]).uniform(size=len(ev))}
    for m in models:
        scores[m.name] = m.predict_uplift(ev[FEATURES])
    _log("scored eval folds", t0)
    intervals = qini_at_k_intervals(y, t, scores, BUDGETS, reference="random", n_boot=N_BOOT, seed=seed)
    _log(f"bootstrapped {N_BOOT} resamples for policy intervals", t0)

    rows, curves = [], {}
    for name, s in scores.items():
        row = {"model": name, "qini": qini_coefficient(y, t, s)}
        frac, q = qini_curve(y, t, s)
        for k in BUDGETS:
            row[f"uplift@{int(k * 100)}%"] = uplift_at_k(y, t, s, k)
            row[f"extra_visits@{int(k * 100)}%"] = float(q[int(round(k * len(y)))])
            band = intervals[name][k]
            row[f"extra_visits@{int(k * 100)}%_ci"] = list(band["extra"])
            row[f"vs_random@{int(k * 100)}%_ci"] = list(band["vs_reference"])
            row[f"x_random@{int(k * 100)}%_ci"] = list(band["ratio"])
        rows.append(row)
        idx = np.linspace(0, len(frac) - 1, 101).astype(int)  # 101 points is enough to plot
        curves[name] = {"frac": frac[idx].tolist(), "qini": q[idx].tolist()}
    report["models"] = rows
    report["qini_curves"] = curves
    _log("done", t0)
    return report


def _print(r: dict):
    srm, ate, conv = r["srm_vs_0.85"], r["ate_visit"], r["ate_conversion"]
    naive, dr = r["ate_visit_naive_eval"], r["ate_visit_aipw_eval"]
    lo, mid, hi = r["propensity_pct_1_50_99"]
    print(f"\nUsers: {r['n']:,}   treated share {srm['observed_treat_frac']:.4f}   "
          f"SRM vs 0.85: p = {srm['p_value']:.3g}")
    print(f"Imbalanced features (|SMD|>0.1): {r['imbalanced_features'] or 'none'}")
    print(f"Treatment predictable from features: AUC {r['treatment_predictability_auc']:.4f} "
          f"(0.5 = perfectly random)  P(treated) 1st/50th/99th pct {lo:.3f}/{mid:.3f}/{hi:.3f}")
    print(f"\nVisit rate      control {ate['control_mean']:.4%}  treated {ate['treatment_mean']:.4%}")
    print(f"  naive, all    {ate['effect']:+.4%}  95% CI [{ate['ci_low']:+.4%}, {ate['ci_high']:+.4%}]")
    print(f"  naive, eval   {naive['effect']:+.4%}  95% CI [{naive['ci_low']:+.4%}, {naive['ci_high']:+.4%}]")
    print(f"  AIPW, eval    {dr['effect']:+.4%}  95% CI [{dr['ci_low']:+.4%}, {dr['ci_high']:+.4%}]"
          f"   <- adjusts for treatment chance varying by user")
    print(f"Conversion      naive {conv['effect']:+.4%}  95% CI [{conv['ci_low']:+.4%}, {conv['ci_high']:+.4%}]")
    print("NOTE: Criteo sub-sampled the data so the true incrementality cannot be recovered; "
          "absolute effects are distorted, rankings are what matter.")

    cols = ["qini"] + [f"uplift@{int(k * 100)}%" for k in BUDGETS] + [f"extra_visits@{int(k * 100)}%" for k in BUDGETS]
    print(f"\n{'model':<14}" + "".join(f"{c:>18}" for c in cols))
    for row in r["models"]:
        cells = [f"{row['qini']:>18.5f}"]
        cells += [f"{row[c]:>18.4%}" for c in cols if c.startswith("uplift@")]
        cells += [f"{row[c]:>18,.0f}" for c in cols if c.startswith("extra_")]
        print(f"{row['model']:<14}" + "".join(cells))

    print(f"\n95% intervals ({N_BOOT}-resample Poisson bootstrap of the evaluation rows; models held fixed):")
    print(f"{'model':<14}" + "".join(f"{'extra visits @' + str(int(k * 100)) + '%':>26}" for k in BUDGETS)
          + f"{'x random @10%':>18}")
    for row in r["models"]:
        cells = "".join(f"{row[f'extra_visits@{int(k * 100)}%_ci'][0]:>12,.0f} - "
                        f"{row[f'extra_visits@{int(k * 100)}%_ci'][1]:<11,.0f}" for k in BUDGETS)
        lo, hi = row["x_random@10%_ci"]
        print(f"{row['model']:<14}{cells}{lo:>9.1f} - {hi:<6.1f}")
    print(f"\nuplift@k = treated minus control visit rate among the top k%. "
          f"extra_visits@k = estimated incremental visits from targeting them (of {r['eval_n']:,} eval users).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-forest", action="store_true", help="skip the slow causal forest")
    a = ap.parse_args()
    if not PARQUET.exists():
        raise SystemExit(f"{PARQUET} missing - run `python -m uplift.data download` then `parquet`")

    report = run(a.seed, a.skip_forest)
    _print(report)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "criteo_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS / 'criteo_report.json'}")


if __name__ == "__main__":
    main()
