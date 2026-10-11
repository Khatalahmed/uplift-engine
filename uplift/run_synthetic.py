"""
run_synthetic.py

WHAT : The full pipeline on a campaign where the true effect is known:
       experiment checks -> CUPED -> uplift models -> budget decision.
WHY  : Proves each step recovers the planted answer BEFORE the same code
       runs on Criteo, where there is no answer key.
FLOW : make_campaign -> srm/balance/effect/cuped on all customers
       -> fit learners on train half -> score test half
       -> Qini, PEHE, and true incremental outcomes at each budget
       -> results/synthetic_report.json

USAGE
    python -m uplift.run_synthetic
    python -m uplift.run_synthetic --n 50000
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import budget_gain, pehe, qini_coefficient
from .experiment import balance, cuped, cupac_covariate, diff_in_means, srm_check
from .learners import ALL_LEARNERS, CausalForest, RiskModel
from .simulate import FEATURES, make_campaign

BUDGETS = [0.1, 0.2, 0.3]
RESULTS = Path("results")


def train_mask(n: int, seed: int) -> np.ndarray:
    """Random half for training. Uses its own stream: reusing `seed` would
    replay make_campaign's first draw (support_pain) and turn the split into
    support_pain < 0.5 - training and testing on different customers."""
    return np.random.default_rng([seed, 1]).uniform(size=n) < 0.5


def run(n: int = 200_000, seed: int = 7) -> dict:
    df = make_campaign(n=n, seed=seed)
    X, t, y = df[FEATURES], df["treatment"], df["outcome"]

    # --- 1. is the experiment trustworthy, and what is the average effect? ---
    report = {
        "n": n,
        "srm": srm_check(t),
        "imbalanced_features": balance(df, FEATURES).query("imbalanced")["feature"].tolist(),
        "true_ate": float(df["true_uplift"].mean()),
        "ate": diff_in_means(y, t),
        "ate_cuped": cuped(y, t, cupac_covariate(X, y, seed=seed)),
    }

    # --- 2. who does it work on? fit on one half, judge on the other ---
    rng = np.random.default_rng([seed, 2])
    train = train_mask(n, seed)
    tr, te = df[train], df[~train]

    models = [cls(seed) for cls in ALL_LEARNERS] + [CausalForest(seed), RiskModel(seed)]
    scores = {"random": rng.uniform(size=len(te))}
    for m in models:
        m.fit(tr[FEATURES], tr["treatment"], tr["outcome"])
        scores[m.name] = m.predict_uplift(te[FEATURES])
    scores["oracle"] = te["true_uplift"].to_numpy()  # ceiling: perfect knowledge

    rows = []
    for name, s in scores.items():
        row = {"model": name,
               "qini": qini_coefficient(te["outcome"], te["treatment"], s)}
        if name not in ("random", "risk", "oracle"):
            row["pehe"] = pehe(te["true_uplift"], s)
            row["corr_with_truth"] = float(np.corrcoef(te["true_uplift"], s)[0, 1])
        for k in BUDGETS:
            row[f"saved@{int(k * 100)}%"] = budget_gain(te["true_uplift"], s, k)
        rows.append(row)
    report["models"] = rows
    return report


def _print(r: dict):
    srm, ate, adj = r["srm"], r["ate"], r["ate_cuped"]
    print(f"\nCustomers: {r['n']:,}   treated share {srm['observed_treat_frac']:.3f}   "
          f"SRM p = {srm['p_value']:.3f} ({'BROKEN' if srm['srm_detected'] else 'ok'})")
    print(f"Imbalanced features: {r['imbalanced_features'] or 'none'}")
    print(f"\nAverage effect   true {r['true_ate']:+.4f}")
    print(f"  plain          {ate['effect']:+.4f}  95% CI [{ate['ci_low']:+.4f}, {ate['ci_high']:+.4f}]")
    print(f"  CUPED          {adj['effect']:+.4f}  95% CI [{adj['ci_low']:+.4f}, {adj['ci_high']:+.4f}]"
          f"   variance -{adj['variance_reduction']:.0%}")

    metric_cols = ["qini", "pehe", "corr_with_truth"]
    saved_cols = [f"saved@{int(k * 100)}%" for k in BUDGETS]
    print(f"\n{'model':<14}" + "".join(f"{c:>16}" for c in metric_cols + saved_cols))
    for row in r["models"]:
        cells = [f"{row[c]:>16.4f}" if c in row else f"{'-':>16}" for c in metric_cols]
        cells += [f"{row[c]:>16.0f}" for c in saved_cols]
        print(f"{row['model']:<14}" + "".join(cells))
    print("\nsaved@k = TRUE extra returning customers when the top k% by that score get the offer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    report = run(a.n, a.seed)
    _print(report)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "synthetic_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS / 'synthetic_report.json'}")


if __name__ == "__main__":
    main()
