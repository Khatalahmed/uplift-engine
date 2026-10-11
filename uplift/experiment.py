"""
experiment.py

WHAT : The checks an A/B readout needs BEFORE anyone looks at the uplift:
       is the split what we asked for, are the groups comparable, and how
       big is the average effect (with an honest interval).
WHY  : A broken split (sample-ratio mismatch) or imbalanced groups makes
       every downstream number wrong, however good the model is.
FLOW : srm_check -> balance -> diff_in_means -> (optional) cuped
LOGIC:
  - SRM: chi-square of observed arm sizes vs the planned ratio. A tiny
    p-value means assignment or logging is broken: stop, don't analyse.
  - Balance: standardized mean difference per pre-treatment feature;
    |SMD| > 0.1 is the usual red flag.
  - Effect: difference in means with Welch standard errors (works for
    binary outcomes too - it is then the difference in proportions).
  - CUPED: subtract theta * (x - mean x), where x is a PRE-treatment
    covariate. Same expected effect, smaller variance -> tighter interval.
    With no pre-period metric (Criteo), x = a cross-fitted prediction of
    the outcome from pre-treatment features (CUPAC). The model never sees
    the treatment, so the adjustment cannot bias the effect - PROVIDED the
    features are balanced across arms. Criteo's are not (several pooled
    tests with different splits), so there use aipw() instead.
  - AIPW: doubly robust effect when the chance of treatment varies with
    the features.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict


def srm_check(t=None, expected_treat_frac: float = 0.5, counts: tuple[int, int] | None = None) -> dict:
    """Sample-ratio mismatch: do arm sizes match the planned split?
    Pass the treatment column, or counts=(n_treatment, n_control)."""
    if counts is not None:
        n_t, n_c = counts
    else:
        t = np.asarray(t)
        n_t, n_c = int(t.sum()), int(len(t) - t.sum())
    n = n_t + n_c
    chi2, p = stats.chisquare([n_t, n_c], f_exp=[n * expected_treat_frac, n * (1 - expected_treat_frac)])
    return {"n_treatment": n_t, "n_control": n_c, "observed_treat_frac": n_t / n,
            "chi2": float(chi2), "p_value": float(p), "srm_detected": bool(p < 0.001)}


def balance(df: pd.DataFrame, features: list[str], treatment_col: str = "treatment") -> pd.DataFrame:
    """Standardized mean difference per feature (treated vs control)."""
    t = df[treatment_col] == 1
    rows = []
    for f in features:
        a, b = df.loc[t, f], df.loc[~t, f]
        pooled_sd = np.sqrt((a.var() + b.var()) / 2)
        smd = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"feature": f, "smd": float(smd), "imbalanced": bool(abs(smd) > 0.1)})
    return pd.DataFrame(rows)


def diff_in_means(y, t, alpha: float = 0.05) -> dict:
    """Average treatment effect = mean(y | t=1) - mean(y | t=0), Welch SE."""
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    effect = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(effect) / se)) if se > 0 else 0.0
    return {"control_mean": float(y0.mean()), "treatment_mean": float(y1.mean()),
            "effect": float(effect), "se": float(se),
            "ci_low": float(effect - z * se), "ci_high": float(effect + z * se),
            "p_value": float(p), "relative_lift": float(effect / y0.mean()) if y0.mean() else None}


def diff_in_proportions(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict:
    """Same as diff_in_means for a binary outcome, from counts only - so a
    14M-row trial can be summarized by SQL without loading rows."""
    p1, p0 = x_t / n_t, x_c / n_c
    effect = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n_t + p0 * (1 - p0) / n_c)
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(effect) / se)) if se > 0 else 0.0
    return {"control_mean": float(p0), "treatment_mean": float(p1),
            "effect": float(effect), "se": float(se),
            "ci_low": float(effect - z * se), "ci_high": float(effect + z * se),
            "p_value": float(p), "relative_lift": float(effect / p0) if p0 else None}


def cuped(y, t, x, alpha: float = 0.05) -> dict:
    """Variance-reduced effect using pre-treatment covariate x."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    raw = diff_in_means(y, t, alpha)
    adj = diff_in_means(y_adj, t, alpha)
    adj["theta"] = float(theta)
    adj["variance_reduction"] = float(1 - (adj["se"] / raw["se"]) ** 2)
    return adj


def aipw(y, t, mu1, mu0, e, alpha: float = 0.05) -> dict:
    """Doubly robust (AIPW) average effect, for when the chance of treatment
    varies with the features - e.g. several tests with different splits
    pooled together (Criteo). Correct if EITHER the outcome models (mu1, mu0)
    OR the propensity e(x) is right. All three must come from models fit on
    OTHER rows (cross-fitting), never the rows being scored."""
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    e = np.clip(np.asarray(e, dtype=float), 0.01, 0.99)
    mu1, mu0 = np.asarray(mu1, dtype=float), np.asarray(mu0, dtype=float)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    effect, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"effect": float(effect), "se": float(se),
            "ci_low": float(effect - z * se), "ci_high": float(effect + z * se),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(effect) / se)))}


def cupac_covariate(X, y, folds: int = 2, seed: int = 7) -> np.ndarray:
    """Cross-fitted P(outcome | pre-treatment features) to use as CUPED's x.
    The model is never given the treatment column."""
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
