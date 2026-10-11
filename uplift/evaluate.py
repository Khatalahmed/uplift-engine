"""
evaluate.py

WHAT : Scores an uplift model - with the true answer when we have it
       (synthetic), and without it when we don't (real trials).
WHY  : On real data nobody observes a person both treated and untreated,
       so accuracy per customer is impossible. The Qini curve measures what
       matters instead: if we target the top k by this score, how many
       EXTRA outcomes do we get versus not treating them?
FLOW : qini_curve -> qini_coefficient; uplift_at_k for a single budget;
       pehe / budget_gain only when true uplift is known.
LOGIC:
  - Qini at prefix k: Y_t(k) - Y_c(k) * N_t(k) / N_c(k)
    = treated outcomes minus control outcomes rescaled to the treated
    count -> estimated incremental outcomes among the top k.
  - Qini coefficient: area between the model's curve and the straight
    random-targeting line. 0 = no better than random.
  - PEHE: root mean squared error against the planted uplift.
"""
import numpy as np


def _sorted(y, t, score):
    order = np.argsort(-np.asarray(score), kind="stable")
    return np.asarray(y)[order].astype(float), np.asarray(t)[order].astype(int)


def qini_curve(y, t, score) -> tuple[np.ndarray, np.ndarray]:
    """(fraction targeted, estimated incremental outcomes) - starts at (0, 0)."""
    y, t = _sorted(y, t, score)
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    frac = np.arange(1, len(y) + 1) / len(y)
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], q])


def qini_coefficient(y, t, score) -> float:
    """Area between the Qini curve and random targeting, per customer."""
    frac, q = qini_curve(y, t, score)
    random_line = frac * q[-1]
    return float(np.trapezoid(q - random_line, frac) / len(y))


def uplift_at_k(y, t, score, k: float) -> float:
    """Observed treated rate minus control rate among the top k fraction."""
    y, t = _sorted(y, t, score)
    top = slice(0, max(1, int(round(k * len(y)))))
    yt, tt = y[top], t[top]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return float("nan")
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def pehe(true_uplift, predicted) -> float:
    """Root mean squared error vs the planted effect (synthetic only)."""
    return float(np.sqrt(np.mean((np.asarray(true_uplift) - np.asarray(predicted)) ** 2)))


def budget_gain(true_uplift, score, k: float) -> float:
    """TRUE incremental outcomes from treating the top k fraction (synthetic only)."""
    order = np.argsort(-np.asarray(score), kind="stable")
    n = max(1, int(round(k * len(order))))
    return float(np.asarray(true_uplift)[order[:n]].sum())
