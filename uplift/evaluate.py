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


def qini_at_k_intervals(y, t, scores: dict, ks, reference: str = "random",
                        n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap intervals for the Qini value (estimated incremental outcomes)
    of targeting the top k by each score, and for each score's gain over
    `reference` - paired, so both policies see the same resampled users.

    Poisson bootstrap: every resample re-weights the rows (weight ~ Poisson(1))
    instead of copying them, so each score is sorted once and a resample costs
    a few weighted sums, which is what makes 7M rows affordable.

    The intervals cover the sampling noise of the EVALUATION rows only: the
    models are held fixed, so they say nothing about how a refit would vary.
    Returns {score: {k: {"extra": (lo, hi), "vs_reference": (lo, hi),
    "ratio": (lo, hi)}}}.
    """
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(int)
    n = len(y)
    cuts = sorted({max(1, int(round(k * n))) for k in ks})
    tops = {}
    for name, score in scores.items():
        order = np.argsort(-np.asarray(score), kind="stable")[:cuts[-1]]
        tops[name] = (order, y[order], t[order])

    def qini_at_cuts(w, order, ys, ts):
        ws = w[order]
        treated = ws * ts
        control = ws - treated
        # sums over [0, cut) for every cut, from one pass over the prefix
        sums = np.cumsum(np.add.reduceat(
            np.vstack([treated, control, treated * ys, control * ys]),
            [0] + cuts[:-1], axis=1), axis=1)
        n_t, n_c, y_t, y_c = sums
        return y_t - y_c * n_t / n_c

    rng = np.random.default_rng(seed)
    draws = {name: [] for name in scores}
    for _ in range(n_boot):
        w = rng.poisson(1.0, n).astype(float)
        for name, top in tops.items():
            draws[name].append(qini_at_cuts(w, *top))
    draws = {name: np.array(d) for name, d in draws.items()}       # (n_boot, len(cuts))

    def band(a):
        return tuple(float(v) for v in np.percentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

    ref = draws[reference]
    out = {}
    for name, d in draws.items():
        out[name] = {}
        for k in ks:
            j = cuts.index(max(1, int(round(k * n))))
            out[name][k] = {"extra": band(d[:, j]),
                            "vs_reference": band(d[:, j] - ref[:, j]),
                            "ratio": band(d[:, j] / ref[:, j])}
    return out


def pehe(true_uplift, predicted) -> float:
    """Root mean squared error vs the planted effect (synthetic only)."""
    return float(np.sqrt(np.mean((np.asarray(true_uplift) - np.asarray(predicted)) ** 2)))


def budget_gain(true_uplift, score, k: float) -> float:
    """TRUE incremental outcomes from treating the top k fraction (synthetic only)."""
    order = np.argsort(-np.asarray(score), kind="stable")
    n = max(1, int(round(k * len(order))))
    return float(np.asarray(true_uplift)[order[:n]].sum())
