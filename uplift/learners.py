"""
learners.py

WHAT : Models that estimate each customer's uplift
       tau(x) = P(y=1 | x, treated) - P(y=1 | x, control),
       plus the "risk" baseline a team would use first.
WHY  : The average effect says whether the campaign works. Uplift says
       WHO it works on - which is what a budget decision needs.
FLOW : fit(X, t, y) -> predict_uplift(X)
LOGIC:
  - S-learner: one model with treatment as a feature. Simple, but trees
    can ignore the treatment column and shrink uplift toward zero.
  - T-learner: separate models for treated and control; uplift is the
    difference. Noisy when one arm is small (Criteo is ~85/15).
  - X-learner: fit T-learner models, impute each person's individual
    effect from the OTHER arm's model, then model those effects and blend
    by the propensity. Built for unbalanced arms.
  - RiskModel: P(y=1 | x, control) only. Targeting the LOWEST return
    chance = "target the riskiest" - the baseline uplift must beat.
MECHANISM: sklearn HistGradientBoosting (fast on millions of rows, no
       extra dependency). Treatment is randomized, so the propensity is
       the constant treated share.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _clf(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          min_samples_leaf=100, random_state=seed)


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         min_samples_leaf=100, random_state=seed)


def _arrays(X, t, y=None):
    X = np.asarray(X, dtype=float)
    t = np.asarray(t).astype(int)
    return (X, t) if y is None else (X, t, np.asarray(y).astype(int))


class SLearner:
    name = "s_learner"

    def __init__(self, seed: int = 7):
        self.model = _clf(seed)

    def fit(self, X, t, y):
        X, t, y = _arrays(X, t, y)
        self.model.fit(np.column_stack([X, t]), y)
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        p1 = self.model.predict_proba(np.column_stack([X, ones]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([X, zeros]))[:, 1]
        return p1 - p0


class TLearner:
    name = "t_learner"

    def __init__(self, seed: int = 7):
        self.m1, self.m0 = _clf(seed), _clf(seed)

    def fit(self, X, t, y):
        X, t, y = _arrays(X, t, y)
        self.m1.fit(X[t == 1], y[t == 1])
        self.m0.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    name = "x_learner"

    def __init__(self, seed: int = 7, fit_propensity: bool = False):
        self.t = TLearner(seed)
        self.tau1, self.tau0 = _reg(seed), _reg(seed)
        self.e = 0.5
        # False: one randomized test, propensity = treated share.
        # True: pooled tests (Criteo), propensity e(x) learned from features.
        self.propensity = _clf(seed) if fit_propensity else None

    def fit(self, X, t, y):
        X, t, y = _arrays(X, t, y)
        self.t.fit(X, t, y)
        X1, y1, X0, y0 = X[t == 1], y[t == 1], X[t == 0], y[t == 0]
        # imputed individual effects, using the OTHER arm's model as counterfactual
        d1 = y1 - self.t.m0.predict_proba(X1)[:, 1]
        d0 = self.t.m1.predict_proba(X0)[:, 1] - y0
        self.tau1.fit(X1, d1)
        self.tau0.fit(X0, d0)
        self.e = t.mean()
        if self.propensity is not None:
            self.propensity.fit(X, t)
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        e = self.e if self.propensity is None else self.propensity.predict_proba(X)[:, 1]
        # weight toward the estimate built from the LARGER arm's counterfactual model
        return e * self.tau0.predict(X) + (1 - e) * self.tau1.predict(X)


class CausalForest:
    """EconML CausalForestDML: residualize outcome and treatment, then grow
    trees that split on where the EFFECT differs (not where the outcome
    does). Slow, so it trains on at most `max_train` rows."""
    name = "causal_forest"

    def __init__(self, seed: int = 7, max_train: int = 300_000, n_estimators: int = 200,
                 fit_propensity: bool = False):
        self.seed, self.max_train, self.n_estimators = seed, max_train, n_estimators
        self.fit_propensity = fit_propensity

    def fit(self, X, t, y):
        from econml.dml import CausalForestDML  # heavy import, only when used
        from sklearn.dummy import DummyClassifier

        X, t, y = _arrays(X, t, y)
        if len(X) > self.max_train:
            keep = np.random.default_rng([self.seed, 3]).choice(len(X), self.max_train, replace=False)
            X, t, y = X[keep], t[keep], y[keep]
        self.model = CausalForestDML(
            model_y=_reg(self.seed),
            # one randomized test: propensity = treated share; pooled tests: learn e(x)
            model_t=_clf(self.seed) if self.fit_propensity else DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=200,
            random_state=self.seed,
        )
        self.model.fit(y.astype(float), t, X=X)
        return self

    def predict_uplift(self, X):
        return self.model.effect(np.asarray(X, dtype=float)).ravel()


class RiskModel:
    """Baseline: rank by churn risk = 1 - P(return | x, no offer)."""
    name = "risk"

    def __init__(self, seed: int = 7):
        self.model = _clf(seed)

    def fit(self, X, t, y):
        X, t, y = _arrays(X, t, y)
        self.model.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        # not an uplift - a targeting score on the same scale of "higher = target first"
        return 1 - self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]


ALL_LEARNERS = [SLearner, TLearner, XLearner]
