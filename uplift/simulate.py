"""
simulate.py

WHAT : A randomized campaign where the TRUE effect for every customer is
       known, because we plant it.
WHY  : On real data nobody knows a customer's true uplift, so an uplift
       model can only be scored indirectly (Qini). Here we can check the
       models against the answer itself before trusting them on Criteo.
FLOW : draw customer features -> baseline return chance p0(x)
       -> planted uplift tau(x) -> random 50/50 treatment -> outcome.
LOGIC: mirrors churn/uplift.py's winnability(): customers who left over
       FIXABLE pain (support, delivery) are persuadable; picky customers
       are lost causes; loyal customers are sure things who return anyway.
       p1 = p0 + tau * (1 - p0) keeps probabilities valid and makes sure
       things (high p0) naturally low-uplift.
       HIGH RISK != HIGH UPLIFT: pickiness lowers p0 (raises risk) AND
       lowers tau, so targeting by risk wastes budget on lost causes.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "support_pain", "delivery_pain", "pickiness",
    "tenure", "orders_90d", "noise_a", "noise_b",
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_campaign(n: int = 200_000, treat_frac: float = 0.5, seed: int = 7) -> pd.DataFrame:
    """One randomized campaign. Columns: FEATURES + treatment, outcome,
    p0 (return chance without offer), true_uplift (planted effect)."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "support_pain": rng.uniform(0, 1, n),
        "delivery_pain": rng.uniform(0, 1, n),
        "pickiness": rng.uniform(0, 1, n),
        "tenure": rng.exponential(1.0, n),          # years as a customer
        "orders_90d": rng.poisson(3.0, n).astype(float),
        "noise_a": rng.normal(0, 1, n),             # irrelevant on purpose
        "noise_b": rng.normal(0, 1, n),
    })

    # baseline: loyal, active customers come back on their own
    p0 = _sigmoid(-1.2 + 0.9 * df["tenure"] + 0.25 * df["orders_90d"]
                  - 1.8 * df["pickiness"] - 0.6 * df["support_pain"])

    # planted effect: fixable pain is winnable, pickiness is not
    tau = np.clip(0.35 * df["support_pain"] + 0.25 * df["delivery_pain"]
                  - 0.30 * df["pickiness"], 0.0, 0.6)
    p1 = p0 + tau * (1 - p0)

    t = (rng.uniform(0, 1, n) < treat_frac).astype(int)
    y = (rng.uniform(0, 1, n) < np.where(t == 1, p1, p0)).astype(int)

    df["treatment"] = t
    df["outcome"] = y
    df["p0"] = p0
    df["true_uplift"] = p1 - p0
    return df
