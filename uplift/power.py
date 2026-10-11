"""
power.py

WHAT : How many customers an experiment needs, and the smallest effect a
       given experiment can detect.
WHY  : An underpowered test cannot tell "no effect" from "too few people".
       The churn repo's coupon hold-out (11 treated vs 4 control) came back
       p = 1.00 - inconclusive by construction. This answers how big it
       needed to be.
FLOW : sample_size() before the test; mde() / achieved_power() to judge a
       test that already ran.
LOGIC: two-proportion z-test, normal approximation. With control rate pc,
       treatment rate pt and n_t = k * n_c:
         n_c = [z_a * sqrt(pbar*qbar*(1 + 1/k)) + z_b * sqrt(pc*qc + pt*qt/k)]^2 / d^2
       where pbar = (pc + k*pt) / (1 + k) is the pooled rate under H0.

USAGE
    python -m uplift.power --p-control 0.08 --mde 0.05
    python -m uplift.power --p-control 0.08 --n-treatment 11 --n-control 4
"""
import argparse
import math

from scipy import stats
from scipy.optimize import brentq


def sample_size(p_control: float, mde: float, alpha: float = 0.05,
                power: float = 0.8, ratio: float = 1.0) -> dict:
    """Customers needed per arm to detect an absolute lift of `mde`.
    ratio = n_treatment / n_control."""
    pc, pt, k = p_control, p_control + mde, ratio
    if not 0 < pc < 1 or not 0 < pt < 1:
        raise ValueError("control and treatment rates must be inside (0, 1)")
    pbar = (pc + k * pt) / (1 + k)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * math.sqrt(pbar * (1 - pbar) * (1 + 1 / k))
           + z_b * math.sqrt(pc * (1 - pc) + pt * (1 - pt) / k)) ** 2
    n_c = math.ceil(num / mde ** 2)
    n_t = math.ceil(n_c * k)
    return {"n_control": n_c, "n_treatment": n_t, "n_total": n_c + n_t}


def achieved_power(p_control: float, p_treatment: float, n_control: int,
                   n_treatment: int, alpha: float = 0.05) -> float:
    """Chance this design detects the given true difference (two-sided)."""
    pc, pt = p_control, p_treatment
    pbar = (n_control * pc + n_treatment * pt) / (n_control + n_treatment)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n_control + 1 / n_treatment))
    se1 = math.sqrt(pc * (1 - pc) / n_control + pt * (1 - pt) / n_treatment)
    z_a = stats.norm.ppf(1 - alpha / 2)
    d = abs(pt - pc)
    return float(stats.norm.cdf((d - z_a * se0) / se1) + stats.norm.cdf((-d - z_a * se0) / se1))


def mde(p_control: float, n_control: int, n_treatment: int,
        alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift this design detects with the given power."""
    f = lambda d: achieved_power(p_control, p_control + d, n_control, n_treatment, alpha) - power
    hi = 1 - p_control - 1e-9
    if f(hi) < 0:
        return float("nan")  # even the largest possible lift is not detectable
    return float(brentq(f, 1e-9, hi))


def main():
    ap = argparse.ArgumentParser(description="Two-proportion power analysis")
    ap.add_argument("--p-control", type=float, required=True)
    ap.add_argument("--mde", type=float, help="absolute lift to detect -> sample size")
    ap.add_argument("--n-control", type=int, help="existing design -> its MDE")
    ap.add_argument("--n-treatment", type=int)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    a = ap.parse_args()

    if a.mde is not None:
        print(sample_size(a.p_control, a.mde, a.alpha, a.power))
    if a.n_control and a.n_treatment:
        d = mde(a.p_control, a.n_control, a.n_treatment, a.alpha, a.power)
        print(f"MDE at {a.power:.0%} power: {d:+.3f} absolute"
              if d == d else "no achievable lift is detectable at this size")


if __name__ == "__main__":
    main()
