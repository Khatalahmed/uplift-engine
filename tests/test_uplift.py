"""Each test checks a method against an answer known in advance."""
import numpy as np
import pytest

from uplift.evaluate import budget_gain, qini_coefficient, uplift_at_k
from uplift.experiment import aipw, balance, cuped, diff_in_means, diff_in_proportions, srm_check
from uplift.learners import RiskModel, TLearner, XLearner
from uplift.power import achieved_power, mde, sample_size
from uplift.simulate import FEATURES, make_campaign


@pytest.fixture(scope="module")
def campaign():
    return make_campaign(n=60_000, seed=11)


# --- power -------------------------------------------------------------------

def test_sample_size_matches_textbook():
    # 10% -> 12%, alpha 0.05, power 0.8: the standard answer is 3,841 per arm
    assert sample_size(0.10, 0.02)["n_control"] == 3841


def test_sample_size_gives_the_power_it_promises():
    n = sample_size(0.08, 0.03, ratio=2.0)
    assert achieved_power(0.08, 0.11, n["n_control"], n["n_treatment"]) == pytest.approx(0.8, abs=0.01)


def test_tiny_test_cannot_detect_realistic_lifts():
    # the churn repo's hold-out: 11 treated vs 4 control
    assert mde(0.08, n_control=4, n_treatment=11) > 0.4 or np.isnan(mde(0.08, 4, 11))


# --- experiment checks -------------------------------------------------------

def test_srm_passes_fair_split_and_flags_broken_one(campaign):
    assert not srm_check(campaign["treatment"])["srm_detected"]
    broken = np.r_[np.ones(5_300), np.zeros(4_700)]
    assert srm_check(broken)["srm_detected"]


def test_randomized_groups_are_balanced(campaign):
    assert not balance(campaign, FEATURES)["imbalanced"].any()


def test_effect_interval_covers_planted_effect(campaign):
    r = diff_in_means(campaign["outcome"], campaign["treatment"])
    assert r["ci_low"] < campaign["true_uplift"].mean() < r["ci_high"]


def test_counts_only_readout_matches_row_level(campaign):
    # the Criteo readout uses SQL counts; it must agree with the row-level maths
    y, t = campaign["outcome"], campaign["treatment"]
    rows = diff_in_means(y, t)
    counts = diff_in_proportions(int(y[t == 1].sum()), int((t == 1).sum()),
                                 int(y[t == 0].sum()), int((t == 0).sum()))
    assert counts["effect"] == pytest.approx(rows["effect"])
    assert counts["se"] == pytest.approx(rows["se"], rel=1e-3)
    assert srm_check(counts=(int((t == 1).sum()), int((t == 0).sum())))["p_value"] == \
        pytest.approx(srm_check(t)["p_value"])


def test_aipw_fixes_pooled_tests_where_naive_fails():
    # two tests pooled: loyal customers were treated 90%, others 50%.
    # naive difference in means is confounded; AIPW with the true nuisances is not.
    df = make_campaign(n=200_000, seed=5)
    loyal = (df["tenure"] > 1).to_numpy()
    e = np.where(loyal, 0.9, 0.5)
    rng = np.random.default_rng([5, 9])  # own stream: seed 5 alone replays the simulator
    t = (rng.uniform(size=len(df)) < e).astype(int)
    p1 = df["p0"] + df["true_uplift"]
    y = (rng.uniform(size=len(df)) < np.where(t == 1, p1, df["p0"])).astype(int)
    truth = df["true_uplift"].mean()

    naive = diff_in_means(y, t)
    dr = aipw(y, t, p1, df["p0"], e)
    assert not naive["ci_low"] < truth < naive["ci_high"]
    assert dr["ci_low"] < truth < dr["ci_high"]


def test_cuped_shrinks_variance_without_moving_estimate(campaign):
    y, t = campaign["outcome"], campaign["treatment"]
    raw = diff_in_means(y, t)
    adj = cuped(y, t, campaign["p0"])  # p0 is pre-treatment: a perfect CUPED covariate
    assert adj["variance_reduction"] > 0.05
    assert abs(adj["effect"] - raw["effect"]) < 2 * raw["se"]


# --- evaluation --------------------------------------------------------------

def test_qini_rewards_truth_and_not_noise(campaign):
    y, t, truth = campaign["outcome"], campaign["treatment"], campaign["true_uplift"]
    noise = np.random.default_rng(0).uniform(size=len(campaign))
    assert qini_coefficient(y, t, truth) > 0
    assert abs(qini_coefficient(y, t, noise)) < qini_coefficient(y, t, truth)
    assert uplift_at_k(y, t, truth, 0.2) > uplift_at_k(y, t, -truth, 0.2)


# --- pipeline ----------------------------------------------------------------

def test_train_test_split_is_independent_of_customers():
    # regression: the split once reused the simulator's seed and became
    # support_pain < 0.5, so models were tested on customers unlike training
    from uplift.run_synthetic import train_mask
    df = make_campaign(n=100_000, seed=7)
    tr = train_mask(len(df), seed=7)
    for col in ("support_pain", "true_uplift"):
        assert abs(df.loc[tr, col].mean() - df.loc[~tr, col].mean()) < 0.01, col


# --- learners ----------------------------------------------------------------

def test_uplift_models_find_the_planted_effect(campaign):
    tr, te = campaign.iloc[:40_000], campaign.iloc[40_000:]
    for model in (TLearner(), XLearner()):
        model.fit(tr[FEATURES], tr["treatment"], tr["outcome"])
        pred = model.predict_uplift(te[FEATURES])
        assert np.corrcoef(te["true_uplift"], pred)[0, 1] > 0.5, model.name


def test_uplift_targeting_beats_risk_targeting(campaign):
    tr, te = campaign.iloc[:40_000], campaign.iloc[40_000:]
    x = XLearner().fit(tr[FEATURES], tr["treatment"], tr["outcome"])
    risk = RiskModel().fit(tr[FEATURES], tr["treatment"], tr["outcome"])
    truth = te["true_uplift"]
    assert (budget_gain(truth, x.predict_uplift(te[FEATURES]), 0.2)
            > budget_gain(truth, risk.predict_uplift(te[FEATURES]), 0.2))
