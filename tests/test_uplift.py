"""Each test checks a method against an answer known in advance."""
import numpy as np
import pytest

from uplift.evaluate import budget_gain, qini_at_k_intervals, qini_coefficient, qini_curve, uplift_at_k
from uplift.experiment import aipw, balance, cuped, diff_in_means, diff_in_proportions, srm_check
from uplift.learners import CausalForest, RiskModel, SLearner, TLearner, XLearner
from uplift.power import achieved_power, mde, sample_size
from uplift.simulate import FEATURES, make_campaign


@pytest.fixture(scope="module")
def campaign():
    return make_campaign(n=60_000, seed=11)


def pooled_campaign(n: int, seed: int):
    """Two tests pooled, like Criteo: loyal customers were treated 90%,
    others 50%. Returns the campaign with treatment and outcome redrawn, and
    the true propensity e. Tenure also raises the baseline, so the naive
    difference in means is confounded."""
    df = make_campaign(n=n, seed=seed)
    loyal = (df["tenure"] > 1).to_numpy()
    e = np.where(loyal, 0.9, 0.5)
    rng = np.random.default_rng([seed, 9])  # own stream: seed alone replays the simulator
    t = (rng.uniform(size=len(df)) < e).astype(int)
    p1 = df["p0"] + df["true_uplift"]
    df["treatment"] = t
    df["outcome"] = (rng.uniform(size=len(df)) < np.where(t == 1, p1, df["p0"])).astype(int)
    return df, e


@pytest.fixture(scope="module")
def pooled():
    return pooled_campaign(n=200_000, seed=5)


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


def test_aipw_fixes_pooled_tests_where_naive_fails(pooled):
    # naive difference in means is confounded; AIPW with the true nuisances is not.
    df, e = pooled
    y, t = df["outcome"], df["treatment"]
    truth = df["true_uplift"].mean()

    naive = diff_in_means(y, t)
    dr = aipw(y, t, df["p0"] + df["true_uplift"], df["p0"], e)
    assert not naive["ci_low"] < truth < naive["ci_high"]
    assert dr["ci_low"] < truth < dr["ci_high"]


def test_aipw_with_learned_nuisances_fixes_pooled_tests(pooled):
    # what run_criteo actually does: T-learner outcome models and a learned
    # propensity, fit on one half and scored on rows they never saw
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, _ = pooled
    tr, ev = df.iloc[:100_000], df.iloc[100_000:]
    outcome = TLearner().fit(tr[FEATURES], tr["treatment"], tr["outcome"])
    propensity = HistGradientBoostingClassifier(max_iter=200, random_state=7)
    propensity.fit(tr[FEATURES].to_numpy(), tr["treatment"])

    X = ev[FEATURES].to_numpy()
    y, t = ev["outcome"], ev["treatment"]
    truth = ev["true_uplift"].mean()
    naive = diff_in_means(y, t)
    dr = aipw(y, t, outcome.m1.predict_proba(X)[:, 1], outcome.m0.predict_proba(X)[:, 1],
              propensity.predict_proba(X)[:, 1])
    assert not naive["ci_low"] < truth < naive["ci_high"]
    assert dr["ci_low"] < truth < dr["ci_high"]


def test_cuped_shrinks_variance_without_moving_estimate(campaign):
    y, t = campaign["outcome"], campaign["treatment"]
    raw = diff_in_means(y, t)
    adj = cuped(y, t, campaign["p0"])  # p0 is pre-treatment: a perfect CUPED covariate
    assert adj["variance_reduction"] > 0.05
    assert abs(adj["effect"] - raw["effect"]) < 2 * raw["se"]



# --- inputs that cannot support a test ---------------------------------------
# Each of these used to return a number. Three returned p = 0.0 - maximally
# "significant" - for an empty arm, a constant outcome and a 2/0 treatment.

_y = np.r_[np.ones(25), np.zeros(25), np.ones(10), np.zeros(40)]
_t = np.r_[np.ones(50), np.zeros(50)]


@pytest.mark.parametrize("y, t, why", [
    (_y[:3], np.ones(3), "at least 2 rows"),                      # empty control arm
    (np.zeros(100), _t, "does not vary"),                         # constant outcome
    (_y, np.r_[np.full(50, 2), np.zeros(50)], "must be 0/1"),     # 2/0 coding
    (_y, _t[:-1], "rows but treatment has"),                      # lengths differ
    (np.r_[_y[:-1], np.nan], _t, "outcome has missing"),
    (_y, np.r_[_t[:-1], np.nan], "treatment has missing"),
])
def test_untestable_inputs_raise_instead_of_returning_a_p_value(y, t, why):
    with pytest.raises(ValueError, match=why):
        diff_in_means(y, t)


def test_count_and_srm_inputs_are_checked():
    with pytest.raises(ValueError, match="successes"):
        diff_in_proportions(12, 10, 3, 10)                        # more successes than units
    with pytest.raises(ValueError, match="successes"):
        diff_in_proportions(1, 1, 3, 10)                          # a one-row arm
    with pytest.raises(ValueError, match="does not vary"):
        diff_in_proportions(0, 50, 0, 50)
    with pytest.raises(ValueError, match="between 0 and 1"):
        srm_check(counts=(50, 50), expected_treat_frac=1.0)
    with pytest.raises(ValueError, match="no units"):
        srm_check(counts=(0, 0))


def test_cuped_and_aipw_check_their_extra_inputs():
    with pytest.raises(ValueError, match="constant"):
        cuped(_y, _t, np.ones(100))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aipw(_y, _t, np.full(100, 0.5), np.full(100, 0.3), np.full(100, 1.2))
    with pytest.raises(ValueError, match="e must have"):
        aipw(_y, _t, np.full(100, 0.5), np.full(100, 0.3), np.full(99, 0.5))


def test_aipw_reports_how_many_propensities_it_clipped():
    e = np.r_[np.full(10, 0.001), np.full(90, 0.5)]
    assert aipw(_y, _t, np.full(100, 0.5), np.full(100, 0.2), e)["clipped"] == 10

# --- evaluation --------------------------------------------------------------

def test_qini_rewards_truth_and_not_noise(campaign):
    y, t, truth = campaign["outcome"], campaign["treatment"], campaign["true_uplift"]
    noise = np.random.default_rng(0).uniform(size=len(campaign))
    assert qini_coefficient(y, t, truth) > 0
    assert abs(qini_coefficient(y, t, noise)) < qini_coefficient(y, t, truth)
    assert uplift_at_k(y, t, truth, 0.2) > uplift_at_k(y, t, -truth, 0.2)



def test_policy_intervals_bracket_the_estimate_and_cover_the_planted_gain(campaign):
    y, t, truth = campaign["outcome"].to_numpy(), campaign["treatment"].to_numpy(), campaign["true_uplift"].to_numpy()
    scores = {"random": np.random.default_rng(1).uniform(size=len(y)), "truth": truth}
    iv = qini_at_k_intervals(y, t, scores, [0.2], n_boot=200)
    _, q = qini_curve(y, t, truth)
    m = int(round(0.2 * len(y)))
    lo, hi = iv["truth"][0.2]["extra"]
    assert lo < q[m] < hi
    # what the Qini value estimates: treated count in the top 20% x their mean uplift
    top = np.argsort(-truth, kind="stable")[:m]
    assert lo < t[top].sum() * truth[top].mean() < hi
    # the reference compared with itself has no gap; the true ranking clearly beats it
    assert iv["random"][0.2]["vs_reference"] == (0.0, 0.0)
    assert iv["truth"][0.2]["vs_reference"][0] > 0 and iv["truth"][0.2]["ratio"][0] > 1

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
    for model in (SLearner(), TLearner(), XLearner(), CausalForest(n_estimators=100)):
        model.fit(tr[FEATURES], tr["treatment"], tr["outcome"])
        pred = model.predict_uplift(te[FEATURES])
        assert np.corrcoef(te["true_uplift"], pred)[0, 1] > 0.5, model.name


def test_learned_propensity_learners_find_the_planted_effect_in_pooled_tests(pooled):
    # the configuration run_criteo uses: the treatment chance is learned from
    # the features, because pooled tests make it vary by customer
    df, e = pooled
    tr, te = df.iloc[:100_000], df.iloc[100_000:]
    x = XLearner(fit_propensity=True)
    for model in (x, CausalForest(n_estimators=100, fit_propensity=True)):
        model.fit(tr[FEATURES], tr["treatment"], tr["outcome"])
        pred = model.predict_uplift(te[FEATURES])
        assert np.corrcoef(te["true_uplift"], pred)[0, 1] > 0.5, model.name
    # and the treatment chance it learned is the one that was planted
    learned = x.propensity.predict_proba(te[FEATURES].to_numpy())[:, 1]
    assert np.corrcoef(e[100_000:], learned)[0, 1] > 0.9


def test_uplift_targeting_beats_risk_targeting(campaign):
    tr, te = campaign.iloc[:40_000], campaign.iloc[40_000:]
    x = XLearner().fit(tr[FEATURES], tr["treatment"], tr["outcome"])
    risk = RiskModel().fit(tr[FEATURES], tr["treatment"], tr["outcome"])
    truth = te["true_uplift"]
    assert (budget_gain(truth, x.predict_uplift(te[FEATURES]), 0.2)
            > budget_gain(truth, risk.predict_uplift(te[FEATURES]), 0.2))
