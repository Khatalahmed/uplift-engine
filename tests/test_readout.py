"""The readout checker: real numbers pass, invented ones are caught, and a
draft that never gets its numbers right is never published. No network."""
import pytest

from uplift.readout import ReadoutRejected, facts, verify, write_readout


@pytest.fixture
def report():
    rows = [{"model": m, **{f"extra_visits@{k}%": v * k for k in (10, 20, 30)}}
            for m, v in [("random", 626.4), ("risk", 10.5), ("s_learner", 3742.4),
                         ("t_learner", 3406.3), ("x_learner", 3580.9), ("causal_forest", 3466.7)]]
    return {
        "n": 13_979_592, "eval_n": 6_987_897,
        "srm_vs_0.85": {"observed_treat_frac": 0.85, "p_value": 0.999},
        "treatment_predictability_auc": 0.5101,
        "propensity_pct_1_50_99": [0.845, 0.848, 0.891],
        "ate_visit": {"control_mean": 0.038201, "treatment_mean": 0.048543},
        "ate_visit_naive_eval": {"effect": 0.010222, "ci_low": 0.009816, "ci_high": 0.010628},
        "ate_visit_aipw_eval": {"effect": 0.007504, "ci_low": 0.007152, "ci_high": 0.007856},
        "models": rows,
    }


def test_facts_are_formatted_once(report):
    f = facts(report)
    assert f["users"]["value"] == "13,979,592"
    assert f["aipw_effect"]["value"] == "+0.75 pp"
    assert f["naive_overstatement"]["value"] == "36%"
    assert f["s_learner_extra_visits_10"]["value"] == "37,424"
    assert f["best_vs_random_10"]["value"] == "6.0x"


def test_true_numbers_and_fair_roundings_pass(report):
    text = ("## Summary\n1. Across 13,979,592 users (about 14M) the corrected effect is +0.75 pp "
            "(95% CI +0.72 pp to +0.79 pp), versus a naive +1.02 pp that is 36% too high. "
            "Visit rates were 3.82% vs 4.85% (roughly 3.8% vs 4.9%). Treatment chance ran "
            "84.5%–89.1% (AUC 0.510 vs 0.5). Targeting the top 10% by the S-learner adds "
            "37,424 visits, 6.0x random; risk targeting adds 105. We compared 4 models.")
    check = verify(text, facts(report))
    assert check["ok"], check["unknown"]


@pytest.mark.parametrize("lie", [
    "a 40% lift",              # invented percentage
    "+0.85 pp",                # wrong effect
    "37,400 extra visits",     # 'rounded' an integer the model must copy
    "-0.75 pp",                # flipped sign
    "since 2018",              # a year is a number too
    "3 models",                # small integers must match exactly
])
def test_invented_numbers_are_caught(report, lie):
    check = verify(f"The effect was +0.75 pp, and {lie}.", facts(report))
    assert not check["ok"]
    assert len(check["unknown"]) == 1


def test_internal_names_are_rejected_even_with_true_numbers(report):
    check = verify("treatment_auc = 0.510; the s_learner adds 37,424 visits.", facts(report))
    assert not check["ok"] and check["unknown"] == []
    assert check["leaked_names"] == ["s_learner", "treatment_auc"]


def test_retry_fixes_a_bad_draft_and_logs_the_attempts(report):
    drafts = iter(["Uplift gives a 40% lift.", "Uplift gives +0.75 pp, 6.0x random."])
    seen = []

    def fake_llm(messages):
        seen.append(messages)
        return next(drafts)

    text, check = write_readout(report, fake_llm)
    assert check["ok"] and check["attempts"] == 2
    assert "40" in seen[1][-1]["content"]  # the retry names the exact offender
    assert check["rejected_drafts"] == [{"attempt": 1, "unknown_numbers": ["40"], "leaked_names": []}]


def test_persistent_hallucination_is_never_published(report):
    with pytest.raises(ReadoutRejected) as e:
        write_readout(report, lambda messages: "A 40% lift, trust me.", max_attempts=3)
    assert e.value.check["attempts"] == 3 and e.value.check["unknown"] == ["40"]
