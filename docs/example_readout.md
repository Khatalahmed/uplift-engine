## Summary

We ran an uplift evaluation on the Criteo trial (dataset version 2.1) with 13,979,592 users overall and 6,987,897 held-out users for model evaluation. The treated share was 85%. We compared 4 uplift models plus two simple baselines (random and risk). The main takeaway: targeting by uplift strongly outperforms random targeting; targeting by risk produces almost no gain. Use the doubly robust (AIPW) estimate as the correct average treatment effect.

## Can we trust the test?

Arm sizes look fine: the sample-ratio check p-value is 0.999. Treatment assignment is slightly predictable from features (AUC 0.510 versus 0.5 for perfect randomness), and a user’s chance of being treated ranges from 84.5% to 89.1% at the extremes. Because several tests were pooled, that predictability biases the naive difference-in-means, so we prefer the corrected (AIPW) estimate.

## How big is the effect?

The naive treated-minus-control visit effect is +1.02 pp (95% interval +0.98 pp to +1.06 pp) but it overstates the corrected estimate by 36%. The AIPW (doubly robust) visit effect is +0.75 pp (95% interval +0.72 pp to +0.79 pp). Treat the AIPW numbers as the unbiased average effect.

## Who should we target?

Below is the ranking of targeting strategies by estimated extra visits at three budgets. Uplift models generate many more extra visits than random; risk targeting generates almost none.

| targeting | top 10% | top 20% | top 30% |
|---|---:|---:|---:|
| random | 6,264 | 12,790 | 18,441 |
| risk | 105 | 138 | 256 |
| S-learner | 37,424 | 44,313 | 51,310 |
| T-learner | 34,063 | 45,211 | 50,928 |
| X-learner | 35,809 | 44,696 | 52,207 |
| causal forest | 34,667 | 44,818 | 50,848 |

The best uplift model at top 10% delivers 6.0x the extra visits of random targeting.

## Recommendation

Use the AIPW estimate for average effect (+0.75 pp) and deploy an uplift-based targeting policy (pick the best-performing uplift model from the evaluation) rather than a risk-based policy.

## Caveats

Criteo’s dataset processing distorts absolute effect sizes for privacy; therefore the ranking and relative gains are more reliable than the raw magnitudes. The slight predictability of treatment means naive estimates are biased, so always use doubly robust corrections in this pooled-testing context.