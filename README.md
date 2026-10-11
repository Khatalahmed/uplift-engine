# Uplift Engine

**Who will a campaign actually change?** Not "who is at risk": who will behave differently *because* of the offer.

The sequel to [customer_churn_alert_agent](https://github.com/Khatalahmed/customer_churn_alert_agent). That project ranks customers by churn risk and prices actions with an *assumed* uplift. Its first hold-out test (11 treated vs 4 control) came back inconclusive, and so does its current one: 300 customers per arm, +1.0 pp, p = 0.68. Its own power analysis asks for ~973 per arm just to settle the *average* effect; with 11–14 churn events per arm, *which* customer responds is far out of reach. This project answers that question where the data allows it: valid experiment checks, power analysis, uplift models and a budget decision. Every method is first tested against a known answer, then run on a 14M-user real randomized trial.

## 1. Synthetic campaign: does each method recover a planted answer?

200,000 customers, 50/50 randomized offer, true average effect +10.3 pp.

| Check | Result |
|---|---|
| Sample-ratio mismatch | none (p = 0.72) |
| Average effect | +10.1 pp, 95% CI [+9.7, +10.5]: **covers the truth** |
| CUPED (cross-fitted covariate) | same estimate, **18% less variance** |

Models trained on one half, judged on the other. "Saved" is the **true** number of extra returning customers when the top 20% by each score get the offer:

| Targeting by | Correlation with true uplift | Saved @ 20% budget |
|---|---|---|
| random | — | 2,076 |
| **churn risk** (the usual approach) | — | 2,172 |
| T-learner | 0.87 | 4,663 |
| S-learner | 0.92 | 4,735 |
| X-learner | 0.95 | 4,812 |
| **Causal forest** (EconML) | **0.95** | **4,835** |
| perfect knowledge (ceiling) | 1.00 | 4,943 |

**Targeting by risk is barely better than random.** The riskiest customers include lost causes the offer can't win back. Targeting by uplift saves **2.2× as many customers** for the same budget, 98% of the ceiling.

**Power analysis:** to detect a +5 pp lift on an 8% base rate, a test needs **589 customers per arm**. The churn repo's first test, with a 4-customer control arm, could only have detected a lift of +71 pp, and its current 300-per-arm experiment reports that it would need ~973.

## 2. Criteo: 13,979,592 real users from a randomized ad trial

Data: [Criteo Uplift v2.1](https://huggingface.co/datasets/criteo/criteo-uplift), ~85% treated, outcome = site visit. Experiment checks run in DuckDB SQL over all rows; models train on a frozen 10% fold and are judged on a separate 50% (6,987,897 users).

**The trap in the data.** The arms are the right size (SRM p = 0.999) and no feature fails the usual balance rule (|SMD| < 0.1). But the features *predict who was treated* (AUC 0.510; P(treated) ranges 0.845 → 0.891). Criteo built the dataset by pooling several incrementality tests with different splits, so the plain treated-minus-control difference is confounded, and CUPED, which assumes balanced features, is invalid here.

| Average effect on visits | Estimate | 95% CI |
|---|---|---|
| Naive difference in means | +1.02 pp | [+0.98, +1.06] |
| **Doubly robust (AIPW)**, learned propensity | **+0.75 pp** | [+0.72, +0.79] |

**The naive number overstates the effect by 36%.** A unit test reproduces the trap: two pooled tests with different splits; the naive interval misses the planted truth and AIPW covers it.

> Criteo sub-sampled the data so the original incrementality can't be recovered: absolute effects are deliberately distorted. **Rankings are what this dataset can judge.**

**Who to target.** Estimated extra visits among the 7M held-out users when the top *k*% by each score are treated:

| Targeting by | Top 10% | Top 20% | Top 30% |
|---|---|---|---|
| random | 6,264 | 12,790 | 18,441 |
| **lowest predicted visit chance ("risk")** | **105** | **138** | **256** |
| S-learner | **37,424** | 44,313 | 51,310 |
| T-learner | 34,063 | **45,211** | 50,928 |
| X-learner (learned propensity) | 35,809 | 44,696 | **52,207** |
| Causal forest (learned propensity) | 34,667 | 44,818 | 50,848 |

- **Uplift targeting: ~6× the extra visits of random** at a 10% budget.
- **Risk targeting: about zero.** The users least likely to visit are not the ones an ad moves. Same lesson as the synthetic campaign, now on real data.
- Learning the propensity lifted the X-learner's Qini from 0.0020 to 0.0029.

## 3. A readout the LLM can't make up numbers in

`uplift/readout.py` has Azure OpenAI write the plain-English experiment readout for a marketing lead, under one rule enforced in code: **every number must be one the pipeline computed.**

1. The pipeline's results are formatted once into a fact table (`+0.75 pp`, `37,424`, `36%`), each with its meaning.
2. The model drafts the readout from those facts.
3. A checker extracts every number from the draft and traces it to a fact. It accepts exact copies or fair roundings (`4.85%` → `4.9%`); integers under 100 must match exactly; a flipped sign fails; leaked internal names (`treatment_auc`) fail.
4. Any failure → the exact offenders go back to the model; after 3 failed drafts **nothing is published**.
5. `results/readout_check.json` maps every published number to its source fact and keeps each rejected draft's errors.

Example output from a real run: [the readout](docs/example_readout.md) and [its audit log](docs/example_readout_check.json).

In one real run the first draft was rejected for writing "1st/99th percentile", numbers the pipeline never produced. The second draft passed with 46 of 46 numbers traced. The checker guards *numbers*: a real number attached to the wrong model would pass, which is why the log records which fact each number matched.

## Run it

```bash
uv sync
uv run pytest                          # 26 checks against known answers (no network)
uv run python -m uplift.run_synthetic  # synthetic pipeline -> results/synthetic_report.json
uv run python -m uplift.power --p-control 0.08 --n-control 4 --n-treatment 11

uv run python -m uplift.data download  # 311 MB, checksum-verified
uv run python -m uplift.data parquet   # ~1 min, adds frozen folds
uv run python -m uplift.run_criteo     # ~8 min on a 4-core laptop -> results/criteo_report.json
cp .env.example .env                   # add Azure OpenAI credentials
uv run python -m uplift.readout        # verified readout -> results/readout.md
```

## Layout

```
uplift/
├── simulate.py       randomized campaign with a planted per-customer effect
├── experiment.py     SRM, balance, effect + CI (rows or SQL counts), CUPED, AIPW
├── power.py          sample size, minimum detectable effect, achieved power
├── learners.py       S / T / X-learners, causal forest, churn-risk baseline
├── evaluate.py       Qini curve & coefficient, uplift@k, PEHE, true budget gain
├── data.py           Criteo download (verified) → Parquet with frozen folds → DuckDB SQL
├── run_synthetic.py  pipeline on the synthetic campaign
├── run_criteo.py     pipeline on the 14M-user trial
└── readout.py        LLM-written readout, every number verified against computed facts
```

## Roadmap

- [x] Experiment checks, power, uplift learners, evaluation, all validated on planted truth
- [x] Causal forest (EconML) as a fourth learner
- [x] Criteo Uplift v2.1: 14M users, pooled-test confounding found and corrected (AIPW)
- [x] LLM-written experiment readout (Azure OpenAI), rejected if any number disagrees with the computed results
- [ ] Re-run the churn repo's coupon test at the powered sample size; replace its assumed uplift with a measured model
- [ ] FastAPI scoring service

## Data licence & citation

Criteo Uplift v2.1 is CC BY-NC-SA 4.0 and is never committed to this repo.
Diemert, Betlei, Renaudin, Amini. *A Large Scale Benchmark for Uplift Modeling.* AdKDD workshop, KDD 2018.
