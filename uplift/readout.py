"""
readout.py

WHAT : An LLM writes the plain-English experiment readout for the Criteo
       run - and a checker refuses to publish it if any number in the text
       is not one the pipeline actually computed.
WHY  : A readout is where analysis turns into decisions, and LLMs are
       fluent at inventing plausible numbers ("~40% lift", "2.3x"). The
       model is allowed to write; it is not allowed to calculate.
FLOW : report.json -> facts() (every number, pre-formatted, with meaning)
       -> LLM drafts -> verify() extracts every number from the draft
       -> unknown number? send the exact offenders back, retry (max 3)
       -> still wrong? ReadoutRejected, nothing is written
       -> results/readout.md + results/readout_check.json (number -> fact)
LOGIC:
  - Facts are formatted ONCE, here. The prompt says to copy them exactly.
  - A number in the draft passes if it equals a fact, or is that fact
    rounded to fewer decimals (0.7504 -> "0.75" -> "0.8"). Integers under
    100 must match exactly, so "4 models" can't pass by rounding 3.82.
  - Sign matters: "-0.75" does not match "+0.75".
  - List markers ("1. ") are not numbers.
LIMIT: this guards NUMBERS. A draft that attaches a real number to the
       wrong model would pass; the per-number log shows which fact each
       number matched so a reviewer can spot that.

USAGE
    python -m uplift.readout                    # needs AZURE_OPENAI_* in .env
    python -m uplift.readout --report results/criteo_report.json
"""
import argparse
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable

RESULTS = Path("results")
MAX_ATTEMPTS = 3
BUDGETS = (10, 20, 30)
UPLIFT_MODELS = ("s_learner", "t_learner", "x_learner", "causal_forest")

Complete = Callable[[list[dict]], str]


class ReadoutRejected(RuntimeError):
    def __init__(self, check: dict):
        super().__init__(f"readout rejected: numbers not in the facts: {check['unknown']}; "
                         f"leaked internal names: {check['leaked_names']}")
        self.check = check


# --------------------------------------------------------------------------- #
# Facts: every number the readout may use, formatted once
# --------------------------------------------------------------------------- #

def facts(r: dict) -> dict[str, dict]:
    """name -> {value: exact text to copy, meaning: what it is}."""
    srm, naive, dr = r["srm_vs_0.85"], r["ate_visit_naive_eval"], r["ate_visit_aipw_eval"]
    lo, _, hi = r["propensity_pct_1_50_99"]
    rows = {m["model"]: m for m in r["models"]}
    pp = lambda x: f"{x * 100:+.2f} pp"

    f = {
        "users": (f"{r['n']:,}", "users in the Criteo trial"),
        "users_rounded": (f"{r['n'] / 1e6:.0f}M", "same, rounded"),
        "eval_users": (f"{r['eval_n']:,}", "held-out users the models were judged on"),
        "eval_users_rounded": (f"{r['eval_n'] / 1e6:.0f}M", "same, rounded"),
        "dataset_version": ("2.1", "Criteo Uplift dataset version"),
        "treated_share": (f"{srm['observed_treat_frac']:.0%}", "share of users treated"),
        "srm_p": (f"{srm['p_value']:.3f}", "sample-ratio check p-value (high = arm sizes fine)"),
        "treatment_auc": (f"{r['treatment_predictability_auc']:.3f}",
                          "AUC of predicting WHO was treated from features"),
        "random_auc": ("0.5", "AUC if treatment were perfectly random"),
        "p_treated_low": (f"{lo:.1%}", "1st percentile of a user's chance of being treated"),
        "p_treated_high": (f"{hi:.1%}", "99th percentile of a user's chance of being treated"),
        "control_visit_rate": (f"{r['ate_visit']['control_mean']:.2%}", "visit rate, control"),
        "treated_visit_rate": (f"{r['ate_visit']['treatment_mean']:.2%}", "visit rate, treated"),
        "ci_level": ("95%", "confidence level of all intervals"),
        "naive_effect": (pp(naive["effect"]), "naive treated-minus-control visit effect (biased here)"),
        "naive_ci_low": (pp(naive["ci_low"]), "naive effect, interval low"),
        "naive_ci_high": (pp(naive["ci_high"]), "naive effect, interval high"),
        "aipw_effect": (pp(dr["effect"]), "doubly robust (AIPW) visit effect - the corrected estimate"),
        "aipw_ci_low": (pp(dr["ci_low"]), "AIPW interval low"),
        "aipw_ci_high": (pp(dr["ci_high"]), "AIPW interval high"),
        "naive_overstatement": (f"{naive['effect'] / dr['effect'] - 1:.0%}",
                                "how much the naive effect overstates the AIPW effect"),
        "uplift_model_count": (str(len(UPLIFT_MODELS)), "number of uplift models compared"),
    }
    for k in BUDGETS:
        f[f"budget_{k}"] = (f"{k}%", f"targeting budget: top {k}% of users")
        for name in ("random", "risk", *UPLIFT_MODELS):
            v = rows[name][f"extra_visits@{k}%"]
            f[f"{name}_extra_visits_{k}"] = (f"{v:,.0f}", f"estimated extra visits, top {k}%, targeting by {name}")
    best10 = max(rows[m]["extra_visits@10%"] for m in UPLIFT_MODELS)
    f["best_vs_random_10"] = (f"{best10 / rows['random']['extra_visits@10%']:.1f}x",
                              "best uplift model's extra visits / random's, top 10%")
    return {k: {"value": v, "meaning": m} for k, (v, m) in f.items()}


# --------------------------------------------------------------------------- #
# Verification: every number in the text must be a fact
# --------------------------------------------------------------------------- #

_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:#+\s*)?\d+[.)]\s")


def _numbers(text: str) -> list[tuple[str, float, int]]:
    """(as written, signed value, decimals) for every number in the text."""
    text = _LIST_MARKER.sub("", text)
    out = []
    for m in _NUM.finditer(text):
        raw = m.group()
        if m.start() > 0 and (text[m.start() - 1].isalpha() or text[m.start() - 1] == "_"):
            continue  # part of a word, e.g. "f11", "gpt5"
        value = float(raw.replace(",", ""))
        before, before2 = text[m.start() - 1:m.start()], text[max(0, m.start() - 2):m.start() - 1]
        if before in "-−" and before and (before2 == "" or before2 in " ([\n\t"):
            value, raw = -value, before + raw
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((raw, value, decimals))
    return out


def _round_half_up(x: float, decimals: int) -> float:
    """Round like a person: 4.85 -> 4.9 (float round() gives 4.8, since 4.85
    is stored as 4.8499...). repr() recovers the number as it was written."""
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def verify(text: str, fact_table: dict[str, dict]) -> dict:
    """Match every number in `text` to a fact. ok = nothing unmatched."""
    allowed = []  # (value, decimals, fact name)
    for name, fact in fact_table.items():
        for _, value, dec in _numbers(fact["value"]):
            allowed.append((value, dec, name))

    matched, unknown = [], []
    for raw, value, dec in _numbers(text):
        hit = None
        for a, a_dec, name in allowed:
            if dec == 0 and abs(value) < 100:
                ok = a_dec == 0 and a == value               # small integers: exact only
            else:
                ok = dec <= a_dec and _round_half_up(a, dec) == value  # exact or a fair rounding
            if ok:
                hit = name
                break
        (matched if hit else unknown).append({"number": raw, "fact": hit} if hit else raw)

    # internal names ("treatment_auc", "s_learner") are for us, not the reader.
    # Only snake_case names count: a key like "users" is also plain English.
    names = [n for n in (*fact_table, *UPLIFT_MODELS) if "_" in n]
    leaked = sorted({n for n in names if re.search(rf"\b{re.escape(n)}\b", text)})
    return {"ok": not unknown and not leaked, "matched": matched, "unknown": unknown,
            "leaked_names": leaked}


# --------------------------------------------------------------------------- #
# Drafting with retries
# --------------------------------------------------------------------------- #

SYSTEM = """You are a senior data scientist writing an experiment readout for a marketing lead.

Rules - a program checks every number you write and rejects the readout if any is wrong:
- Use ONLY the facts provided. Copy every number EXACTLY as written in "value".
- Never calculate, round differently, estimate or introduce any other number
  (no years, no counts, no percentages that are not in the facts).
- Prefer words over numbers when a number is not essential.
- Never write the fact names (the JSON keys); describe things in plain English.
  Call the models "S-learner", "T-learner", "X-learner" and "causal forest".
- Do not list every model at every budget in prose. Put the targeting results in ONE
  Markdown table (rows: random, risk, the four models; columns: top 10% / 20% / 30%).

Write Markdown, 250-350 words plus the table, for a reader who is not a statistician,
with these sections:
## Summary
## Can we trust the test?
## How big is the effect?
## Who should we target?
## Recommendation
## Caveats

Key points the facts support: arm sizes are fine, but treatment is slightly predictable
from the features because several tests were pooled, so the naive effect is biased and
the AIPW estimate is the one to use; targeting by uplift beats random by a wide margin,
while targeting by risk yields almost nothing; Criteo distorted absolute effect sizes for
privacy, so the ranking of targeting strategies is what to trust."""


def _first_prompt(fact_table: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "FACTS:\n" + json.dumps(fact_table, indent=1)}]


def write_readout(report: dict, complete: Complete, max_attempts: int = MAX_ATTEMPTS) -> tuple[str, dict]:
    """Draft -> verify -> retry with the exact offenders. Raises ReadoutRejected."""
    fact_table = facts(report)
    messages = _first_prompt(fact_table)
    rejected = []  # what each failed draft got wrong - the audit trail
    check = {}
    for attempt in range(1, max_attempts + 1):
        text = complete(messages)
        check = verify(text, fact_table) | {"attempts": attempt, "rejected_drafts": rejected}
        if check["ok"]:
            return text, check
        rejected.append({"attempt": attempt, "unknown_numbers": check["unknown"],
                         "leaked_names": check["leaked_names"]})
        problems = []
        if check["unknown"]:
            problems.append(f"these numbers are not in the facts: {check['unknown']}")
        if check["leaked_names"]:
            problems.append(f"these internal fact names appear in the text: {check['leaked_names']}")
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                f"Rejected: {'; '.join(problems)}. Rewrite the whole readout, using only numbers "
                "copied exactly from the facts (or words instead) and plain-English names.")},
        ]
    raise ReadoutRejected(check)


def azure_complete() -> Complete:
    """Azure OpenAI via the v1 endpoint (same pattern as the churn and ACME repos)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
                    api_key=os.environ["AZURE_OPENAI_API_KEY"])
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def complete(messages: list[dict]) -> str:
        # reasoning model: no temperature; leave room for reasoning tokens
        resp = client.chat.completions.create(model=deployment, messages=messages,
                                              max_completion_tokens=6000)
        return resp.choices[0].message.content or ""
    return complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=RESULTS / "criteo_report.json")
    a = ap.parse_args()

    report = json.loads(a.report.read_text())
    try:
        text, check = write_readout(report, azure_complete())
    except ReadoutRejected as e:
        (RESULTS / "readout_check.json").write_text(json.dumps(e.check, indent=2))
        raise SystemExit(f"{e}\nNothing written. See {RESULTS / 'readout_check.json'}")

    (RESULTS / "readout.md").write_text(text, encoding="utf-8")
    (RESULTS / "readout_check.json").write_text(json.dumps(check, indent=2))
    print(text)
    print(f"\nverified: {len(check['matched'])} numbers, all traced to computed facts "
          f"(attempt {check['attempts']} of {MAX_ATTEMPTS})")
    print(f"wrote {RESULTS / 'readout.md'} and {RESULTS / 'readout_check.json'}")


if __name__ == "__main__":
    main()
