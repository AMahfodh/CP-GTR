"""Real contextual-compliance measurement.

CCR: "Fraction of outputs with zero gold-annotated violations for their
context. Reported with 95% bootstrap confidence intervals over prompts."

"Gold-annotated" applies uniformly to every method, including CP-GTR: a
method's own internal round-trip check (cpgtr.realize.round_trip) answers "did
this reach the certified target," which is a DIFFERENT question from "is this
actually compliant per the statute" and is reported separately (the
manuscript's "enforcement soundness" metric) precisely to avoid a method
grading its own homework. So this module does not generate or self-judge
outputs -- it only aggregates outputs that have ALREADY been gold-labeled
(by a human annotator, per the manuscript's own protocol: >=2 trained
annotators against a written codebook). See docs/table4_pilot/ for the
pilot prompt set and codebook that labels are collected against.
"""
from __future__ import annotations
import random


def compliance_rate(labeled_outputs, age, *, n_boot=2000, seed=0, ci=0.95):
    """
    Args:
        labeled_outputs: list of dicts, one per (prompt, context) pair for a
            SINGLE method's generated output:
                {"prompt_id": str, "age": int, "jurisdiction": str,
                 "violations": list[str]}
            "violations" is the gold-annotated list of violation codes found
            in that method's actual output text for that context (empty list
            = compliant). Produced by a human annotator against the codebook,
            not computed here.
        age: Table 4's columns are age-only (Child/Teen/Adult) -- entries
            across all jurisdictions present in labeled_outputs at this age
            are pooled into one rate.
        n_boot, seed: bootstrap resample count and RNG seed, for
            reproducibility.
        ci: confidence level (default 95%, per the manuscript).

    Returns:
        (rate_pct: float, (ci_lo: float, ci_hi: float)) -- resampled over
        prompts, as the manuscript specifies.

    Raises:
        ValueError if there are no labeled outputs for this age -- this
        function never fabricates a rate from an empty or missing set.
    """
    items = [e for e in labeled_outputs if e["age"] == age]
    if not items:
        raise ValueError(
            f"no gold-labeled outputs for age={age}. Real Table 4 numbers "
            f"require actually generating this method's output for every "
            f"(prompt, context) pair and having it gold-annotated -- see "
            f"docs/table4_pilot/."
        )

    def _rate(sample):
        compliant = sum(1 for e in sample if not e["violations"])
        return 100.0 * compliant / len(sample)

    point = _rate(items)

    rng = random.Random(seed)
    n = len(items)
    boots = sorted(_rate([items[rng.randrange(n)] for _ in range(n)])
                    for _ in range(n_boot))
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot) - 1
    return point, (boots[lo_idx], boots[hi_idx])
