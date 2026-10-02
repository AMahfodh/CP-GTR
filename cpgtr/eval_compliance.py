"""Contextual Compliance Rate (CCR) aggregation with bootstrap confidence intervals.

CCR is the percentage of outputs with zero gold-annotated violations for their
context, reported with a 95% bootstrap confidence interval.

"Gold-annotated" applies to every method, including CP-GTR. A method's own
round-trip check (`cpgtr.realize.round_trip`) answers "did this reach the
certified target", which is a different question from "is this compliant under
the statute", and is reported separately as enforcement soundness so that no
method grades its own output. This module therefore does not generate or judge
anything: it only aggregates outputs that have already been labeled by human
annotators against the written codebook (see ``docs/table4_pilot/`` and
`codebook.py`).
"""
from __future__ import annotations
import random


def compliance_rate(labeled_outputs, age, *, n_boot=2000, seed=0, ci=0.95):
    """Compliance rate (percent) at one age, with a percentile bootstrap CI.

    Args:
        labeled_outputs: list of dicts, one per (prompt, context) pair for a
            single method's generated output:
                {"prompt_id": str, "age": int, "jurisdiction": str,
                 "violations": list[str]}
            "violations" is the gold-annotated list of violation codes found
            in the output text for that context (empty list = compliant). It
            is produced by human annotation, not computed here.
        age: the age column to report. Entries at this age are pooled across
            all jurisdictions present in `labeled_outputs` into one rate.
        n_boot, seed: bootstrap resample count and RNG seed (for
            reproducibility).
        ci: confidence level (default 0.95).

    Returns:
        ``(rate_pct, (ci_lo, ci_hi))``. The interval is obtained by resampling
        the pooled (prompt, context) entries at this age with replacement.

    Raises:
        ValueError: if there are no labeled outputs for this age. A rate is
        never computed from an empty set.
    """
    items = [e for e in labeled_outputs if e["age"] == age]
    if not items:
        raise ValueError(
            f"no gold-labeled outputs for age={age}. Real compliance-rate numbers "
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
