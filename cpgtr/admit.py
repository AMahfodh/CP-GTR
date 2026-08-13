"""
cpgtr.admit — admission gate for CP-GTR.

Adapts the faithful certification implementation in cpgtr.certify /
cpgtr.cpa (Algorithm 3, Bootstrap-and-Refine; refinement/Stage 3
is not implemented here, matching certify.py) to the AdmitLog-based interface
that run_e2e.py  read.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .certify import admit as _certify_admit


CONTEXTS = [(a, j) for a in (9, 15, 21) for j in ("US", "EU", "UK")]


@dataclass
class AdmitLog:
    """Audit record. Tables 7/8 read strat_rejections directly.

    nonterm_incidents is NOT populated here: a non-termination incident is an
    empirical event observed while actually repairing a held-out host (see
    eval_repair.run_repair), not a static, admission-time count. It is kept
    on this dataclass only so any caller that reads it gets 0 rather than an
    AttributeError.
    """
    nonterm_incidents: int = 0
    strat_rejections: int = 0
    admitted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def admit(candidates, contexts=None, *, enable_cpa=True, enable_strat=True, dmax=None):
    """
    Run the admission gate by delegating to cpgtr.certify.admit (the
    manuscript's Algorithm 3 termination + confluence checks).

    Args:
        candidates:  iterable of candidate rules (from extract() or lib)
        contexts:    optional override of the (A,J) points checked for
                     confluence. Defaults to None, which makes certify.admit()
                     compute the CONTEXT CELLS of `candidates` automatically
                     (CP-GTR_V2.tex "Context cells") -- pass admit.CONTEXTS or
                     another explicit list only for smaller/faster ad hoc
                     checks (e.g. tests), not for anything whose result is
                     meant to be manuscript-faithful.
        enable_cpa:  toggle critical-pair / strong-joinability confluence check
                     (--CPA ablation)
        enable_strat:toggle the real GLOBAL stratified repair-typing measure
                     vs. the naive size-measure fallback (--Strat ablation)
        dmax:        max stratification rank tried by certify.stratify().
                     Defaults to None, which makes certify.stratify() use
                     len(candidates) -- the bound the manuscript's own
                     Proposition gives (a rank reaching |R| witnesses a
                     cycle), not an arbitrary small constant.

    Returns:
        (cert, log) where cert is the certified rule list (in admission order)
        and log is an AdmitLog with strat_rejections populated for Tables 7/8.
    """
    candidates = list(candidates)
    by_name = {r.name: r for r in candidates}

    cert, raw_log = _certify_admit(
        candidates, dmax=dmax, contexts=contexts,
        use_cpa=enable_cpa, use_strat=enable_strat, verbose=False,
    )

    log = AdmitLog()
    for name, status, _modality, _detail in raw_log:
        rule = by_name[name]
        log.notes.append((name, status))
        if status == "admit":
            log.admitted.append(rule)
        else:
            log.rejected.append(rule)
            if status == "reject:no-measure":
                log.strat_rejections += 1

    return cert, log
