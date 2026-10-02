"""Admission gate: a thin adapter over `cpgtr.certify`.

Wraps `cpgtr.certify.admit` (the paper's Algorithm 3, without the refinement
stage) and returns an `AdmitLog` that records, per candidate, whether it was
admitted, rejected, or left unprocessed because a wall-clock deadline passed,
plus the number of stratification rejections reported by the evaluation scripts.

Also defines `CONTEXTS`, the fixed (age, jurisdiction) evaluation grid.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .certify import admit as _certify_admit, nac_is_vacuous

# Fixed (age, jurisdiction) grid used to instantiate benchmark prompts and
# measure compliance: ages {9, 15, 21} crossed with jurisdictions {US, EU, UK}.
# Certification does not iterate over this grid. `certify.admit` defaults to the
# context cells of `cpgtr.context`, because the termination and confluence
# guarantees must hold for every (A, J), not just these nine sample points.
# Pass `CONTEXTS` as `contexts=` only for smaller, faster ad hoc checks.
CONTEXTS = [(a, j) for a in (9, 15, 21) for j in ("US", "EU", "UK")]


@dataclass
class AdmitLog:
    """Audit record of one admission run.

    `nonterm_incidents` is not populated by admission: a non-termination
    incident is an empirical event observed while repairing a held-out host
    (see `eval_repair.run_repair`), not an admission-time count. The field is
    kept so callers that read it get 0 rather than an AttributeError.
    `strat_rejections` counts candidates rejected with "reject:no-measure".
    """
    nonterm_incidents: int = 0
    strat_rejections: int = 0
    budget_exceeded: bool = False
    admitted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    unprocessed: list = field(default_factory=list)   # deadline hit before reaching these
    notes: list = field(default_factory=list)


def admit(candidates, contexts=None, *, enable_cpa=True, enable_strat=True, dmax=None,
          edgefree="keep", deadline=None):
    """Run the admission gate by delegating to `cpgtr.certify.admit`.

    Args:
        candidates: iterable of candidate rules (from `extract` or `library`).
        contexts: optional explicit list of (A, J) points to check for
            confluence. None (default) lets `certify.admit` compute the
            context cells of `candidates`; pass `CONTEXTS` or another list only
            for smaller, faster ad hoc checks.
        enable_cpa: enable the critical-pair / strong-joinability confluence
            check (disable for the no-CPA ablation).
        enable_strat: use the global stratified repair measure; if False, use
            the naive size-measure fallback (the no-stratification ablation).
        dmax: maximum stratification rank; None uses len(candidates), which
            always suffices.
        edgefree: "keep" (default) or "reject": whether a candidate whose L
            has no edge goes through the normal pipeline or is rejected
            outright. See `certify.admit`.
        deadline: optional `time.time()`-style wall-clock cutoff. See
            `certify.admit` for the "budget-exceeded" semantics.

    Returns:
        (cert, log): `cert` is the certified rule list in admission order and
        `log` is an `AdmitLog` with `strat_rejections` populated.
    """
    candidates = list(candidates)
    # Match raw_log entries back to Rule objects by position, not by `.name`:
    # names are not unique across extracted candidates, so a name-keyed lookup
    # could attribute a log entry to the wrong rule. `certify.admit` sorts its
    # input by (-confidence, content_hash()), logs edge-free rejections first
    # when edgefree="reject", then vacuous-NAC rejections, then the remaining
    # candidates. Reproduce that exact order here so raw_log[i] corresponds to
    # ordered_for_log[i] with no lookup.
    sorted_candidates = sorted(
        candidates,
        key=lambda r: (-(r.confidence if r.confidence is not None else -1),
                        r.content_hash()),
    )
    if edgefree == "reject":
        edgefree_first = [r for r in sorted_candidates if not r.L.edges]
        after_edgefree = [r for r in sorted_candidates if r.L.edges]
    else:
        edgefree_first, after_edgefree = [], sorted_candidates
    # Vacuous-NAC candidates are split out next, preserving the sorted order.
    vacuous_nac_first = [r for r in after_edgefree if nac_is_vacuous(r)]
    kept = [r for r in after_edgefree if not nac_is_vacuous(r)]
    ordered_for_log = edgefree_first + vacuous_nac_first + kept

    cert, raw_log = _certify_admit(
        candidates, dmax=dmax, contexts=contexts,
        use_cpa=enable_cpa, use_strat=enable_strat, verbose=False,
        edgefree=edgefree, deadline=deadline,
    )
    # `certify.admit` logs every remaining candidate as "budget-exceeded" when
    # the deadline passes, so the lengths still match and the positional
    # correspondence holds for an unfinished run.
    assert len(raw_log) == len(ordered_for_log), (
        f"internal order mismatch: raw_log has {len(raw_log)} entries, "
        f"expected {len(ordered_for_log)} -- certify.admit()'s internal "
        f"sort/edgefree-split logic no longer matches this replication"
    )

    log = AdmitLog()
    for (name, status, _modality, _detail), rule in zip(raw_log, ordered_for_log):
        log.notes.append((name, status))
        if status == "admit":
            log.admitted.append(rule)
        elif status == "budget-exceeded":
            log.unprocessed.append(rule)
            log.budget_exceeded = True
        else:
            log.rejected.append(rule)
            if status == "reject:no-measure":
                log.strat_rejections += 1

    return cert, log
