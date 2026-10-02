"""Context cells: the partition of (age, jurisdiction) space by active rule set.

Every rule's guard (`Rule.phi`) is a Boolean combination of age comparisons
over a finite jurisdiction set, so the guards induce a finite partition of
N x J into cells on which the set of active rules is constant. Certification
quantifies over these cells rather than over the fixed evaluation grid, which
only samples cells for measurement; the termination and confluence guarantees
must hold for every (A, J).

`Rule.phi` is an opaque callable (hand-written rules and LLM-extracted rules
both end up as closures), so cells are computed by exact enumeration over a
closed domain rather than by symbolic threshold extraction. This is exact
within that domain: `JURISDICTIONS` is the closed set the extraction schema
can emit, and `AGE_RANGE` covers every consent-age threshold in the corpus
(13, 16, 18).
"""
from __future__ import annotations

JURISDICTIONS = ("US", "EU", "UK", "US-CA")
AGE_RANGE = range(0, 121)


def active_signature(rules, A, J):
    """Return the set of `id(rule)` for rules in `rules` active at (A, J).

    This signature distinguishes one context cell from its neighbours.
    """
    return frozenset(id(r) for r in rules if r.phi(A, J))


def context_cells(rules):
    """Partition JURISDICTIONS x AGE_RANGE into maximal contiguous-age cells
    on which the active subset of `rules` is constant.

    Returns a list of dicts: {"jurisdiction", "age_lo", "age_hi", "active"},
    where "active" is the list of rules active throughout that cell (one
    representative age suffices to characterize the whole cell, since the
    active set does not change within it).
    """
    rules = list(rules)
    cells = []
    for J in JURISDICTIONS:
        cur_sig, cur_lo = None, None
        for A in AGE_RANGE:
            sig = active_signature(rules, A, J)
            if sig != cur_sig:
                if cur_sig is not None:
                    cells.append({
                        "jurisdiction": J, "age_lo": cur_lo, "age_hi": A - 1,
                        "active": [r for r in rules if id(r) in cur_sig],
                    })
                cur_sig, cur_lo = sig, A
        cells.append({
            "jurisdiction": J, "age_lo": cur_lo, "age_hi": AGE_RANGE[-1],
            "active": [r for r in rules if id(r) in cur_sig],
        })
    return cells


def representative_contexts(rules):
    """Return one (A, J) witness per context cell with at least one active rule.

    Checking a property at these points is equivalent to checking it at every
    (A, J), because the active set, and anything that depends only on it such
    as which rules can conflict, is constant within a cell.
    """
    return [(c["age_lo"], c["jurisdiction"]) for c in context_cells(rules) if c["active"]]


def jointly_satisfiable(phi_a, phi_b):
    """True if some (A, J) in the enumerated domain satisfies both guards.

    Used by the creation-dependency relation: rules whose guards are never
    simultaneously active cannot interact on any host, so they do not create a
    stratification dependency.
    """
    return any(phi_a(A, J) and phi_b(A, J)
               for J in JURISDICTIONS for A in AGE_RANGE)
