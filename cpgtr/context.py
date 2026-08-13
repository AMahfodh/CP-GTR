"""Context cells .

Because every rule's guard Phi_r is a Boolean combination of comparisons
A ⋈ c over a finite jurisdiction set, the family {Phi_r} induces a finite
partition of N x J into context cells on which the active set R(A,J) is
constant. Certification must quantify over these cells, not over the
evaluation grid A in {9,15,21} x J in {US,EU,UK} -- the grid only SAMPLES
cells for measurement; the termination/confluence guarantee is discharged
for every (A,J) in N x J.

Rule.phi is an opaque callable rather than a symbolic predicate (both the
hand-written rules in library.py and the LLM-synthesized ones from extract.py
end up as closures), so cells are computed by exact enumeration rather than
symbolic threshold extraction. This is exact -- not an approximation -- within
the enumerated domain: JURISDICTIONS is the fixed, closed set the schema in
extract.py ever emits (US, EU, UK, US-CA; see TG_NODES/TG_EDGES's sibling
enumeration there), and AGE_RANGE comfortably covers every realistic
consent-age threshold in the corpus (13, 16, 18). A guard that references an
age or jurisdiction outside these enumerations would already be rejected by
extract.py's syntactic well-formedness check (CP-GTR_V2.tex sec:synthesis,
item 3), so nothing of significance is being left out.
"""
from __future__ import annotations

JURISDICTIONS = ("US", "EU", "UK", "US-CA")
AGE_RANGE = range(0, 121)


def active_signature(rules, A, J):
    """Which of `rules` are active at (A,J), as a frozenset of identities --
    the "signature" that distinguishes one context cell from its neighbors."""
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
    """One (A,J) witness per non-empty context cell -- checking a property at
    these points is equivalent to checking it for every (A,J) in N x J,
    because the active set (and hence anything that depends only on it, such
    as which rules can conflict) is constant within a cell."""
    return [(c["age_lo"], c["jurisdiction"]) for c in context_cells(rules) if c["active"]]


def jointly_satisfiable(phi_a, phi_b):
    """Does there exist (A,J) in the enumerated domain with both phi_a(A,J)
    and phi_b(A,J) true? Used by the creation-dependency relation
    (CP-GTR_V2.tex Definition "Creation dependency"): two rules whose guards
    can never both be active can never actually interact on any real host, so
    they must not be treated as creating a stratification dependency."""
    return any(phi_a(A, J) and phi_b(A, J)
               for J in JURISDICTIONS for A in AGE_RANGE)
