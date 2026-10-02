"""Per-context-cell joinability of one LIB-FULL critical pair.

Supports: the post-hoc certification audit, claim that LIB-FULL does not
terminate in the California-specific context cell, which the evaluation never
samples.

Takes the critical pair between RequireParentalConsentForChildProtection and
RequireParentalConsentForChildDataProcessing and tests whether it is strongly
joinable separately in each (age, jurisdiction) cell where both rules are
active, using only the library rules active in that cell. This mirrors the
critical-pair loop of admission, which is context-scoped. The cell
(9, US-CA) is the one in which the joinability search does not finish within
its exploration cap. Offline, deterministic, no model calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_P1b.py
Order: after verify_partA_A1.py; before verify_P1c.py, which exhibits the
concrete cycle in that cell.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (printed per-cell results).
"""
import sys
sys.path.insert(0, '.')
from analysis.phase2_harness import build_rules_for_library
from cpgtr.cpa import critical_pairs, strongly_joinable
from cpgtr.context import JURISDICTIONS

rules, _ = build_rules_for_library('LIB-FULL')
by_name = {r.name: r for r in rules}
r1 = by_name['RequireParentalConsentForChildProtection']
r2 = by_name['RequireParentalConsentForChildDataProcessing']

cps = critical_pairs(r1, r2)
S, H1, H2, pers = cps[0]

# Mirror admit()'s critical-pair loop: for each context cell (A, J) where the
# rule is active, the active set is the already-admitted rules that are also
# active at (A, J), plus the rule itself. The admitted set is approximated by
# the full LIB-FULL set, an over-approximation of what had been admitted at
# that point of the run, but properly restricted to the cell (A, J).
for A in [0, 9, 13, 15, 18, 21]:
    for J in JURISDICTIONS:
        if not (r1.phi(A, J) and r2.phi(A, J)):
            continue
        active = [x for x in rules if x.phi(A, J)]
        try:
            ok = strongly_joinable(H1, H2, pers, active)
            print(f'(A={A},J={J}): active_count={len(active)}  strongly_joinable={ok}')
        except RuntimeError as e:
            print(f'(A={A},J={J}): active_count={len(active)}  RAISED({e})')
