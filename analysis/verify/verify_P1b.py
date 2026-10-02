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

# Reproduce EXACTLY what admit()'s internal CPA loop does: for each context
# cell (A,J) where r (the later-processed candidate) is active, active = rules
# in cert (already admitted) that are ALSO active at (A,J), plus r.
# Approximate cert as the full LIB-FULL set (an over-approximation of whatever
# was actually admitted at that point in the real run -- still properly
# restricted to the (A,J) that matters, unlike my earlier full-unfiltered test).
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
