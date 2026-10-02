"""Exhibit a non-terminating derivation of LIB-FULL in the California cell.

Supports: the post-hoc certification audit, claim that at the
California-specific jurisdiction value three rules applied in turn return to a
graph isomorphic to an earlier one after four steps, so the full library does
not terminate in that context cell.

At context (9, US-CA) it takes the two host graphs of the critical pair between
RequireParentalConsentForChildProtection and
RequireParentalConsentForChildDataProcessing and follows one deterministic
derivation from each (the first applicable rule at every step) among the
library rules active in that cell. It reports the outcome (success, revisit of
an isomorphic graph, or inconclusive at the step cap), the fired rules and the
graph sizes, and also runs the branching normal_forms() search with growing
exploration budgets to show that the exhaustive search does not complete.
Offline, deterministic, no model calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_P1c.py
Order: after verify_P1b.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (printed walk and search results).
"""
import sys
sys.path.insert(0, '.')
from analysis.phase2_harness import build_rules_for_library
from cpgtr.cpa import critical_pairs
from cpgtr.rules import one_step, normal_forms
from cpgtr.graph import Graph, iso

rules, _ = build_rules_for_library('LIB-FULL')
by_name = {r.name: r for r in rules}
r1 = by_name['RequireParentalConsentForChildProtection']
r2 = by_name['RequireParentalConsentForChildDataProcessing']
cps = critical_pairs(r1, r2)
S, H1, H2, pers = cps[0]

A, J = 9, 'US-CA'
active = [x for x in rules if x.phi(A, J)]
print('active rule count at (9, US-CA):', len(active))
print('active rule names:', [r.name for r in active])


def signature(G):
    nsig = tuple(sorted(G.nodes.values()))
    esig = tuple(sorted((G.nodes.get(s), ty, G.nodes.get(t)) for (s, t, ty) in G.edges.values()))
    return (nsig, esig)


def single_path_walk(rules, G0, ctx, max_steps=60):
    seen = {}
    cur = G0
    seen.setdefault(signature(cur), []).append(cur)
    sizes, fired = [], []
    for step in range(1, max_steps + 1):
        st = one_step(rules, cur, ctx)
        if not st:
            return 'success', step - 1, fired, sizes
        r, m, h = st[0]
        fired.append(r.name)
        cur = h
        sizes.append((step, len(cur.nodes), len(cur.edges)))
        sig = signature(cur)
        bucket = seen.setdefault(sig, [])
        if any(iso(cur, prior) for prior in bucket):
            return 'nonterm-isomorphic-revisit', step, fired, sizes
        bucket.append(cur)
    return 'inconclusive-at-cap', max_steps, fired, sizes


for label, host in [('H1', H1), ('H2', H2)]:
    outcome, steps, fired, sizes = single_path_walk(active, host, (A, J), max_steps=60)
    print(f'--- single deterministic path from {label} at (9,US-CA): outcome={outcome} steps={steps}')
    print('   fired:', fired[:15])
    for s in [10, 20, 40]:
        m = [x for x in sizes if x[0] == s]
        print(f'   size at step {s}:', m[0] if m else '(ended earlier)')
    print('   final size:', sizes[-1] if sizes else None)

print()
print('--- normal_forms branching search at (9,US-CA), increasing budgets ---')
for budget in [500, 2000, 5000, 20000]:
    try:
        nfs = normal_forms(active, H1, (A, J), limit=budget)
        print(f'  budget={budget}: completed, {len(nfs)} normal form(s)')
        break
    except RuntimeError:
        print(f'  budget={budget}: RAISED')
