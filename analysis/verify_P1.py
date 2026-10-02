"""P1: resolve the A1/A2 apparent contradiction about LIB-FULL non-termination."""
import sys
sys.path.insert(0, '.')
from analysis.phase2_harness import build_rules_for_library
from cpgtr.cpa import overlaps, critical_pairs
from cpgtr.rules import one_step, normal_forms, applicable
from cpgtr.graph import Graph, iso

rules, _ = build_rules_for_library('LIB-FULL')
by_name = {r.name: r for r in rules}
r1 = by_name['RequireParentalConsentForChildProtection']
r2 = by_name['RequireParentalConsentForChildDataProcessing']

cps = critical_pairs(r1, r2)
print('critical pairs found:', len(cps))
S, H1, H2, pers = cps[0]
print('S nodes:', S.nodes)
print('S edges:', S.edges)

ctx = (9, 'US')  # both rules jointly active here (unconditioned in J, age<13 gate on r1? check)
print('r1.phi(9,US)=', r1.phi(9, 'US'), ' r2.phi(9,US)=', r2.phi(9, 'US'))


def signature(G):
    nsig = tuple(sorted(G.nodes.values()))
    esig = tuple(sorted((G.nodes.get(s), ty, G.nodes.get(t)) for (s, t, ty) in G.edges.values()))
    return (nsig, esig)


def single_path_walk(rules, G0, ctx, max_steps=60):
    """Same policy as eval_repair._run_one_host / rules.normalize: always fire
    the FIRST applicable rule from one_step()'s list, track graph size, detect
    a revisit to an isomorphic earlier state."""
    seen = {}
    cur = G0
    seen.setdefault(signature(cur), []).append(cur)
    sizes = []
    fired = []
    for step in range(1, max_steps + 1):
        st = one_step(rules, cur, ctx)
        if not st:
            return 'success', step - 1, fired, sizes
        r, m, h = st[0]
        fired.append(r.name)
        cur = h
        size = len(cur.nodes) + len(cur.edges)
        sizes.append((step, len(cur.nodes), len(cur.edges)))
        sig = signature(cur)
        bucket = seen.setdefault(sig, [])
        if any(iso(cur, prior) for prior in bucket):
            return 'nonterm-isomorphic-revisit', step, fired, sizes
        bucket.append(cur)
    return 'inconclusive-at-cap', max_steps, fired, sizes


outcome, steps, fired, sizes = single_path_walk(rules, H1, ctx, max_steps=60)
print()
print('=== single-path deterministic walk from H1 (same policy as normalize()/_run_one_host) ===')
print('outcome:', outcome, 'steps:', steps)
print('fired sequence (first 20):', fired[:20])
for s in [10, 20, 40]:
    match = [x for x in sizes if x[0] == s]
    print(f'  size at step {s}:', match[0] if match else '(walk ended before this step)')
print('final size reached:', sizes[-1] if sizes else None)

outcome2, steps2, fired2, sizes2 = single_path_walk(rules, H2, ctx, max_steps=60)
print()
print('=== single-path walk from H2 ===')
print('outcome:', outcome2, 'steps:', steps2, 'fired (first 20):', fired2[:20])

# Now reproduce the branching normal_forms() RuntimeError on the SAME host, and
# see if lowering the budget shows it's a branching-count issue, not per-path growth.
print()
print('=== normal_forms() branching search on H1, various budgets ===')
for budget in [50, 200, 1000, 5000]:
    try:
        nfs = normal_forms(rules, H1, ctx, limit=budget)
        print(f'  budget={budget}: completed, {len(nfs)} normal form(s) found')
    except RuntimeError as e:
        print(f'  budget={budget}: RAISED ({e})')
