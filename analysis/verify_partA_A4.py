"""Which rules --Strat admits, and why every deployment host then diverges.

Supports: the deployment-realism claim (Table 8) that with the stratified
measure disabled every host diverges, and that each divergence traces to a rule
with neither a deletion nor a self-discharging NAC, which keeps matching on
structure it has already created.

Admits the 200 canonical candidates with critical-pair analysis on and the
stratified measure off (the --Strat configuration), lists the admitted rules
that fail (R1), then repairs each of the twelve final deployment hosts in its
own context, applying the first applicable rule at each step. A host is
reported as diverging if the derivation revisits a graph isomorphic to an
earlier one, as successful if it reaches a normal form, and as inconclusive if
it is still running after 200 steps. The fired-rule sequence is printed for
each host. Offline, deterministic, no model calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_partA_A4.py
Order: after verify_partA_A3.py. The admitted set depends on the admission
code at the checked-out revision.

Inputs : cache/canonical_extraction_cache.jsonl,
         corpus/holdout_hosts_real_canonical_v4.json.
Outputs: none (printed per-host outcomes).
"""
import sys, json
sys.path.insert(0, '.')
from cpgtr.extract import _load_cache, json_to_rule
from cpgtr.admit import admit
from cpgtr.rules import one_step
from cpgtr.graph import Graph, iso
from cpgtr.certify import r1_satisfied

cache = _load_cache('cache/canonical_extraction_cache.jsonl')
candidates = [json_to_rule(e['spec'], source=e['source'], enforce_topology=True) for e in cache.values()]

cert, log = admit(candidates, enable_cpa=True, enable_strat=False)
print("certified under --Strat:", len(cert))
r1_fail_in_cert = [r.name for r in cert if not r1_satisfied(r)]
print("of those, how many FAIL R1 (no self-discharging NAC / no deletion):", len(r1_fail_in_cert))
print(r1_fail_in_cert)

hosts = json.load(open('corpus/holdout_hosts_real_canonical_v4.json', encoding='utf-8'))


def host_to_graph(entry):
    G = Graph()
    for n in entry['nodes']:
        G.add_node(n['id'], n['type'])
    for e in entry['edges']:
        G.add_edge(e['id'], e['src'], e['tgt'], e['type'])
    return G


def signature(G):
    nsig = tuple(sorted(G.nodes.values()))
    esig = tuple(sorted((G.nodes.get(s), ty, G.nodes.get(t)) for (s, t, ty) in G.edges.values()))
    return (nsig, esig)


for entry in hosts:
    G0 = host_to_graph(entry)
    ctx = (entry['context']['A'], entry['context']['J'])
    seen = {}
    cur = G0
    seen.setdefault(signature(cur), []).append(cur)
    fired = []
    outcome = None
    for step in range(1, 200):
        st = one_step(cert, cur, ctx)
        if not st:
            outcome = ('success', step - 1, fired)
            break
        r, m, h = st[0]
        fired.append(r.name)
        cur = h
        sig = signature(cur)
        bucket = seen.setdefault(sig, [])
        if any(iso(cur, prior) for prior in bucket):
            outcome = ('nonterm', step, fired)
            break
        bucket.append(cur)
    else:
        outcome = ('inconclusive-at-200', 200, fired)
    print(entry['id'], ctx, outcome[0], 'steps=', outcome[1], 'fired sequence:', outcome[2])
