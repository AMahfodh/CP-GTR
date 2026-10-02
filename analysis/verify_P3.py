import sys, json
sys.path.insert(0, '.')
from collections import defaultdict
from analysis.phase2_harness import build_rules_for_library
from cpgtr.eval_repair import run_repair

# (a) firing census across all 90 documents per arm
pairs = json.load(open('analysis/phase2_generation_pairs.json', encoding='utf-8'))
fired_a = {'LIB-FULL': defaultdict(int), 'LIB-FAITHFUL-P-v2': defaultdict(int)}
arm_of_lib = {'LIB-FULL': 'CP-GTR-Ungated', 'LIB-FAITHFUL-P-v2': 'CP-GTR'}
for lib, arm in arm_of_lib.items():
    for pid, entry in pairs.items():
        for name in entry.get('_meta', {}).get(f'{arm}_rules_fired', []):
            fired_a[lib][name] += 1

rules_full, _ = build_rules_for_library('LIB-FULL')
rules_faith, _ = build_rules_for_library('LIB-FAITHFUL-P-v2')

# (b) firing census on the 12 Table IX deployment hosts (Full pipeline = LIB-FULL, 31 certified rules)
hosts = json.load(open('corpus/holdout_hosts_real_canonical_v4.json', encoding='utf-8'))
res = run_repair(rules_full, hosts, T_max=10000)
fired_b = defaultdict(int)
# run_repair doesn't record which rule fired per host -- re-derive via one_step walk
from cpgtr.graph import Graph, iso
from cpgtr.rules import one_step


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


any_stepcap_b = False
any_nonterm_b = False
for entry in hosts:
    G0 = host_to_graph(entry)
    ctx = (entry['context']['A'], entry['context']['J'])
    seen = {}
    cur = G0
    seen.setdefault(signature(cur), []).append(cur)
    for step in range(1, 10001):
        st = one_step(rules_full, cur, ctx)
        if not st:
            break
        r, m, h = st[0]
        fired_b[r.name] += 1
        cur = h
        sig = signature(cur)
        bucket = seen.setdefault(sig, [])
        if any(iso(cur, prior) for prior in bucket):
            any_nonterm_b = True
            break
        bucket.append(cur)
    else:
        any_stepcap_b = True

print('=== (a) firing census across all 90 documents (per arm) ===')
for lib in ['LIB-FULL', 'LIB-FAITHFUL-P-v2']:
    rules = rules_full if lib == 'LIB-FULL' else rules_faith
    fired_names = set(fired_a[lib].keys())
    print(f'-- {lib} ({arm_of_lib[lib]} arm), n={len(rules)} rules --')
    for r in rules:
        status = f'FIRED x{fired_a[lib][r.name]}' if r.name in fired_names else 'never fired'
        print(f'   {r.name}: {status}')

print()
print('=== (b) firing census on the 12 Table IX deployment hosts (LIB-FULL, Full pipeline) ===')
for r in rules_full:
    status = f'FIRED x{fired_b[r.name]}' if r.name in fired_b else 'never fired'
    print(f'   {r.name}: {status}')
print('any step-cap (T_max) hit on the 12 hosts:', any_stepcap_b)
print('any confirmed non-termination on the 12 hosts:', any_nonterm_b)
print('(cross-check against run_repair():', res['nonterm_incidents'], 'nonterm,', res['inconclusive'], 'inconclusive)')

print()
print('=== (c) mechanism-validation hosts ===')
print('N/A for every LIB-FULL/LIB-FAITHFUL-P-v2 rule: Table VIII (tab:ablation-a) uses only')
print('library.py hand-written fixtures (r_PC, r_WD, r_SALE, r_RET, r_NOTIFY, r_BAD, r_RET_BAD,')
print('r_FLIP1, r_FLIP2) against corpus/holdout_hosts_mechanism_probe.json -- confirmed by')
print('run_e2e.py: validation_candidates()/table7a() never importing build_rules_for_library.')
