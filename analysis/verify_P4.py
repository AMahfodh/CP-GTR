import sys, json, itertools
sys.path.insert(0, '.')
from cpgtr.context import jointly_satisfiable
from cpgtr.certify import edge_types, node_types, r1_satisfied, _has_cycle, _sccs
from analysis.phase2_harness import build_rules_for_library
from collections import defaultdict


def _deleted_types(r):
    nt = {r.L.nodes[n] for n in r.del_nodes()}
    et = {r.L.edges[e][2] for e in r.del_edges()}
    return nt, et


def d1(r, rp):
    ce = edge_types(r.R) - edge_types(r.L); cn = node_types(r.R) - node_types(r.L)
    return bool((ce & edge_types(rp.L)) or (cn & node_types(rp.L)))


def d2(r, rp):
    dnt, det = _deleted_types(r)
    if not dnt and not det: return False
    for nac in rp.nacs:
        en = {nac.nodes[n] for n in nac.nodes if n not in rp.L.nodes}
        ee = {ty for e,(s,t,ty) in nac.edges.items() if e not in rp.L.edges}
        if (dnt & en) or (det & ee): return True
    return False


def d3(r, rp):
    rpdnt = {rp.L.nodes[n] for n in rp.del_nodes()}
    if not rpdnt: return False
    for e in r.del_edges():
        s,t,ty = r.L.edges[e]
        if r.L.nodes.get(s) in rpdnt or r.L.nodes.get(t) in rpdnt: return True
    return False


def dep(r, rp):
    if not jointly_satisfiable(r.phi, rp.phi): return False
    return d1(r,rp) or d2(r,rp) or d3(r,rp)


def is_dead(r):
    return any((not nac.nodes and not nac.edges) for nac in r.nacs)


# firing census (union of P3 a+b)
pairs = json.load(open('analysis/phase2_generation_pairs.json', encoding='utf-8'))
fired_names = {'LIB-FULL': set(), 'LIB-FAITHFUL-P-v2': set()}
arm_of_lib = {'LIB-FULL': 'CP-GTR-Ungated', 'LIB-FAITHFUL-P-v2': 'CP-GTR'}
for lib, arm in arm_of_lib.items():
    for pid, entry in pairs.items():
        for name in entry.get('_meta', {}).get(f'{arm}_rules_fired', []):
            fired_names[lib].add(name)

hosts = json.load(open('corpus/holdout_hosts_real_canonical_v4.json', encoding='utf-8'))
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


rules_full, _ = build_rules_for_library('LIB-FULL')
for entry in hosts:
    G0 = host_to_graph(entry)
    ctx = (entry['context']['A'], entry['context']['J'])
    cur = G0
    seen = {signature(cur): [cur]}
    for step in range(1, 10001):
        st = one_step(rules_full, cur, ctx)
        if not st:
            break
        r, m, h = st[0]
        fired_names['LIB-FULL'].add(r.name)
        cur = h
        sig = signature(cur)
        bucket = seen.setdefault(sig, [])
        if any(iso(cur, prior) for prior in bucket):
            break
        bucket.append(cur)


def find_constrained_fvs(rules, dep_fn, never_fired, mandatory):
    """Search for an FVS subset of (never_fired) rules, always including
    `mandatory`, that makes the D1+D2+D3 graph acyclic. Tries mandatory-only
    first, then grows by adding never-fired rules one at a time (greedy:
    always add a rule that participates in a remaining cycle) up to the full
    never-fired pool. Not necessarily minimum -- P4 only asks for existence
    within the never-fired constraint."""
    name_to_rule = {r.name: r for r in rules}
    pool = [r for r in rules if r.name in never_fired]
    removal = set(mandatory) & set(r.name for r in rules)

    def acyclic_after_removal(removal_names):
        remaining = [r for r in rules if r.name not in removal_names]
        ids = [id(r) for r in remaining]
        id2r = {id(r): r for r in remaining}

        def edge_fn(rid):
            r = id2r[rid]
            return [id(o) for o in remaining if o is not r and dep_fn(r, o)]
        return not _has_cycle(ids, edge_fn), remaining, id2r, edge_fn

    ok, remaining, id2r, edge_fn = acyclic_after_removal(removal)
    if ok:
        return removal, remaining
    # greedily add never-fired rules that appear in a cycle until acyclic
    changed = True
    while not ok and changed:
        changed = False
        sccs = _sccs([id(r) for r in remaining], edge_fn)
        cyclic_ids = set()
        for comp in sccs:
            if len(comp) > 1 or (len(comp) == 1 and comp[0] in edge_fn(comp[0])):
                cyclic_ids.update(comp)
        cyclic_names = {id2r[rid].name for rid in cyclic_ids}
        addable = cyclic_names & never_fired - removal
        if not addable:
            break  # a fired rule is on the cycle -- cannot proceed
        removal |= {sorted(addable)[0]}  # add one at a time, deterministic
        changed = True
        ok, remaining, id2r, edge_fn = acyclic_after_removal(removal)
    return (removal, remaining) if ok else (None, None)


DEAD_NAC_NAMES = {'RemoveParentalConsentWhenNotTargetedToChildren',
                  'ReplaceIndefiniteRetentionWithBoundedDuration',
                  'RecruitmentAgeBoundRule',
                  'OperatorCollectsPersistentIdentifierInternalUse_NoNotice',
                  'Replace indefinite retention with bounded retention to comply with limitation on restrictions'}

for lib in ['LIB-FULL', 'LIB-FAITHFUL-P-v2']:
    rules, _ = build_rules_for_library(lib)
    all_names = [r.name for r in rules]
    never_fired = set(all_names) - fired_names[lib]
    mandatory = DEAD_NAC_NAMES & set(all_names)
    print(f'=== {lib}: n={len(rules)}, never-fired count={len(never_fired)}, mandatory-dead in lib={mandatory} ===')
    removal, remaining = find_constrained_fvs(rules, dep, never_fired, mandatory)
    if removal is None:
        print('  NO valid constrained FVS found -- a fired rule lies on a cycle that cannot be broken without removing it.')
    else:
        print(f'  FVS FOUND, size={len(removal)}: {sorted(removal)}')
        print(f'  remaining library size: {len(remaining)}')
    print()
