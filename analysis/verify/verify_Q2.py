"""Per-context-cell certification check of the certified sub-libraries.

Supports: the post-hoc certification audit claim that, in every context cell of
the certification domain (including the California-specific cell that the
evaluation never samples), both sub-libraries are acyclic under (D1)-(D3) and
every critical pair is strongly joinable using only the rules active in that
cell.

The sub-libraries are LIB-FULL and LIB-FAITHFUL-P-v2, each minus the rules that
never fired in any evaluated derivation (removal sets as content hashes; see
verify_P4_verify.py). The cells come from cpgtr.context.context_cells(), the
partition of (age, jurisdiction) over which admission quantifies, so the check
covers the whole domain and not a sampled grid. For each cell with active rules
it checks (ii) acyclicity of the dependency graph restricted to the active
rules and (iii) strong joinability of every critical pair, again with the
active rules only. The dependency relation here is the type-only (D1)|(D2)|(D3)
relation without the joint-satisfiability gate, because activity in this one
cell already establishes joint activity.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify/verify_Q2.py
Order: after ../verify_P4_verify.py; before verify_R1.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (cells with a cycle or a non-joinable pair are printed).
"""
import sys
sys.path.insert(0, '.')
from cpgtr.context import context_cells, jointly_satisfiable
from cpgtr.certify import edge_types, node_types, _has_cycle, _sccs
from cpgtr.cpa import critical_pairs, strongly_joinable
from analysis.phase2_harness import build_rules_for_library


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


def dep_typeonly(r, rp):
    """(D1)|(D2)|(D3) without the joint-satisfiability gate, for per-cell use.
    Both rules being active at this cell already establishes joint activity.
    Using the type-only relation here, rather than the existential-over-all-
    cells jointly_satisfiable, is what makes this a per-cell check and not a
    restatement of the global one."""
    return d1(r,rp) or d2(r,rp) or d3(r,rp)


LIB_FULL_STAR_REMOVE = {
    'e3fcf2ccba772f3744d94670193af7ce926657d31529d9b1496ed44a274e1770',
    '75193b645534babe0b18cd68698513c3fdc2fba7c29256f968dded80814f0d74',
    'bcdfac426f8fb808082c72afc0cb7952608fde3fde54e1ccaeb7e1df8e0e746b',
    '17d6a892dc40549d745643cf24ff6cb04af002e00ae9c7792916d1f616f21049',
    '3e62a311cc627fa1a3ed097e14fffdef25bd758e6d2b70645b524f4949122af6',
    '47443c1b6dee27afb3933639efbea88acce5cf32fc4d8e5a6d288bc058d9151f',
    '3188e5d1696575695290a59966563be6e9830493254a18f7ee61ddf46986a393',
    '59aece82386be85988c2e75fd9b26a22681bddf22ad0234a5c9c1e3a4b18d58f',
    '9db8f19ab63934788a6bace3665dae54ed661e28a6cc9d6a5639ae271776985e',
    'b1ccfb74badf69583b95870fecb3b3373f59a8736eeab9e7c1b14e8972bab7bf',
    'c3bc3e4948b355647580e25568b4889dbd87384f7e66822ef6af9767f8d4a036',
    'e477c6fddc4b07408eb10410afc84324d517c0e099c6a34910a53edf644d116d',
}
LIB_FAITH_STAR_REMOVE = {
    '3188e5d1696575695290a59966563be6e9830493254a18f7ee61ddf46986a393',
    '17d6a892dc40549d745643cf24ff6cb04af002e00ae9c7792916d1f616f21049',
}

for lib_name, remove_set, star_name in [
    ('LIB-FULL', LIB_FULL_STAR_REMOVE, 'LIB-FULL*'),
    ('LIB-FAITHFUL-P-v2', LIB_FAITH_STAR_REMOVE, 'LIB-FAITHFUL*'),
]:
    rules, _ = build_rules_for_library(lib_name)
    kept = [r for r in rules if r.content_hash() not in remove_set]
    print(f'=== {star_name}: {len(kept)} rules, full-domain per-cell sweep ===')

    cells = context_cells(kept)
    print(f'total cells (all 4 jurisdictions): {len(cells)}')

    any_cycle_cell = []
    any_nonjoinable_cell = []
    cell_checked = 0
    for cell in cells:
        active = cell['active']
        if not active:
            continue
        cell_checked += 1
        # (ii) acyclicity restricted to this cell's active rules
        ids = [id(r) for r in active]
        id2r = {id(r): r for r in active}
        def edge_fn(rid, _active=active, _id2r=id2r):
            r = _id2r[rid]
            return [id(o) for o in _active if o is not r and dep_typeonly(r, o)]
        if _has_cycle(ids, edge_fn):
            sccs = _sccs(ids, edge_fn)
            cyc = [[id2r[rid].name for rid in comp] for comp in sccs
                   if len(comp) > 1 or (len(comp) == 1 and comp[0] in edge_fn(comp[0]))]
            any_cycle_cell.append((cell['jurisdiction'], cell['age_lo'], cell['age_hi'], cyc))
            continue  # the cell already fails on the cycle; skip its CPA check

        # (iii) strong joinability restricted to this cell's active rules
        for r1_ in active:
            for r2_ in active:
                cps = critical_pairs(r1_, r2_)
                for S, H1, H2, pers in cps:
                    try:
                        ok = strongly_joinable(H1, H2, pers, active)
                    except RuntimeError:
                        ok = False
                    if not ok:
                        any_nonjoinable_cell.append(
                            (cell['jurisdiction'], cell['age_lo'], cell['age_hi'], r1_.name, r2_.name))

    print(f'cells with active rules (checked): {cell_checked}')
    print(f'cells where the dependency graph has a cycle: {len(any_cycle_cell)}')
    for c in any_cycle_cell:
        print('   ', c)
    print(f'cells where a critical pair failed strong joinability: {len(any_nonjoinable_cell)}')
    for c in any_nonjoinable_cell[:20]:
        print('   ', c)
    print()
