import sys
sys.path.insert(0, '.')
from cpgtr.context import jointly_satisfiable
from cpgtr.certify import edge_types, node_types, r1_satisfied, _has_cycle, _sccs
import cpgtr.certify as certify_mod
from cpgtr.cpa import overlaps, critical_pairs
from cpgtr.rules import nac_is_closed
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


def dep(r, rp):
    if not jointly_satisfiable(r.phi, rp.phi): return False
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


def key(r):
    return r.content_hash()


def build_reduced(lib_name, remove_keys):
    rules, _ = build_rules_for_library(lib_name)
    kept = [r for r in rules if key(r) not in remove_keys]
    removed = [r for r in rules if key(r) in remove_keys]
    return kept, removed, rules


for lib_name, remove_keys in [('LIB-FULL', LIB_FULL_STAR_REMOVE), ('LIB-FAITHFUL-P-v2', LIB_FAITH_STAR_REMOVE)]:
    star_name = 'LIB-FULL*' if lib_name == 'LIB-FULL' else 'LIB-FAITHFUL*'
    kept, removed, original = build_reduced(lib_name, remove_keys)
    print(f'=== {star_name}: {len(kept)} rules (removed {len(removed)} from {len(original)}) ===')
    assert len(removed) == len(remove_keys), f'expected to remove {len(remove_keys)}, actually removed {len(removed)}'

    # (i) R1
    r1_fail = [r.name for r in kept if not r1_satisfied(r)]
    print(f'(i) R1 satisfied by all: {not r1_fail}', r1_fail if r1_fail else '')

    # (ii) acyclic under D1+D2+D3, depth d
    ids = [id(r) for r in kept]
    id2r = {id(r): r for r in kept}
    def edge_fn(rid, _kept=kept):
        r = id2r[rid]
        return [id(o) for o in kept if o is not r and dep(r, o)]
    has_cycle = _has_cycle(ids, edge_fn)
    print(f'(ii) acyclic: {not has_cycle}')
    depth = None
    if not has_cycle:
        orig_dep = certify_mod.creation_dependency
        certify_mod.creation_dependency = dep
        try:
            kept2, _, _ = build_reduced(lib_name, remove_keys)
            cyclic_ids, _ = certify_mod.stratify(kept2)
            assert not cyclic_ids
            depth = max((r.rho for r in kept2 if r.rho is not None), default=None)
            print(f'    stratification depth d = {depth}')
        finally:
            certify_mod.creation_dependency = orig_dep

    # (iii) every non-parallel-independent CP within the reduced library is strongly joinable
    from cpgtr.rules import normal_forms
    from cpgtr.graph import iso
    zero_length_count = 0
    nonzero_pairs = []
    for r1_ in kept:
        for r2_ in kept:
            cps = critical_pairs(r1_, r2_)
            for S, H1, H2, pers in cps:
                try:
                    nf1 = normal_forms(kept, H1, limit=5000)
                    nf2 = normal_forms(kept, H2, limit=5000)
                except RuntimeError:
                    nonzero_pairs.append((r1_.name, r2_.name, 'NONTERM'))
                    continue
                joinable, min_len, best_trace_pair = False, None, None
                for X1, t1 in nf1:
                    for X2, t2 in nf2:
                        if all(x in X1.nodes for x in pers) and all(x in X2.nodes for x in pers):
                            if iso(X1, X2, anchor={x: x for x in pers}):
                                joinable = True
                                L = len(t1) + len(t2)
                                if min_len is None or L < min_len:
                                    min_len, best_trace_pair = L, (t1, t2)
                if min_len == 0:
                    zero_length_count += 1
                elif joinable:
                    t1, t2 = best_trace_pair
                    nac_open = any(not nac_is_closed(rr, nac)
                                    for trace in (t1, t2) for rr, _m in trace for nac in rr.nacs)
                    nonzero_pairs.append((r1_.name, r2_.name, f'joinable len={min_len} NAC_open={nac_open}'))
                else:
                    nonzero_pairs.append((r1_.name, r2_.name, 'NOT JOINABLE'))
    print(f'(iii) zero-length-join pairs (no NAC-consistency check needed): {zero_length_count}')
    print(f'      non-zero-length pairs: {len(nonzero_pairs)}')
    for p in nonzero_pairs[:30]:
        print('        ', p)

    # (iv) produce-forbid screen hits within the reduced library, all L-overlap covered?
    screen_hits = 0
    not_l_covered = 0
    for r1_ in kept:
        cn = node_types(r1_.R) - node_types(r1_.L)
        ce = edge_types(r1_.R) - edge_types(r1_.L)
        if not cn and not ce:
            continue
        for r2_ in kept:
            for nac in r2_.nacs:
                en = {nac.nodes[n] for n in nac.nodes if n not in r2_.L.nodes}
                ee = {ty for e,(s,t,ty) in nac.edges.items() if e not in r2_.L.edges}
                if (cn & en) or (ce & ee):
                    screen_hits += 1
                    ov = sum(1 for _ in overlaps(r1_.L, r2_.L))
                    if ov == 0:
                        not_l_covered += 1
    print(f'(iv) produce-forbid screen hits: {screen_hits}, NOT L-overlap-covered: {not_l_covered}')
    print()
