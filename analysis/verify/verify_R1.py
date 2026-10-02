"""NAC-consistency check of the joining derivations of the certified sub-libraries.

Supports: the post-hoc certification audit: in the reviewed sub-library both
critical pairs that need a joining derivation satisfy the NAC-implication
criterion, while in the full sub-library the criterion fails in 212 of the 216
NAC checks that its 66 joining derivations require and is unresolved in the
other four. The criterion is sufficient but not necessary, so a failure does not
show non-confluence.

The check follows the paper's definition of the extension diagram and
consistency. It relies on the identity-id convention of cpgtr/rules.py:
preserved elements share one id across L, K and R, apply_rule keeps those ids
stable, and fresh ids are used only for genuinely new elements.

  Peak NAC n: S' -> N. N is S' (as a base graph, with the identity inclusion)
    extended by the peak rule's NAC "extra" elements (the NAC minus rule.L),
    attached through the peak's own match (rule.L -> S'). This is the
    construction overlaps() already uses to build S' from two left-hand
    sides, generalised to a rule and its own NAC.

  Join NAC n': S' -> N'.
    Step 1 (shift along the join step's match): build N_Y, the graph Y the
    join rule fires on, extended by the join rule's NAC extra part, attached
    through the join rule's own match (rule.L -> Y).
    Step 2 (shift back along the peak step): trace the join rule's match from
    Y back to S' by the identity-id convention. An element of Y traces back to
    S' exactly when the same id exists in S', in which case it is untouched or
    preserved by the peak step. If every element of the match traces back,
    N' is S' extended by the join NAC's extra part, attached through the
    traced-back match, as in the peak case. If some element lands on
    something the peak step created (no pre-image in S'), the identity trace
    does not apply and the pair is reported as UNRESOLVED rather than
    guessed at.

  The criterion asks for an injective, type-preserving, edge-preserving map
  h: N -> N' that is the identity on S' (forced, since n and n' are both
  inclusions of S') and free on N's extra part. The search is brute force over
  all injective assignments; N's extra part has only 1-3 elements per NAC.

The two sub-libraries are LIB-FAITHFUL-P-v2 and LIB-FULL, each minus the rules
that never fired in any evaluated derivation (the removal sets are listed in
the script as content hashes; see ../verify_P4_verify.py).

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify/verify_R1.py
Order: after ../verify_P4_verify.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (per-check statuses and a summary are printed).
"""
import sys, itertools
sys.path.insert(0, '.')
from cpgtr.graph import Graph
from cpgtr.cpa import critical_pairs, overlaps, _match_for
from cpgtr.rules import normal_forms
from cpgtr.graph import iso
from analysis.phase2_harness import build_rules_for_library

_ctr = [0]


def _fresh(prefix):
    _ctr[0] += 1
    return f'{prefix}{_ctr[0]}'


def nac_extra(rule, nac):
    node_extra = {n: nac.nodes[n] for n in nac.nodes if n not in rule.L.nodes}
    edge_extra = {e: nac.edges[e] for e in nac.edges if e not in rule.L.edges}
    return node_extra, edge_extra


def extend_by_nac(base, rule, nac, match):
    """base extended by nac's extra part, attached via `match` (rule.L ->
    base's ids). Returns (N, extra_node_ids, extra_edge_ids). base is left
    unmodified; N is a fresh copy."""
    node_extra, edge_extra = nac_extra(rule, nac)
    N = Graph()
    for nid, ty in base.nodes.items():
        N.add_node(nid, ty)
    for eid, (s, t, ty) in base.edges.items():
        N.add_edge(eid, s, t, ty)
    fresh_ids = {}
    for nid, ty in node_extra.items():
        f = _fresh('X')
        fresh_ids[nid] = f
        N.add_node(f, ty)

    def resolve(x):
        return match['nodes'][x] if x in match['nodes'] else fresh_ids[x]

    extra_edge_ids = []
    for eid, (s, t, ty) in edge_extra.items():
        f = _fresh('XE')
        N.add_edge(f, resolve(s), resolve(t), ty)
        extra_edge_ids.append(f)
    return N, set(fresh_ids.values()), set(extra_edge_ids)


def trace_match_to_S(match, S):
    """Trace a match (into some graph Y produced from S by firing the OTHER
    peak rule) back to S via the identity-id convention. Returns (match_S,
    ok)."""
    m_nodes, m_edges = {}, {}
    ok = True
    for x, yid in match['nodes'].items():
        if yid in S.nodes:
            m_nodes[x] = yid
        else:
            ok = False
    for x, yid in match['edges'].items():
        if yid in S.edges:
            m_edges[x] = yid
        else:
            ok = False
    return {'nodes': m_nodes, 'edges': m_edges}, ok


def find_injective_h(N, extra_n, extra_e, Nprime, extra_np, extra_ep, S):
    """Search for injective h: N -> N' fixing S pointwise, type-preserving,
    edge-preserving. Returns h (dict node_id->node_id) or None."""
    s_ids = set(S.nodes)
    free_N = sorted(extra_n)
    if not free_N:
        # N has no extra nodes beyond S, so h is just the identity on S and is
        # trivially valid.
        return {nid: nid for nid in s_ids}
    # candidate targets in N': S-ids (fixed, but those are taken by identity
    # already) plus N''s own extra nodes, grouped by type for pruning.
    targets_by_type = {}
    for nid in extra_np:
        targets_by_type.setdefault(Nprime.nodes[nid], []).append(nid)
    for nid in s_ids:
        # in principle h could send an N-extra node onto an S-element too,
        # as long as types match and injectivity holds elsewhere
        targets_by_type.setdefault(Nprime.nodes.get(nid), []).append(nid)

    def types_ok(candidates):
        for nid in free_N:
            ty = N.nodes[nid]
            if not any(Nprime.nodes.get(c) == ty for c in candidates):
                return False
        return True

    domains = []
    for nid in free_N:
        ty = N.nodes[nid]
        domains.append([t for t in (extra_np | (s_ids - set())) if Nprime.nodes.get(t) == ty])

    for assignment in itertools.product(*domains):
        if len(set(assignment)) != len(assignment):
            continue  # not injective among themselves
        h = {nid: nid for nid in s_ids}
        h.update(dict(zip(free_N, assignment)))
        if len(set(h.values())) != len(h):
            continue  # not injective overall (collided with an S-id)
        # edge-preservation: every edge of N must map to a real edge of N'
        ok = True
        for eid, (s, t, ty) in N.edges.items():
            hs, ht = h.get(s, s), h.get(t, t)
            if not any(a == hs and b == ht and c == ty for (a, b, c) in Nprime.edges.values()):
                ok = False
                break
        if ok:
            return h
    return None


def check_pair(r1, r2, kept_rules, label, verbose=False):
    results = []
    # Re-derive the matches m1 and m2 (not just H1 and H2) the way
    # critical_pairs() does internally, via overlaps(): critical_pairs()
    # returns only (S, H1, H2, pers), but the peak matches are needed to
    # translate each peak's own NAC to S'.
    for S, o1, o2 in overlaps(r1.L, r2.L):
        m1 = _match_for(r1.L, S, o1)
        m2 = _match_for(r2.L, S, o2)
        if m1 is None or m2 is None:
            continue
        from cpgtr.rules import applicable
        if not applicable(r1, S, m1) or not applicable(r2, S, m2):
            continue
        from cpgtr.cpa import parallel_independent, persistent, apply_rule
        if parallel_independent(r1, m1, r2, m2, S):
            continue
        H1 = apply_rule(r1, S, m1)
        H2 = apply_rule(r2, S, m2)
        pers = persistent(r1, m1, r2, m2, S)

        nf1 = normal_forms(kept_rules, H1, limit=5000)
        nf2 = normal_forms(kept_rules, H2, limit=5000)
        best = None
        for X1, t1 in nf1:
            for X2, t2 in nf2:
                if all(x in X1.nodes for x in pers) and all(x in X2.nodes for x in pers):
                    if iso(X1, X2, anchor={x: x for x in pers}):
                        L = len(t1) + len(t2)
                        if best is None or L < best[0]:
                            best = (L, t1, t2, m1, m2)
        if best is None or best[0] == 0:
            continue
        L, t1, t2, m1_, m2_ = best

        peak_nacs = []  # list of (peak_rule, N, extra_n, extra_e)
        for peak_rule, m in [(r1, m1_), (r2, m2_)]:
            for nac in peak_rule.nacs:
                N, en, ee = extend_by_nac(S, peak_rule, nac, m)
                peak_nacs.append((peak_rule.name, N, en, ee))

        # branch 1: peak is r1 (S->H1 via m1), joining trace t1 fires on H1
        # branch 2: peak is r2 (S->H2 via m2), joining trace t2 fires on H2
        for branch_name, other_peak_rule, other_match, trace, Y0 in [
            ('branch1(peak=r1)', r1, m1_, t1, H1),
            ('branch2(peak=r2)', r2, m2_, t2, H2),
        ]:
            cur = Y0
            for rj, mj in trace:
                for nac in rj.nacs:
                    N_Y, _, _ = extend_by_nac(cur, rj, nac, mj)
                    mj_S, ok = trace_match_to_S(mj, S)
                    entry = {
                        'pair': label, 'branch': branch_name, 'join_rule': rj.name,
                    }
                    if not ok:
                        entry['status'] = 'UNRESOLVED (join match touches a freshly-created element)'
                        results.append(entry)
                        continue
                    Np, en_p, ee_p = extend_by_nac(S, rj, nac, mj_S)
                    entry['Nprime_nodes'] = dict(Np.nodes)
                    entry['Nprime_edges'] = dict(Np.edges)
                    found_h = None
                    found_via = None
                    for peak_name, N, en, ee in peak_nacs:
                        h = find_injective_h(N, en, ee, Np, en_p, ee_p, S)
                        if h is not None:
                            found_h = h
                            found_via = (peak_name, dict(N.nodes), dict(N.edges))
                            break
                    if found_h is not None:
                        entry['status'] = 'PASS'
                        entry['implied_by'] = found_via[0]
                        entry['N_nodes'] = found_via[1]
                        entry['N_edges'] = found_via[2]
                        entry['h'] = found_h
                    else:
                        entry['status'] = 'FAIL'
                        entry['peak_nacs_tried'] = [(pn, dict(N.nodes), dict(N.edges)) for pn, N, _, _ in peak_nacs]
                    results.append(entry)
                # advance cur for potential further steps in the trace
                from cpgtr.rules import apply_rule as _apply
                cur = _apply(rj, cur, mj)
    return results


FAITH_REMOVE = {'3188e5d1696575695290a59966563be6e9830493254a18f7ee61ddf46986a393',
                '17d6a892dc40549d745643cf24ff6cb04af002e00ae9c7792916d1f616f21049'}
rules_f, _ = build_rules_for_library('LIB-FAITHFUL-P-v2')
faith_star = [r for r in rules_f if r.content_hash() not in FAITH_REMOVE]

print("=========== LIB-FAITHFUL* ===========")
all_results = []
for r1 in faith_star:
    for r2 in faith_star:
        res = check_pair(r1, r2, faith_star, f'{r1.name} vs {r2.name}')
        all_results.extend(res)

for e in all_results:
    print(e.get('pair'), '|', e.get('branch'), '| join=', e.get('join_rule'), '| status=', e.get('status'))
    if e.get('status') == 'PASS':
        print('   implied by:', e['implied_by'])
        print('   N  =', e['N_nodes'], e['N_edges'])
        print('   N\' =', e['Nprime_nodes'], e['Nprime_edges'])
        print('   h  =', e['h'])
    elif e.get('status') == 'FAIL':
        print('   N\' =', e['Nprime_nodes'], e['Nprime_edges'])
        print('   peak NACs tried:', e['peak_nacs_tried'])

print()
print("=========== LIB-FULL* ===========")
FULL_REMOVE = {
 'e3fcf2ccba772f3744d94670193af7ce926657d31529d9b1496ed44a274e1770','75193b645534babe0b18cd68698513c3fdc2fba7c29256f968dded80814f0d74',
 'bcdfac426f8fb808082c72afc0cb7952608fde3fde54e1ccaeb7e1df8e0e746b','17d6a892dc40549d745643cf24ff6cb04af002e00ae9c7792916d1f616f21049',
 '3e62a311cc627fa1a3ed097e14fffdef25bd758e6d2b70645b524f4949122af6','47443c1b6dee27afb3933639efbea88acce5cf32fc4d8e5a6d288bc058d9151f',
 '3188e5d1696575695290a59966563be6e9830493254a18f7ee61ddf46986a393','59aece82386be85988c2e75fd9b26a22681bddf22ad0234a5c9c1e3a4b18d58f',
 '9db8f19ab63934788a6bace3665dae54ed661e28a6cc9d6a5639ae271776985e','b1ccfb74badf69583b95870fecb3b3373f59a8736eeab9e7c1b14e8972bab7bf',
 'c3bc3e4948b355647580e25568b4889dbd87384f7e66822ef6af9767f8d4a036','e477c6fddc4b07408eb10410afc84324d517c0e099c6a34910a53edf644d116d',
}
rules_full, _ = build_rules_for_library('LIB-FULL')
full_star = [r for r in rules_full if r.content_hash() not in FULL_REMOVE]

full_results = []
for r1 in full_star:
    for r2 in full_star:
        res = check_pair(r1, r2, full_star, f'{r1.name} vs {r2.name}')
        full_results.extend(res)

n_pass = sum(1 for e in full_results if e.get('status') == 'PASS')
n_fail = sum(1 for e in full_results if e.get('status') == 'FAIL')
n_unresolved = sum(1 for e in full_results if e.get('status', '').startswith('UNRESOLVED'))
print(f'total NAC-consistency checks: {len(full_results)}  PASS={n_pass}  FAIL={n_fail}  UNRESOLVED={n_unresolved}')
print()
print('--- FAILURES ---')
for e in full_results:
    if e.get('status') == 'FAIL':
        print(e['pair'], '|', e['branch'], '| join=', e['join_rule'])
        print('   N\' =', e['Nprime_nodes'], e['Nprime_edges'])
        print('   peak NACs tried:', e['peak_nacs_tried'])
print()
print('--- UNRESOLVED ---')
for e in full_results:
    if e.get('status', '').startswith('UNRESOLVED'):
        print(e['pair'], '|', e['branch'], '| join=', e['join_rule'], '|', e['status'])
