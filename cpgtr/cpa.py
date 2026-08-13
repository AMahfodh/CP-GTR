"""Critical Pair Analysis: overlaps, conflict detection, strong joinability.

This specializes the standard DPO development to typed rules with NACs, for the
small left-hand sides. It is a reference
implementation validated on the running examples, not an industrial CPA engine.
"""
from __future__ import annotations
from .graph import Graph, find_matches, iso
from .rules import Rule, applicable, apply_rule, nac_is_closed


def overlaps(L1: Graph, L2: Graph):
    """Jointly-surjective overlaps: identify type-compatible node subsets of L2
    with nodes of L1. Yields (S, o1, o2)."""
    l2 = list(L2.nodes)
    partials = []

    def bt(i, mp, used):
        if i == len(l2):
            partials.append(dict(mp))
            return
        n = l2[i]
        bt(i + 1, mp, used)                         # leave n un-identified
        for m in L1.nodes:                          # identify n with an L1 node
            if L1.nodes[m] == L2.nodes[n] and m not in used:
                mp[n] = m
                used.add(m)
                bt(i + 1, mp, used)
                used.discard(m)
                del mp[n]

    bt(0, {}, set())

    for mp in partials:
        S = Graph()
        for n, t in L1.nodes.items():
            S.add_node(n, t)
        o2 = {}
        for n in L2.nodes:
            if n in mp:
                o2[n] = mp[n]
            else:
                o2[n] = S.add_node(f"B_{n}", L2.nodes[n])
        for e, (s, t, ty) in L1.edges.items():
            S.add_edge(e, s, t, ty)
        for e, (s, t, ty) in L2.edges.items():
            ss, tt = o2[s], o2[t]
            if not any(a == ss and b == tt and c == ty for (a, b, c) in S.edges.values()):
                S.add_edge(f"B_{e}", ss, tt, ty)
        o1 = {n: n for n in L1.nodes}
        yield S, o1, o2


def _match_for(L, S, node_map):
    for m in find_matches(L, S):
        if m["nodes"] == node_map:
            return m
    return None


def _deleted_nodes(r: Rule, m):
    return {m["nodes"][n] for n in r.del_nodes()}


def _deleted_edges(r: Rule, m):
    return {m["edges"][e] for e in r.del_edges()}


def parallel_independent(r1, m1, r2, m2, S) -> bool:
    shared_n = set(m1["nodes"].values()) & set(m2["nodes"].values())
    if shared_n & (_deleted_nodes(r1, m1) | _deleted_nodes(r2, m2)):
        return False                                          # delete-use (nodes)
    shared_e = set(m1["edges"].values()) & set(m2["edges"].values())
    if shared_e & (_deleted_edges(r1, m1) | _deleted_edges(r2, m2)):
        return False                                          # delete-use (edges)
    H1 = apply_rule(r1, S, m1)                                 # produce-forbid
    if all(v in H1.nodes for v in m2["nodes"].values()) and not applicable(r2, H1, m2):
        return False
    H2 = apply_rule(r2, S, m2)
    if all(v in H2.nodes for v in m1["nodes"].values()) and not applicable(r1, H2, m1):
        return False
    return True


def persistent(r1, m1, r2, m2, S):
    """

    Only nodes are returned, not edges, but this does not weaken the check:
    in this graph model an edge has no identity beyond its (endpoints, type)
    triple, iso()'s edge comparison already requires X1 and X2 to match
    exactly on that structural basis, and the dangling condition forces any
    edge incident to a deleted node to be deleted too -- so a "persistent"
    edge's endpoints are automatically persistent nodes, and anchoring on
    persistent nodes already pins down every persistent edge between them.
    Tracking edges explicitly here would add bookkeeping without changing
    what strongly_joinable can distinguish.
    """
    dead = _deleted_nodes(r1, m1) | _deleted_nodes(r2, m2)
    return [n for n in S.nodes if n not in dead]


def critical_pairs(r1: Rule, r2: Rule):
    """Conflicting minimal overlaps of r1, r2 (parallel-independent ones dropped)."""
    cps = []
    for S, o1, o2 in overlaps(r1.L, r2.L):
        m1, m2 = _match_for(r1.L, S, o1), _match_for(r2.L, S, o2)
        if m1 is None or m2 is None:
            continue
        if not applicable(r1, S, m1) or not applicable(r2, S, m2):
            continue
        if parallel_independent(r1, m1, r2, m2, S):
            continue
        cps.append((S, apply_rule(r1, S, m1), apply_rule(r2, S, m2),
                    persistent(r1, m1, r2, m2, S)))
    return cps


def _nac_consistent(trace) -> bool:
    """Conservative NacConsistent(d) (Algorithm "StronglyJoinable"):
    every rule fired along `trace` (a list of (rule, match) pairs, as returned
    by rules.normal_forms) must carry only "closed" NACs (rules.nac_is_closed).
    An open NAC could in principle be satisfied by unrelated structure a
    future embedding adds, retroactively blocking a step that fired here --
    which would invalidate the joinability argument once this critical pair
    is embedded in a real host, so such a derivation is rejected rather than
    trusted."""
    return all(nac_is_closed(rule, nac)
               for rule, _m in trace for nac in rule.nacs)


def strongly_joinable(H1: Graph, H2: Graph, pers, rules) -> bool:
    """Exists common normal form, reached by NAC-consistent derivations on
    both branches, on which every persistent element survives and is
    identified consistently (strong NAC-joinability, CP-GTR_V2.tex Definition
    "Strong joinability")."""
    from .rules import normal_forms
    nf1 = normal_forms(rules, H1)
    nf2 = normal_forms(rules, H2)
    for X1, trace1 in nf1:
        if not _nac_consistent(trace1):
            continue
        for X2, trace2 in nf2:
            if not _nac_consistent(trace2):
                continue
            if all(x in X1.nodes for x in pers) and all(x in X2.nodes for x in pers):
                if iso(X1, X2, anchor={x: x for x in pers}):
                    return True
    return False