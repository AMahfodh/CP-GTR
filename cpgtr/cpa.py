"""Critical pair analysis: overlaps, conflict detection, strong joinability.

Confluence of a rule set is checked through its critical pairs. `critical_pairs`
enumerates the minimal overlaps of two rules' left-hand sides in which the two
rules conflict, and `strongly_joinable` decides whether the two divergent
results can be rejoined while keeping every persistent element. `cpgtr.certify`
uses both when admitting candidates.

This specializes the standard double-pushout development to typed rules with
NACs, for the small left-hand sides (|L| <= 6) the method targets. It is a
reference implementation, not an industrial critical-pair engine.
"""
from __future__ import annotations
from .graph import Graph, find_matches, iso
from .rules import Rule, applicable, apply_rule, nac_is_closed


def overlaps(L1: Graph, L2: Graph):
    """Enumerate jointly-surjective overlaps of two left-hand sides.

    Each overlap identifies a subset of L2's nodes with type-compatible nodes
    of L1 (injectively); unidentified L2 nodes are added as fresh nodes, and
    edges are merged. Yields (S, o1, o2), where S is the overlap graph and
    o1, o2 map the nodes of L1 and L2 into S.
    """
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
    """Return the match of `L` into `S` with the given node map, or None."""
    for m in find_matches(L, S):
        if m["nodes"] == node_map:
            return m
    return None


def _deleted_nodes(r: Rule, m):
    """Host nodes deleted by rule `r` under match `m`."""
    return {m["nodes"][n] for n in r.del_nodes()}


def _deleted_edges(r: Rule, m):
    """Host edges deleted by rule `r` under match `m`."""
    return {m["edges"][e] for e in r.del_edges()}


def parallel_independent(r1, m1, r2, m2, S) -> bool:
    """True if the two matches do not conflict, so the overlap is not critical.

    The matches conflict if one rule deletes an element the other uses
    (delete-use), or if applying one rule makes the other's NACs violated so it
    is not applicable at its match (produce-forbid).
    """
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
    """Nodes of S preserved by both r1 and r2 (the persistent elements).

    Only nodes are returned, not edges. In this graph model an edge has no
    identity beyond its (endpoints, type) triple, `iso` already requires an
    exact structural match of edges, and the dangling condition forces every
    edge incident to a deleted node to be deleted too. A persistent edge's
    endpoints are therefore persistent nodes, and anchoring on those nodes pins
    down every persistent edge between them.
    """
    dead = _deleted_nodes(r1, m1) | _deleted_nodes(r2, m2)
    return [n for n in S.nodes if n not in dead]


def critical_pairs(r1: Rule, r2: Rule):
    """Critical pairs of rules r1 and r2.

    For each minimal overlap of the two left-hand sides in which both rules are
    applicable and conflict, returns (S, H1, H2, pers): the overlap graph, the
    results of applying r1 and r2 to it, and the persistent nodes.
    """
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
    """Conservative NAC-consistency of a derivation.

    Every rule fired along `trace` (a list of (rule, match) pairs, as returned
    by `rules.normal_forms`) must carry only closed NACs (`rules.nac_is_closed`).
    An open NAC could be satisfied by unrelated structure that a later
    embedding into a larger host adds, retroactively blocking a step that fired
    in the small overlap graph and invalidating the joinability argument. Such
    derivations are rejected rather than trusted. This is sound but incomplete:
    it catches disconnected NACs but not every possible future embedding.
    """
    return all(nac_is_closed(rule, nac)
               for rule, _m in trace for nac in rule.nacs)


def strongly_joinable(H1: Graph, H2: Graph, pers, rules) -> bool:
    """Strong NAC-joinability of a critical pair.

    True if H1 and H2 have a common normal form under `rules`, reached by
    NAC-consistent derivations on both branches, in which every persistent
    node `pers` survives and is identified consistently (isomorphism anchored
    on `pers`). May raise RuntimeError if the normal-form search exceeds its
    step budget.
    """
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