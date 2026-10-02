"""Typed graph-transformation rules and double-pushout (DPO) rewriting.

A `Rule` is a left pattern L, a preserved interface K and a right pattern R,
together with negative application conditions (NACs), a context predicate
phi(A, J), a stratification rank rho and a natural-language template. This
module provides rule application (`applicable`, `apply_rule`), exhaustive and
deterministic rewriting (`one_step`, `normal_forms`, `normalize`), and the NAC
helpers used by certification.

Convention: preserved (interface K) elements share the same id across L, K and
R, and `apply_rule` keeps those ids stable in the result. The track morphism on
preserved elements is therefore the identity on ids, which makes the
strong-joinability comparison (`graph.iso` with an anchor) straightforward.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from .graph import Graph, find_matches, new_id


@dataclass
class Rule:
    """A typed DPO rule with NACs, a context predicate and a repair rank."""
    name: str
    L: Graph
    K_nodes: set
    K_edges: set
    R: Graph                     # preserved elems keep same ids as in L
    nacs: list = field(default_factory=list)   # each: Graph superset of L
    phi: Callable = lambda A, J: True          # context predicate Phi_r(A,J)
    rho: int = None                             # stratification rank
    template: str = ""                          # realization template (text)
    source: str = None                          # source statute of the provision;
                                                 # None for hand-written library rules
    confidence: float = None                    # extractor's self-reported confidence
                                                 # in [0, 1]; certify.admit() orders
                                                 # candidates by it (descending).
                                                 # None if not reported

    def content_hash(self) -> str:
        """Deterministic tie-break key for ordering candidate rules.

        `.name` is not unique across extracted candidates, so it cannot break
        ties between equally confident candidates. This hash covers the rule's
        structural content (L and R reduced to sorted node-type and
        (source type, edge type, target type) signatures, so it ignores
        arbitrary node and edge ids), the sizes of K and of the NAC list, and
        `.name` and `.source`. Structurally distinct rules practically never
        collide, while content-identical rules hash the same.
        """
        import hashlib

        def sig(g):
            node_types = tuple(sorted(g.nodes.values()))
            edge_sig = tuple(sorted(
                (g.nodes.get(s), ty, g.nodes.get(t)) for (s, t, ty) in g.edges.values()
            ))
            return (node_types, edge_sig)

        parts = (
            sig(self.L), sig(self.R),
            len(self.K_nodes), len(self.K_edges), len(self.nacs),
            self.name, self.source,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def del_nodes(self):
        """Ids of L nodes deleted by the rule (in L but not in K)."""
        return [n for n in self.L.nodes if n not in self.K_nodes]

    def del_edges(self):
        """Ids of L edges deleted by the rule."""
        return [e for e in self.L.edges if e not in self.K_edges]

    def add_nodes(self):
        """Ids of R nodes created by the rule (in R but not in K)."""
        return [n for n in self.R.nodes if n not in self.K_nodes]

    def add_edges(self):
        """Ids of R edges created by the rule."""
        return [e for e in self.R.edges if e not in self.K_edges]

    def modality(self):
        """Return "additive", "subtractive" or "substitutive".

        Additive rules only create elements, subtractive rules only delete
        them, and substitutive rules do both (or neither).
        """
        d = bool(self.del_nodes() or self.del_edges())
        a = bool(self.add_nodes() or self.add_edges())
        if a and not d:
            return "additive"
        if d and not a:
            return "subtractive"
        return "substitutive"


def nac_violated(rule: Rule, nac: Graph, node_map, G: Graph) -> bool:
    """True if `node_map` extends to an occurrence of `nac` in `G`.

    The NAC's nodes outside L are searched for injectively across the whole
    host graph; its edges outside L must then exist between the mapped nodes.
    A violated NAC forbids applying the rule at this match.
    """
    extra = [n for n in nac.nodes if n not in rule.L.nodes]

    def bt(i, mm, used):
        if i == len(extra):
            for e, (s, t, ty) in nac.edges.items():
                if e in rule.L.edges:
                    continue
                if s not in mm or t not in mm:
                    return False
                if not any(gty == ty and gs == mm[s] and gt == mm[t]
                           for (gs, gt, gty) in G.edges.values()):
                    return False
            return True
        n = extra[i]
        ty = nac.nodes[n]
        for gn, gty in G.nodes.items():
            if gty == ty and gn not in used:
                mm[n] = gn
                used.add(gn)
                if bt(i + 1, mm, used):
                    return True
                used.discard(gn)
                del mm[n]
        return False

    return bt(0, dict(node_map), set(node_map.values()))


def nac_is_closed(rule: Rule, nac: Graph) -> bool:
    """True if every NAC element beyond rule.L is anchored to L's image.

    An extra node is anchored if it is reachable from an L node through the
    NAC's own edges (treated as undirected), rather than floating disconnected
    from the matched pattern.

    This matters because `nac_violated` searches the whole host graph for the
    NAC's extra elements. If they are disconnected from the match, a later
    embedding that adds a node of the right type anywhere could instantiate the
    NAC and block a step that fired in a smaller graph, invalidating a
    strong-joinability argument. An anchored (closed) NAC can only be completed
    by structure attached to the already-fixed match.

    This is a sound but incomplete approximation of NAC-consistency: it does
    not rule out a later embedding adding a new edge directly onto an anchored
    node. An open NAC is treated as a risk signal, never as a proof of
    soundness.
    """
    extra = {n for n in nac.nodes if n not in rule.L.nodes}
    if not extra:
        return True
    reached = set(rule.L.nodes)
    changed = True
    while changed:
        changed = False
        for (s, t, _ty) in nac.edges.values():
            if s in reached and t not in reached:
                reached.add(t); changed = True
            if t in reached and s not in reached:
                reached.add(s); changed = True
    return extra <= reached


def applicable(rule: Rule, G: Graph, match) -> bool:
    """True if `rule` can be applied at `match` in `G`.

    Requires that no NAC is violated and that the dangling condition holds: no
    edge outside the match may be incident to a node the rule deletes.
    """
    nm, em = match["nodes"], match["edges"]
    for nac in rule.nacs:
        if nac_violated(rule, nac, nm, G):
            return False
    matched = set(em.values())
    for dn in rule.del_nodes():                      # dangling condition
        for ge in G.incident(nm[dn]):
            if ge not in matched:
                return False
    return True


def apply_rule(rule: Rule, G: Graph, match) -> Graph:
    """Apply `rule` at `match` and return the resulting graph; `G` is unchanged.

    Deleted elements are removed, preserved nodes keep their host ids, and
    created elements get fresh ids.
    """
    nm, em = match["nodes"], match["edges"]
    H = G.copy()
    for e in rule.del_edges():
        H.edges.pop(em[e], None)
    for n in rule.del_nodes():
        H.nodes.pop(nm[n], None)
    newid = {}
    for n in rule.add_nodes():
        fid = new_id("_n")
        newid[n] = fid
        H.nodes[fid] = rule.R.nodes[n]

    def resolve(x):
        if x in rule.K_nodes:
            return nm[x]            # preserved: stable id from G
        return newid[x]

    for e in rule.add_edges():
        s, t, ty = rule.R.edges[e]
        H.add_edge(new_id("_e"), resolve(s), resolve(t), ty)
    return H


def one_step(rules, G: Graph, ctx=None):
    """All one-step derivations of `G`, as (rule, match, result) triples.

    If `ctx` = (A, J) is given, only rules whose context predicate holds there
    are considered.
    """
    steps = []
    for r in rules:
        if ctx is not None and not r.phi(*ctx):
            continue
        for m in find_matches(r.L, G):
            if applicable(r, G, m):
                steps.append((r, m, apply_rule(r, G, m)))
    return steps


def normal_forms(rules, G: Graph, ctx=None, limit=5000):
    """Enumerate the reachable normal forms of `G`, up to isomorphism.

    Returns a list of (normal_form, trace) pairs, where `trace` is the list of
    (rule, match) pairs fired from `G` to that normal form. The trace lets
    `cpa.strongly_joinable` check which NACs a derivation relied on.

    Certified rule sets terminate by construction. `limit` bounds the number of
    states explored so that runs with a faulty or disabled termination measure
    raise RuntimeError("non-termination incident") instead of looping.
    """
    from .graph import iso
    results, stack, budget = [], [(G, [])], limit
    while stack:
        cur, trace = stack.pop()
        budget -= 1
        if budget < 0:
            raise RuntimeError("non-termination incident")
        st = one_step(rules, cur, ctx)
        if not st:
            if not any(iso(cur, x) for x, _ in results):
                results.append((cur, trace))
        else:
            for r, m, H in st:
                stack.append((H, trace + [(r, m)]))
    return results


def normalize(rules, G: Graph, ctx=None, limit=5000):
    """Rewrite `G` to a single normal form, taking the first available step.

    The result is independent of step order when the rule set is confluent.
    Raises RuntimeError after `limit` steps (suspected non-termination).
    """
    cur, budget = G, limit
    while True:
        budget -= 1
        if budget < 0:
            raise RuntimeError("non-termination incident")
        st = one_step(rules, cur, ctx)
        if not st:
            return cur
        cur = st[0][2]