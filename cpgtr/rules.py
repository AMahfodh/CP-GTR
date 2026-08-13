"""CP-GTR rules and DPO direct derivation.

Convention: preserved (interface K) elements share the same id across L, K, R.
apply_rule keeps those ids stable in H, so the track morphism on preserved
elements is the identity on ids -- which is what makes strong-joinability
comparison (graph/graph.iso with an anchor) straightforward.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from .graph import Graph, find_matches, new_id


@dataclass
class Rule:
    name: str
    L: Graph
    K_nodes: set
    K_edges: set
    R: Graph                     # preserved elems keep same ids as in L
    nacs: list = field(default_factory=list)   # each: Graph superset of L
    phi: Callable = lambda A, J: True          # context predicate Phi_r(A,J)
    rho: int = None                             # stratification rank
    template: str = ""                          # realization template (text)
    source: str = None                          # provisions.json "source" field,
                                                 # set by extract.py for real candidates;
                                                 # None for hand-written library.py rules
    confidence: float = None                    # extractor-self-reported confidence in
                                                 # [0,1], set by extract.py's json_to_rule()
                                                 # from the LLM's own output; used by
                                                 # certify.admit() to sort R_prop by
                                                 # "descending extractor confidence" per
                                                 # Algorithm 3 (CP-GTR_V2.tex:30) before the
                                                 # admission loop -- without this the loop's
                                                 # per-candidate confluence check (which only
                                                 # compares against already-admitted rules) is
                                                 # order-dependent on real data. None for
                                                 # hand-written library.py rules and any cached
                                                 # spec extracted before this field existed.

    # convenience
    def del_nodes(self):
        return [n for n in self.L.nodes if n not in self.K_nodes]

    def del_edges(self):
        return [e for e in self.L.edges if e not in self.K_edges]

    def add_nodes(self):
        return [n for n in self.R.nodes if n not in self.K_nodes]

    def add_edges(self):
        return [e for e in self.R.edges if e not in self.K_edges]

    def modality(self):
        d = bool(self.del_nodes() or self.del_edges())
        a = bool(self.add_nodes() or self.add_edges())
        if a and not d:
            return "additive"
        if d and not a:
            return "subtractive"
        return "substitutive"


def nac_violated(rule: Rule, nac: Graph, node_map, G: Graph) -> bool:
    """True if match `node_map` can be extended so the forbidden NAC exists in G."""
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
    """True if every 'extra' element of `nac` (the part beyond rule.L) is
    structurally anchored to rule.L's image via nac's own edges -- reachable,
    treating edges as undirected for reachability -- rather than floating
    disconnected from the matched pattern.

    Why this matters (CP-GTR_V2.tex Definition "Extension diagram and
    consistency", the NAC-consistency clause): nac_violated() searches the
    WHOLE host graph for the NAC's extra elements, not just structure
    attached to the match. If those extra elements are disconnected from L's
    image, ANY future embedding that happens to add a node of the right type
    ANYWHERE could retroactively instantiate the NAC and block a step that
    fired successfully in a smaller graph -- which would invalidate a
    strong-joinability argument once the critical pair is embedded in a real
    host. An anchored (closed) NAC can only be completed by structure
    attached to the already-fixed match, which is fully determined within
    the graph at hand.

    This is a SOUND BUT INCOMPLETE approximation of the manuscript's general
    NAC-consistency (it does not rule out a future embedding adding a brand
    new edge directly onto an anchored match node -- see
    docs/manuscript_excerpts.md section 3 / CLAUDE.md for the caveat), used
    conservatively: an "open" (non-closed) NAC is treated as a risk signal,
    never as a soundness proof by itself.
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
    """All one-step derivations. If ctx=(A,J) given, only Phi-active rules fire."""
    steps = []
    for r in rules:
        if ctx is not None and not r.phi(*ctx):
            continue
        for m in find_matches(r.L, G):
            if applicable(r, G, m):
                steps.append((r, m, apply_rule(r, G, m)))
    return steps


def normal_forms(rules, G: Graph, ctx=None, limit=5000):
    """Enumerate all reachable (normal_form, trace) pairs. Raises on suspected
    non-termination.

    `trace` is the list of (rule, match) pairs fired along the path from G to
    that normal form -- needed by cpa.strongly_joinable's NacConsistent check,
    which must inspect which NACs a joining derivation actually relied on, not
    just its endpoint.

    Safe because certified rule sets are terminating by construction; the limit
    is a guard so ABLATION runs (with a bad measure) surface a real incident.
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
    """Deterministic single normal form (any order; unique by confluence)."""
    cur, budget = G, limit
    while True:
        budget -= 1
        if budget < 0:
            raise RuntimeError("non-termination incident")
        st = one_step(rules, cur, ctx)
        if not st:
            return cur
        cur = st[0][2]