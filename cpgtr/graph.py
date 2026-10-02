"""Typed directed graphs (Semantic Legal Graphs), matching and isomorphism.

Nodes and edges carry only a type. Age and jurisdiction are not stored on
nodes: they live in each rule's context predicate Phi(A, J), so matching here
is purely structural and typed, which keeps it decidable and small.
"""
from __future__ import annotations
import itertools

_fresh = itertools.count()


class Graph:
    """A typed directed multigraph: node id -> type, edge id -> (src, tgt, type)."""

    def __init__(self):
        self.nodes = {}          # id(str) -> type(str)
        self.edges = {}          # id(str) -> (src_id, tgt_id, type)

    def copy(self) -> "Graph":
        """Return a copy that shares no mutable state with this graph."""
        g = Graph()
        g.nodes = dict(self.nodes)
        g.edges = dict(self.edges)
        return g

    def add_node(self, nid, ntype):
        """Add (or retype) node `nid` and return its id."""
        self.nodes[nid] = ntype
        return nid

    def add_edge(self, eid, src, tgt, etype):
        """Add (or replace) edge `eid` from `src` to `tgt` and return its id."""
        self.edges[eid] = (src, tgt, etype)
        return eid

    def incident(self, nid):
        """Ids of all edges with `nid` as source or target."""
        return [e for e, (s, t, _) in self.edges.items() if s == nid or t == nid]

    def __repr__(self):
        ns = ", ".join(f"{i}:{t}" for i, t in self.nodes.items())
        es = ", ".join(f"{s}-{ty}->{t}" for (s, t, ty) in self.edges.values())
        return f"Graph[{ns} | {es}]"


def find_matches(L: Graph, G: Graph):
    """All injective, type-preserving matches of pattern `L` into graph `G`.

    Returns a list of {'nodes': node_map, 'edges': edge_map}, mapping L ids to
    G ids. Distinct L edges map to distinct G edges.
    """
    lnodes = list(L.nodes)
    out = []

    def edges_ok(nm):
        em, used = {}, set()
        for le, (ls, lt, lty) in L.edges.items():
            hit = None
            for ge, (gs, gt, gty) in G.edges.items():
                if ge in used:
                    continue
                if gty == lty and gs == nm[ls] and gt == nm[lt]:
                    hit = ge
                    break
            if hit is None:
                return None
            em[le] = hit
            used.add(hit)
        return em

    def bt(i, nm, used):
        if i == len(lnodes):
            em = edges_ok(nm)
            if em is not None:
                out.append({"nodes": dict(nm), "edges": em})
            return
        ln = lnodes[i]
        lty = L.nodes[ln]
        for gn, gty in G.nodes.items():
            if gty == lty and gn not in used:
                nm[ln] = gn
                used.add(gn)
                bt(i + 1, nm, used)
                used.discard(gn)
                del nm[ln]

    bt(0, {}, set())
    return out


def iso(G1: Graph, G2: Graph, anchor=None):
    """Return a type-preserving isomorphism G1 -> G2 as a node map, or None.

    `anchor` forces given G1 node ids to map to fixed G2 ids; it is used to
    keep persistent (surviving) elements aligned when checking strong
    joinability.
    """
    if len(G1.nodes) != len(G2.nodes) or len(G1.edges) != len(G2.edges):
        return None
    anchor = anchor or {}
    n1 = sorted(G1.nodes, key=lambda x: x not in anchor)  # anchored first

    def edges_ok(nm):
        used = set()
        for _, (s, t, ty) in G1.edges.items():
            hit = None
            for e2, (s2, t2, ty2) in G2.edges.items():
                if e2 in used:
                    continue
                if ty2 == ty and s2 == nm[s] and t2 == nm[t]:
                    hit = e2
                    break
            if hit is None:
                return False
            used.add(hit)
        return True

    def bt(i, nm, used):
        if i == len(n1):
            return dict(nm) if edges_ok(nm) else None
        x = n1[i]
        tx = G1.nodes[x]
        cands = [anchor[x]] if x in anchor else list(G2.nodes)
        for y in cands:
            if y in used or G2.nodes.get(y) != tx:
                continue
            nm[x] = y
            used.add(y)
            r = bt(i + 1, nm, used)
            if r:
                return r
            used.discard(y)
            del nm[x]
        return None

    return bt(0, {}, set())


def new_id(prefix="_"):
    """Return a fresh, process-unique id string starting with `prefix`."""
    return f"{prefix}{next(_fresh)}"