"""Typed directed Semantic Legal Graphs, matching, and isomorphism.

Attributes (age, jurisdiction) are NOT stored on nodes: per the paper they live
in the context predicate Phi(A, J), so matching here is purely structural/typed.
This is what keeps matching decidable and small.
"""
from __future__ import annotations
import itertools

_fresh = itertools.count()


class Graph:
    def __init__(self):
        self.nodes = {}          # id(str) -> type(str)
        self.edges = {}          # id(str) -> (src_id, tgt_id, type)

    def copy(self) -> "Graph":
        g = Graph()
        g.nodes = dict(self.nodes)
        g.edges = dict(self.edges)
        return g

    def add_node(self, nid, ntype):
        self.nodes[nid] = ntype
        return nid

    def add_edge(self, eid, src, tgt, etype):
        self.edges[eid] = (src, tgt, etype)
        return eid

    def incident(self, nid):
        return [e for e, (s, t, _) in self.edges.items() if s == nid or t == nid]

    def __repr__(self):
        ns = ", ".join(f"{i}:{t}" for i, t in self.nodes.items())
        es = ", ".join(f"{s}-{ty}->{t}" for (s, t, ty) in self.edges.values())
        return f"Graph[{ns} | {es}]"


def find_matches(L: Graph, G: Graph):
    """All injective, type-preserving matches L -> G.

    Returns list of {'nodes': node_map, 'edges': edge_map}.
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
    """Return a type-preserving isomorphism G1->G2 (dict) or None.

    `anchor` forces certain G1 node ids to map to fixed G2 ids -- used to keep
    persistent (surviving) elements aligned for strong joinability.
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
    return f"{prefix}{next(_fresh)}"