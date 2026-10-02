"""Smatch-style scorer for NL -> Semantic Legal Graph parsers.

Role in the pipeline: measures how well a parser (`RuleParser` or `LLMParser`)
recovers the hand-labelled gold graphs in ``corpus/gold_parses.json``.

Method: each graph is decomposed into instance triples (a node and its type)
and relation triples (source, label, target). Node ids are arbitrary, so the
scorer searches for the node alignment between predicted and gold graphs that
maximizes the number of matched triples. The search is restricted to
same-type node pairs, which is correct for instance matching and shrinks the
search space, and uses hill-climbing with random restarts, the standard Smatch
procedure for small graphs. Precision, recall and F1 are micro-averaged over
the corpus, overall and per node type and edge label.

Inputs: a parser object with ``parse(text) -> Document`` and a list of gold
entries ``{"id", "text", "nodes": [type, ...], "edges": [[src, label, tgt], ...]}``
(gold edges refer to node positions). Output: the metrics dict returned by
`score_corpus`.
"""
from __future__ import annotations
import random
from collections import defaultdict


# ---------------------------------------------------------------- extraction
def graph_triples(G):
    """Return (node_types: dict id->type, edges: list[(src, tgt, label)])."""
    node_types = dict(G.nodes)
    edges = []
    for eid, e in G.edges.items():
        if isinstance(e, (tuple, list)):
            s, t, lab = e[0], e[1], e[2]
        else:                                   # object with attributes
            s, t = e.src, e.tgt
            lab = getattr(e, "label", getattr(e, "type", None))
        edges.append((s, t, lab))
    return node_types, edges


def gold_triples(entry):
    """Return (node_types, edges) for a gold entry, in the same form as
    `graph_triples`. Gold node ids are their positions in ``entry["nodes"]``."""
    node_types = {i: t for i, t in enumerate(entry["nodes"])}
    edges = [(s, t, lab) for (s, lab, t) in entry["edges"]]   # [s,label,t] -> (s,t,label)
    return node_types, edges


# ---------------------------------------------------------------- alignment
def _total(mapping, pred_edges, gold_edge_set):
    inst = sum(1 for v in mapping.values() if v is not None)   # same-type => match
    rel = 0
    for (s, t, lab) in pred_edges:
        ms, mt = mapping.get(s), mapping.get(t)
        if ms is not None and mt is not None and (ms, mt, lab) in gold_edge_set:
            rel += 1
    return inst + rel


def best_alignment(pred_nodes, pred_edges, gold_nodes, gold_edges,
                   restarts=10, seed=0):
    """Find the node alignment that maximizes matched triples.

    Args:
        pred_nodes, gold_nodes: dicts id -> node type.
        pred_edges, gold_edges: lists of (src, tgt, label).
        restarts: number of random restarts of the hill-climb.
        seed: RNG seed (the search is deterministic for a fixed seed).

    Each restart starts from a random injective same-type mapping and
    repeatedly moves or swaps one node's assignment while the total number of
    matched instance + relation triples improves. Returns
    ``(mapping, best_total_triples)`` where `mapping` sends each predicted node
    id to a gold node id, or None if unmatched."""
    rng = random.Random(seed)
    gold_edge_set = set(gold_edges)
    # candidate gold ids per pred node (same type only)
    by_type = defaultdict(list)
    for gid, gt in gold_nodes.items():
        by_type[gt].append(gid)
    cands = {pid: list(by_type[pt]) for pid, pt in pred_nodes.items()}

    best_map, best_score = {p: None for p in pred_nodes}, 0
    for r in range(restarts):
        # random injective init
        mapping = {p: None for p in pred_nodes}
        used = set()
        for p in rng.sample(list(pred_nodes), len(pred_nodes)):
            free = [g for g in cands[p] if g not in used]
            if free:
                g = rng.choice(free)
                mapping[p] = g
                used.add(g)
        # hill-climb: reassign or swap one node at a time
        improved = True
        while improved:
            improved = False
            cur = _total(mapping, pred_edges, gold_edge_set)
            for p in pred_nodes:
                for g in cands[p] + [None]:
                    if mapping[p] == g:
                        continue
                    old_p = mapping[p]
                    # who currently holds g?
                    holder = next((q for q, v in mapping.items() if v == g), None) \
                        if g is not None else None
                    mapping[p] = g
                    if holder is not None:
                        mapping[holder] = old_p        # swap
                    new = _total(mapping, pred_edges, gold_edge_set)
                    if new > cur:
                        cur, improved = new, True
                        break
                    mapping[p] = old_p                 # revert
                    if holder is not None:
                        mapping[holder] = g
                if improved:
                    break
        score = _total(mapping, pred_edges, gold_edge_set)
        if score > best_score:
            best_score, best_map = score, dict(mapping)
    return best_map, best_score


# ---------------------------------------------------------------- scoring
def _prf(match, pred, gold):
    p = match / pred if pred else (1.0 if match == 0 else 0.0)
    r = match / gold if gold else (1.0 if match == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_corpus(parser, gold, restarts=10):
    """Parse every gold text with `parser` and return aggregated metrics.

    Args:
        parser: object with ``parse(text) -> Document``.
        gold: list of gold entries (see the module docstring).
        restarts: hill-climbing restarts per entry.

    Returns a dict with:
        overall: {"inst", "rel", "all"} -> (precision, recall, F1), micro-averaged.
        per_node / per_edge: the same triple per node type / edge label.
        per_item: list of (id, precision, recall, F1) for each gold entry.
        counts: raw (matched, predicted, gold) triple counts.
        fallback_count / fallback_ids: entries for which an `LLMParser` fell
            back to `RuleParser`. A score with a nonzero fallback count
            partly measures `RuleParser` rather than the parser under test, so
            report it alongside the scores.
    """
    agg = {"inst": [0, 0, 0], "rel": [0, 0, 0], "all": [0, 0, 0]}  # match,pred,gold
    per_node = defaultdict(lambda: [0, 0, 0])
    per_edge = defaultdict(lambda: [0, 0, 0])
    per_item = []
    fallback_ids = []

    for entry in gold:
        gN, gE = gold_triples(entry)
        doc = parser.parse(entry["text"])
        if getattr(doc, "parse_fallback", False):
            fallback_ids.append(entry["id"])
        pN, pE = graph_triples(doc.graph)
        mapping, _ = best_alignment(pN, pE, gN, gE, restarts=restarts)

        # counts
        inst_match = sum(1 for v in mapping.values() if v is not None)
        gold_edge_set = set(gE)
        rel_match = sum(1 for (s, t, l) in pE
                        if mapping.get(s) is not None and mapping.get(t) is not None
                        and (mapping[s], mapping[t], l) in gold_edge_set)

        agg["inst"][0] += inst_match; agg["inst"][1] += len(pN); agg["inst"][2] += len(gN)
        agg["rel"][0] += rel_match;   agg["rel"][1] += len(pE); agg["rel"][2] += len(gE)
        agg["all"][0] += inst_match + rel_match
        agg["all"][1] += len(pN) + len(pE)
        agg["all"][2] += len(gN) + len(gE)

        # per node-type
        for t in pN.values():
            per_node[t][1] += 1
        for t in gN.values():
            per_node[t][2] += 1
        for p, g in mapping.items():
            if g is not None:
                per_node[pN[p]][0] += 1
        # per edge-label
        for (_, _, l) in pE:
            per_edge[l][1] += 1
        for (_, _, l) in gE:
            per_edge[l][2] += 1
        for (s, t, l) in pE:
            if mapping.get(s) is not None and mapping.get(t) is not None \
               and (mapping[s], mapping[t], l) in gold_edge_set:
                per_edge[l][0] += 1

        per_item.append((entry["id"], *_prf(inst_match + rel_match,
                                            len(pN) + len(pE), len(gN) + len(gE))))

    return {
        "overall": {k: _prf(*v) for k, v in agg.items()},
        "per_node": {t: _prf(*c) for t, c in sorted(per_node.items())},
        "per_edge": {l: _prf(*c) for l, c in sorted(per_edge.items())},
        "per_item": per_item,
        "counts": agg,
        "fallback_count": len(fallback_ids),
        "fallback_ids": fallback_ids,
    }