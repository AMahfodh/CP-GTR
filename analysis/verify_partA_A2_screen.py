"""Produce-forbid screen over the two deployed libraries.

Supports: the post-hoc certification audit, claim that every produce-forbid
pair is already exposed by an overlap of left-hand sides.

A rule r1 can enable a violation of another rule r2's NAC only if r1 creates
an element type that r2's NAC contains beyond its left-hand side. This script
lists every ordered pair (r1, r2), including r1 = r2, in LIB-FULL and
LIB-FAITHFUL-P-v2 for which a created node or edge type matches such an extra
NAC type. The pairs passing this cheap type-level screen are then decided by
verify_partA_A2_decide.py. Offline, deterministic, no model calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_partA_A2_screen.py
Order: after verify_partA_A1.py; before verify_partA_A2_decide.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (printed list of screened pairs).
"""
import sys
sys.path.insert(0, '.')
from cpgtr.certify import edge_types, node_types
from analysis.phase2_harness import build_rules_for_library


def created_types(r):
    cn = node_types(r.R) - node_types(r.L)
    ce = edge_types(r.R) - edge_types(r.L)
    return cn, ce


def nac_extra_types(rule, nac):
    extra_n = {nac.nodes[n] for n in nac.nodes if n not in rule.L.nodes}
    extra_e = {ty for e, (s, t, ty) in nac.edges.items() if e not in rule.L.edges}
    return extra_n, extra_e


def screen(lib_name):
    rules, _ = build_rules_for_library(lib_name)
    print(f"=== {lib_name}: n={len(rules)} pairs (incl. r1=r2): {len(rules)**2} ===")
    hits = []
    for r1 in rules:
        cn, ce = created_types(r1)
        if not cn and not ce:
            continue
        for r2 in rules:
            for nac in r2.nacs:
                en, ee = nac_extra_types(r2, nac)
                if (cn & en) or (ce & ee):
                    hits.append((r1.name, r2.name, sorted(cn & en), sorted(ce & ee)))
    print(f"pairs passing screen: {len(hits)}")
    for h in hits:
        print(' ', h)
    return hits


all_hits = {}
for lib in ["LIB-FULL", "LIB-FAITHFUL-P-v2"]:
    all_hits[lib] = screen(lib)
    print()
