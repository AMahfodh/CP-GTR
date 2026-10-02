"""Post-hoc creation-dependency audit of the two deployed libraries.

Supports: the post-hoc certification audit (cycles under the complete
creation-dependency relation, and the (R1) check).

The admission run used a creation-dependency relation that implemented only
clause (D1). This script re-examines the two libraries the evaluation actually
deploys, LIB-FULL (31 rules) and LIB-FAITHFUL-P-v2 (8 rules), not the
200-candidate pool, under all three clauses:
  (D1) the rule creates an element type that appears in the other rule's
       left-hand side;
  (D2) the rule deletes an element type that the other rule's NAC forbids;
  (D3) the rule deletes an edge incident to a node the other rule deletes;
each conditioned on the two context guards being jointly satisfiable. For each
library it checks that every rule satisfies (R1), builds the dependency graph
over distinct ordered pairs, tests it for cycles and, when acyclic, computes
the stratification depth by temporarily substituting this relation into
cpgtr.certify.stratify() (restored afterwards). It does not modify
cpgtr/certify.py or any library. Offline and deterministic, with no model
calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_partA_A1.py
Order: first script of the post-hoc audit; then verify_partA_A2_screen.py,
verify_partA_A2_decide.py, verify_partA_A3.py, verify_partA_A4.py,
verify_P1b.py, verify_P1c.py, verify_P4b.py and verify_P4_verify.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (results are printed, ending with a JSON summary).
"""
import sys
sys.path.insert(0, '.')
from cpgtr.context import jointly_satisfiable
from cpgtr.certify import edge_types, node_types, r1_satisfied, _has_cycle, _sccs
import cpgtr.certify as certify_mod
from analysis.phase2_harness import build_rules_for_library


def _deleted_types(r):
    node_types_del = {r.L.nodes[n] for n in r.del_nodes()}
    edge_types_del = {r.L.edges[e][2] for e in r.del_edges()}
    return node_types_del, edge_types_del


def d1(r, r_prime):
    created_e = edge_types(r.R) - edge_types(r.L)
    created_n = node_types(r.R) - node_types(r.L)
    return bool((created_e & edge_types(r_prime.L)) or (created_n & node_types(r_prime.L)))


def d2(r, r_prime):
    del_n_types, del_e_types = _deleted_types(r)
    if not del_n_types and not del_e_types:
        return False
    for nac in r_prime.nacs:
        extra_n_types = {nac.nodes[n] for n in nac.nodes if n not in r_prime.L.nodes}
        extra_e_types = {ty for e, (s, t, ty) in nac.edges.items() if e not in r_prime.L.edges}
        if (del_n_types & extra_n_types) or (del_e_types & extra_e_types):
            return True
    return False


def d3(r, r_prime):
    rp_del_node_types = {r_prime.L.nodes[n] for n in r_prime.del_nodes()}
    if not rp_del_node_types:
        return False
    for e in r.del_edges():
        s, t, ty = r.L.edges[e]
        if r.L.nodes.get(s) in rp_del_node_types or r.L.nodes.get(t) in rp_del_node_types:
            return True
    return False


def creation_dependency_full(r, r_prime):
    if not jointly_satisfiable(r.phi, r_prime.phi):
        return False
    return d1(r, r_prime) or d2(r, r_prime) or d3(r, r_prime)


def analyze(lib_name):
    rules, meta = build_rules_for_library(lib_name)
    print(f"=== {lib_name}: n={len(rules)} ===")

    # R1 check
    r1_fail = [r.name for r in rules if not r1_satisfied(r)]
    print(f"R1 satisfied by all rules: {not r1_fail}")
    if r1_fail:
        print(f"  R1 FAILURES: {r1_fail}")

    # Build the D1+D2+D3 dependency graph restricted to this library. Self-pairs
    # are excluded because stratify() and _sccs() evaluate the relation only on
    # ordered pairs of distinct rules.
    edges = {}
    for r in rules:
        deps = []
        for rp in rules:
            if r is rp:
                continue
            if creation_dependency_full(r, rp):
                deps.append(rp.name)
        edges[r.name] = deps

    by_name = {r.name: r for r in rules}
    name_to_id = {r.name: id(r) for r in rules}
    id_to_name = {id(r): r.name for r in rules}

    def edge_fn(rid):
        r = by_name[id_to_name[rid]]
        return [name_to_id[n] for n in edges[r.name]]

    node_ids = [id(r) for r in rules]
    has_cycle = _has_cycle(node_ids, edge_fn)
    print(f"Acyclic under D1+D2+D3 (restricted to this library): {not has_cycle}")

    if has_cycle:
        sccs = _sccs(node_ids, edge_fn)
        cyclic_sccs = [[id_to_name[rid] for rid in comp] for comp in sccs
                       if len(comp) > 1 or (len(comp) == 1 and comp[0] in edge_fn(comp[0]))]
        print(f"  CYCLES FOUND: {cyclic_sccs}")
        return {"lib": lib_name, "n": len(rules), "r1_fail": r1_fail,
                "acyclic": False, "cycles": cyclic_sccs, "depth": None}

    # No cycle: compute the stratification depth with the shipped stratify(),
    # substituting this relation for creation_dependency, so the rank
    # assignment is exactly the library's own and not a reimplementation.
    orig = certify_mod.creation_dependency
    certify_mod.creation_dependency = creation_dependency_full
    try:
        fresh_rules, _ = build_rules_for_library(lib_name)
        cyclic_ids, _peers = certify_mod.stratify(fresh_rules)
        assert not cyclic_ids
        depth = max((r.rho for r in fresh_rules if r.rho is not None), default=None)
        none_count = sum(1 for r in fresh_rules if r.rho is None)
        print(f"Stratification depth d = {depth} (rho=None count: {none_count})")
    finally:
        certify_mod.creation_dependency = orig

    return {"lib": lib_name, "n": len(rules), "r1_fail": r1_fail,
            "acyclic": True, "depth": depth}


results = {}
for lib in ["LIB-FULL", "LIB-FAITHFUL-P-v2"]:
    results[lib] = analyze(lib)
    print()

import json
print(json.dumps(results, indent=2, default=str))
