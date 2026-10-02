"""List the critical pairs that need a joining derivation.

Supports: the post-hoc certification audit, claim about the critical pairs
whose joinability depends on NAC-consistency.

For every ordered pair of rules in LIB-FULL and LIB-FAITHFUL-P-v2 it
enumerates the critical pairs (cpgtr.cpa.critical_pairs) and prints those that
are not parallel independent, i.e. that require a joining derivation, with
whether each NAC of the two rules is "closed" (every NAC element anchored to
the rule's left-hand side, see cpgtr.rules.nac_is_closed). Offline,
deterministic, no model calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_partA_A3.py
Order: after verify_partA_A2_decide.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (printed list of critical pairs).
"""
import sys
sys.path.insert(0, '.')
from cpgtr.cpa import critical_pairs
from cpgtr.rules import nac_is_closed
from analysis.phase2_harness import build_rules_for_library

for lib in ["LIB-FULL", "LIB-FAITHFUL-P-v2"]:
    rules, _ = build_rules_for_library(lib)
    print(f"=== {lib}: n={len(rules)} ===")
    found = []
    for r1 in rules:
        for r2 in rules:
            cps = critical_pairs(r1, r2)
            for S, H1, H2, pers in cps:
                nac_info = []
                for r in (r1, r2):
                    for nac in r.nacs:
                        nac_info.append((r.name, nac_is_closed(r, nac)))
                found.append((r1.name, r2.name, nac_info))
    print(f"critical pairs needing a joining derivation (non-parallel-independent): {len(found)}")
    if not found:
        print("  NONE.")
    for r1n, r2n, nac_info in found:
        print(f"  {r1n} vs {r2n} | NACs: {nac_info}")
    print()
