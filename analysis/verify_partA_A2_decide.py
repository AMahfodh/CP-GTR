"""Decide whether screened produce-forbid pairs add a new critical pair.

Supports: the post-hoc certification audit, claim that every pair the
produce-forbid construction would add is already exposed by an overlap of
left-hand sides.

For each pair that passes the type-level screen of verify_partA_A2_screen.py
(listed in pairs_full, one entry per distinct rule pattern), this uses the
shipped overlap and joinability machinery of cpgtr.cpa to check whether the
relevant overlap is already reachable as a plain overlap of the two left-hand
sides (which would put a shared match on the table already), or would need a
new overlap of the first rule's right-hand side with the second rule's NAC that
no left-hand-side overlap provides. It prints, per pair, the number of
left-hand-side overlaps, the critical pairs found by critical_pairs() and
whether each is strongly joinable in the library. Offline, deterministic, no
model calls.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_partA_A2_decide.py
Order: after verify_partA_A2_screen.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (printed per-pair results).
"""
import sys
sys.path.insert(0, '.')
from cpgtr.cpa import overlaps, critical_pairs, strongly_joinable
from analysis.phase2_harness import build_rules_for_library

pairs_full = [
    ('RequireParentalConsentForChildProtection', 'RequireParentalConsentForChildProtection'),
    ('RequireParentalConsentForChildProtection', 'RequireParentalConsentForChildDataProcessing'),
    ('RequireParentalConsentForChildProtection', 'AddParentalConsentRequirementForChildAdoption'),
    ('RequireParentalConsentForChildProtection', 'AddParentalConsentForChildConsentProcess'),
    ('RequireParentalConsentForChildProtection', 'RequireParentalConsentForChildren'),
    ('RequireParentalConsentForChildProtection', 'Under13ParentalConsentRequirement'),
    ('RequireParentalConsentForChildProtection', 'DeleteInfoWhenNoParentalConsent'),
    ('RequireParentalConsentForChildren', 'RequireParentalConsentForChildren'),
    ('RequireParentalConsentForChildren', 'Under13ParentalConsentRequirement'),
    ('RequireParentalConsentForChildren', 'DeleteInfoWhenNoParentalConsent'),
    ('InterpreterAssistanceRequirement', 'InterpreterAssistanceRequirement'),
    ('AddWithdrawalPermissionToConsentProcess', 'AddWithdrawalPermissionToConsentProcess'),
]

for lib in ["LIB-FULL", "LIB-FAITHFUL-P-v2"]:
    print(f"=== {lib} ===")
    rules, _ = build_rules_for_library(lib)
    by_name = {r.name: r for r in rules}
    for n1, n2 in pairs_full:
        if n1 not in by_name or n2 not in by_name:
            continue
        r1, r2 = by_name[n1], by_name[n2]
        # (a) How many overlaps of the two left-hand sides exist?
        ov_count = sum(1 for _ in overlaps(r1.L, r2.L))
        # (b) Which critical pairs does the shipped critical_pairs() find?
        cps = critical_pairs(r1, r2)
        joinable_results = []
        for S, H1, H2, pers in cps:
            try:
                ok = strongly_joinable(H1, H2, pers, rules)
                joinable_results.append(ok)
            except RuntimeError as e:
                joinable_results.append(f"EXPLORATION-NONTERM({e})")
        print(f"  {n1} -> {n2}: L-overlaps={ov_count}, critical_pairs found={len(cps)}, "
              f"strongly_joinable={joinable_results}")
    print()
