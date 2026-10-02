"""Firing census of the two deployed libraries, at content-hash granularity.

Supports: the post-hoc certification audit, claim that every rule removed to
form the certified sub-libraries never fired in any evaluated derivation, and
the count of deployed rules that fired in the evaluation.

Counts which rules fired across the 90 evaluation pairs: LIB-FULL in the
CP-GTR-Ungated arm and LIB-FAITHFUL-P-v2 in the CP-GTR arm. Rules are
identified by content hash and not by name, because several LIB-FULL rules share
a name (Provision1 three times, Provision2 and Provision4 twice each) while
being structurally and contextually distinct. Each recorded firing is
disambiguated by checking which same-named instance's guard is active at the
document's (age, jurisdiction); if more than one is active, all are marked
possibly fired and the event is listed as ambiguous. Offline and deterministic.

Run from the repository root (the script adds '.' to sys.path):
    python analysis/verify_P4b.py
Order: after verify_partA_A4.py; before verify_P4_verify.py, whose removal sets
come from rules marked "never fired" here.

Inputs : analysis/phase2_generation_pairs.json (per-pair rules fired),
         analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (printed census).
"""
import sys, json
sys.path.insert(0, '.')
from collections import defaultdict
from analysis.phase2_harness import build_rules_for_library

pairs = json.load(open('analysis/phase2_generation_pairs.json', encoding='utf-8'))
rules_full, _ = build_rules_for_library('LIB-FULL')
rules_faith, _ = build_rules_for_library('LIB-FAITHFUL-P-v2')

by_name_full = defaultdict(list)
for r in rules_full:
    by_name_full[r.name].append(r)

fired_hash = {'LIB-FULL': set(), 'LIB-FAITHFUL-P-v2': set()}
ambiguous_events = []

for pid, entry in pairs.items():
    A, J = entry['context']['A'], entry['context']['J']
    for arm, lib in [('CP-GTR-Ungated', 'LIB-FULL'), ('CP-GTR', 'LIB-FAITHFUL-P-v2')]:
        for name in entry.get('_meta', {}).get(f'{arm}_rules_fired', []):
            candidates = by_name_full[name] if lib == 'LIB-FULL' else [r for r in rules_faith if r.name == name]
            active = [r for r in candidates if r.phi(A, J)]
            if len(active) == 1:
                fired_hash[lib].add(active[0].content_hash())
            elif len(active) > 1:
                # More than one same-named instance is active at this (A, J), so
                # the firing name alone cannot disambiguate: mark all of them as
                # possibly fired.
                for r in active:
                    fired_hash[lib].add(r.content_hash())
                ambiguous_events.append((pid, arm, name, A, J, len(active)))
            else:
                ambiguous_events.append((pid, arm, name, A, J, 0))

print('ambiguous/zero-match firing events:', len(ambiguous_events))
for e in ambiguous_events[:20]:
    print('  ', e)

print()
for lib, rules in [('LIB-FULL', rules_full), ('LIB-FAITHFUL-P-v2', rules_faith)]:
    print(f'=== {lib}: content-hash-level firing status ===')
    for r in rules:
        status = 'FIRED' if r.content_hash() in fired_hash[lib] else 'never fired'
        print(f'  {r.name} [{r.source}]: {status}')
