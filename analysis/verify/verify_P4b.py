"""Redo the firing census at content_hash granularity (not name), since
several LIB-FULL rules share a name (Provision1 x3, Provision2 x2, Provision4
x2) but are structurally and contextually distinct rules. Disambiguate each
firing record by checking which same-named instance's phi is actually active
at that document's (A,J)."""
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
                # more than one same-named instance active at this (A,J) -- can't
                # disambiguate from firing name alone; mark all as POSSIBLY fired
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
