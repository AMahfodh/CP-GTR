"""Task 3c (pasted task list, 2026-09-23): re-report LIB-FULL and
LIB-FAITHFUL-P coverage on the Task 3b re-parsed hosts
(corpus/holdout_hosts_real_canonical_v2.json), compared against the coverage
already recorded in analysis/phase2_library.json (computed on the
truncated-cap corpus/holdout_hosts_real_canonical.json). No new API calls --
pure graph-matching over already-parsed hosts.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase1a_run import coverage_metrics_v3
from analysis.phase2_harness import build_rules_for_library
from cpgtr.graph import Graph


def load_hosts_as_graphs(path):
    hosts = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for h in hosts:
        g = Graph()
        for n in h["nodes"]:
            g.add_node(n["id"], n["type"])
        for e in h["edges"]:
            g.add_edge(e["id"], e["src"], e["tgt"], e["type"])
        ctx = (h["context"]["A"], h["context"]["J"])
        out.append((h["id"], g, ctx))
    return out


old_hosts = load_hosts_as_graphs(REPO_ROOT / "corpus" / "holdout_hosts_real_canonical.json")
new_hosts = load_hosts_as_graphs(REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v2.json")

for lib_id in ("LIB-FULL", "LIB-FAITHFUL-P"):
    rules, rec = build_rules_for_library(lib_id)
    print(f"=== {lib_id} (n={len(rules)}, sha256={rec['sha256'][:16]}...) ===")

    old_cov = coverage_metrics_v3(rules, old_hosts)
    new_cov = coverage_metrics_v3(rules, new_hosts)

    print(f"  old (truncated-cap) canonical hosts: host={old_cov['host_coverage_n']} "
          f"rule={old_cov['rule_coverage_n']} struct_all={old_cov['structural_coverage_all_n']} "
          f"struct_nonef={old_cov['structural_coverage_nonedgefree_n']}")
    print(f"  new (fixed max_tokens) canonical hosts: host={new_cov['host_coverage_n']} "
          f"rule={new_cov['rule_coverage_n']} struct_all={new_cov['structural_coverage_all_n']} "
          f"struct_nonef={new_cov['structural_coverage_nonedgefree_n']}")
    print()
