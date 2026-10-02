"""Task C (pasted task list, 2026-09-23): re-measure what the contaminated
host file (corpus/holdout_hosts_real_canonical.json, majority RuleParser
fallback per analysis/phase2_task3_report.md) touched, now that Task A
(retry + visible fallback tracking) and Task B (negative-example
instruction) are both live in cpgtr/parse.py.

(a) Canonical conformance rate on the v2 hosts (Task 3b's re-parse, built
    with only the Task 1 max_tokens fix -- no retry/negative-example yet).
(b) Re-run Task 3b's host parse with Task A+B applied. Reports node/edge
    counts against BOTH earlier runs (old truncated-cap, and v2). Writes
    corpus/holdout_hosts_real_canonical_v3.json ONLY if any count differs
    from v2 -- v2 is never overwritten.
(c) Re-reports LIB-FULL / LIB-FAITHFUL-P coverage on the FINAL host file
    (v3 if written, else v2 stands).

Real API calls for (b)'s re-parse; (a) and (c) are pure re-computation.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
from cpgtr.llm import make_complete
from cpgtr.topology import canonical_violations
from cpgtr.graph import Graph
from analysis.phase1a_run import coverage_metrics_v3
from analysis.phase2_harness import build_rules_for_library


def load_hosts(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def host_to_graph(h):
    g = Graph()
    for n in h["nodes"]:
        g.add_node(n["id"], n["type"])
    for e in h["edges"]:
        g.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return g


def as_graphs_with_ctx(hosts):
    return [(h["id"], host_to_graph(h), (h["context"]["A"], h["context"]["J"])) for h in hosts]


# ---------------------------------------------------------------- (a)
old_hosts = load_hosts(REPO_ROOT / "corpus" / "holdout_hosts_real_canonical.json")
v2_hosts = load_hosts(REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v2.json")

print("=== (a) Canonical conformance rate, v2 hosts ===")
v2_conformant = 0
for h in v2_hosts:
    g = host_to_graph(h)
    violations = canonical_violations(g)
    ok = len(violations) == 0
    v2_conformant += ok
    print(f"  {h['id']:10s} conformant={ok}  violations={len(violations)}")
print(f"v2 conformance: {v2_conformant}/{len(v2_hosts)}")
print("(compare: the previously-reported 11/12 -> 12/12 'hub parse' figure "
      "in analysis/phase2_config.md was computed on the CONTAMINATED "
      "(majority-RuleParser-fallback) old canonical file, not a genuine "
      "LLMParser+enforce_topology measurement.)")

# ---------------------------------------------------------------- (b)
print("\n=== (b) Re-parse with Task A + Task B applied ===")
real_hosts = load_hosts(REPO_ROOT / "corpus" / "holdout_hosts_real.json")
old_by_id = {e["id"]: e for e in old_hosts}
v2_by_id = {e["id"]: e for e in v2_hosts}

complete = make_complete()
parser = LLMParser(complete, enforce_topology=True)

v3_hosts = []
any_diff = False
fallback_ids = []
print(f"{'host':10s} {'old N':>6s} {'v2 N':>6s} {'v3 N':>6s} {'old E':>6s} {'v2 E':>6s} {'v3 E':>6s}  fallback")
for h in real_hosts:
    doc = parser.parse(h["source_text"])
    if doc.parse_fallback:
        fallback_ids.append(h["id"])
    nodes = [{"id": nid, "type": t} for nid, t in doc.graph.nodes.items()]
    edges = [{"id": eid, "src": s, "tgt": t, "type": ty} for eid, (s, t, ty) in doc.graph.edges.items()]
    entry = dict(h)
    entry["nodes"] = nodes
    entry["edges"] = edges
    entry["parse_fallback"] = doc.parse_fallback
    entry["parse_attempts"] = doc.parse_attempts
    v3_hosts.append(entry)

    old = old_by_id[h["id"]]
    v2 = v2_by_id[h["id"]]
    diff = (len(nodes), len(edges)) != (len(v2["nodes"]), len(v2["edges"]))
    any_diff = any_diff or diff
    print(f"{h['id']:10s} {len(old['nodes']):6d} {len(v2['nodes']):6d} {len(nodes):6d} "
          f"{len(old['edges']):6d} {len(v2['edges']):6d} {len(edges):6d}  "
          f"{'FALLBACK' if doc.parse_fallback else ('DIFF' if diff else '')}")

print(f"\nany host fell back (even after Task A retries)? {fallback_ids or 'none'}")
print(f"any node/edge count differs from v2? {any_diff}")

if any_diff:
    out_path = REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v3.json"
    out_path.write_text(json.dumps(v3_hosts, indent=2), encoding="utf-8")
    print(f"Written: {out_path} (counts differ from v2)")
    final_hosts = v3_hosts
    final_label = "v3"
else:
    print("No count differs from v2 -- v3 NOT written, v2 stands as final.")
    final_hosts = v2_hosts
    final_label = "v2"

# ---------------------------------------------------------------- (c)
print(f"\n=== (c) LIB-FULL / LIB-FAITHFUL-P coverage, final host file ({final_label}) ===")
final_graphs = as_graphs_with_ctx(final_hosts)
v2_graphs = as_graphs_with_ctx(v2_hosts)
old_graphs = as_graphs_with_ctx(old_hosts)

coverage_report = {}
for lib_id in ("LIB-FULL", "LIB-FAITHFUL-P"):
    rules, rec = build_rules_for_library(lib_id)
    old_cov = coverage_metrics_v3(rules, old_graphs)
    final_cov = coverage_metrics_v3(rules, final_graphs)
    coverage_report[lib_id] = {"old": old_cov, "final": final_cov, "n_rules": len(rules),
                                "sha256": rec["sha256"]}
    print(f"{lib_id} (n={len(rules)}):")
    print(f"  old (contaminated) canonical hosts: host={old_cov['host_coverage_n']} "
          f"rule={old_cov['rule_coverage_n']} struct_all={old_cov['structural_coverage_all_n']}")
    print(f"  final ({final_label}) canonical hosts:        host={final_cov['host_coverage_n']} "
          f"rule={final_cov['rule_coverage_n']} struct_all={final_cov['structural_coverage_all_n']}")

out = {
    "v2_conformance": f"{v2_conformant}/{len(v2_hosts)}",
    "final_label": final_label,
    "fallback_ids": fallback_ids,
    "any_diff_from_v2": any_diff,
    "coverage": {
        lib_id: {
            "old": {k: v for k, v in d["old"].items() if k.endswith("_n")},
            "final": {k: v for k, v in d["final"].items() if k.endswith("_n")},
        } for lib_id, d in coverage_report.items()
    },
}
(REPO_ROOT / "analysis" / "phase2_taskC_output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskC_output.json")
