"""Build the final twelve deployment hosts by re-parsing their source text.

Supports: the deployment-realism ablation (Table 8) and the host and rule
coverage figures, which use corpus/holdout_hosts_real_canonical_v4.json as the
final host file.

Re-parses the twelve real holdout hosts from their stored source text with the
canonical model-based parser at the final parse token limit
(PARSE_MAX_TOKENS=8192, with retry and visible fallback tracking), checks
whether any host falls back to the deterministic parser (in particular
real008), and compares node and edge counts with the previous host file
(holdout_hosts_real_canonical_v3.json, which must exist for the comparison).
If any count differs, the new hosts are written to
holdout_hosts_real_canonical_v4.json (the previous file is never overwritten)
and become the final host file; otherwise the previous file stands. It then
reports LIB-FULL and LIB-FAITHFUL-P coverage on the previous and final host
files with phase1a_run.coverage_metrics_v3().

Run (needs an LLM provider key; see cpgtr/llm.py):
    python analysis/phase2_taskF_host_remeasure.py
Order: after phase1a_run.py; before phase2_recompute_tables_4_7_8.py and
phase2_taskY_coverage.py, which consume the v4 host file.

Inputs : corpus/holdout_hosts_real.json (host source text),
         corpus/holdout_hosts_real_canonical_v3.json (comparison only),
         analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: corpus/holdout_hosts_real_canonical_v4.json (LLM-parsed graphs),
         analysis/phase2_taskF_output.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser, PARSE_MAX_TOKENS
from cpgtr.llm import make_complete
from cpgtr.graph import Graph
from analysis.phase1a_run import coverage_metrics_v3
from analysis.phase2_harness import build_rules_for_library

print(f"PARSE_MAX_TOKENS = {PARSE_MAX_TOKENS}")


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


real_hosts = load_hosts(REPO_ROOT / "corpus" / "holdout_hosts_real.json")
v3_hosts = load_hosts(REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v3.json")
v3_by_id = {e["id"]: e for e in v3_hosts}

complete = make_complete()
parser = LLMParser(complete, enforce_topology=True)

v4_hosts = []
any_diff = False
fallback_ids = []
print(f"{'host':10s} {'v3 N':>6s} {'v4 N':>6s} {'v3 E':>6s} {'v4 E':>6s}  note")
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
    v4_hosts.append(entry)

    v3 = v3_by_id[h["id"]]
    diff = (len(nodes), len(edges)) != (len(v3["nodes"]), len(v3["edges"]))
    any_diff = any_diff or diff
    note = "FALLBACK" if doc.parse_fallback else ("DIFF" if diff else "")
    print(f"{h['id']:10s} {len(v3['nodes']):6d} {len(nodes):6d} {len(v3['edges']):6d} {len(edges):6d}  {note}")

print(f"\nreal008 fallback cleared? {'real008' not in fallback_ids}")
print(f"any host still falls back? {fallback_ids or 'none'}")
print(f"any node/edge count differs from v3? {any_diff}")

# The previous host file is never overwritten: the new parse is written to a
# new file only when it differs from it.
if any_diff:
    out_path = REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v4.json"
    out_path.write_text(json.dumps(v4_hosts, indent=2), encoding="utf-8")
    print(f"Written: {out_path} (counts differ from v3)")
    final_hosts, final_label = v4_hosts, "v4"
else:
    print("No count differs from v3 -- v4 NOT written, v3 stands as final.")
    final_hosts, final_label = v3_hosts, "v3"

print(f"\n=== LIB-FULL / LIB-FAITHFUL-P coverage, final host file ({final_label}) ===")
final_graphs = as_graphs_with_ctx(final_hosts)
v3_graphs = as_graphs_with_ctx(v3_hosts)

coverage_report = {}
for lib_id in ("LIB-FULL", "LIB-FAITHFUL-P"):
    rules, rec = build_rules_for_library(lib_id)
    v3_cov = coverage_metrics_v3(rules, v3_graphs)
    final_cov = coverage_metrics_v3(rules, final_graphs)
    coverage_report[lib_id] = {"v3": v3_cov, "final": final_cov}
    print(f"{lib_id} (n={len(rules)}):")
    print(f"  v3:    host={v3_cov['host_coverage_n']} rule={v3_cov['rule_coverage_n']} "
          f"struct_all={v3_cov['structural_coverage_all_n']}")
    print(f"  {final_label} (final): host={final_cov['host_coverage_n']} rule={final_cov['rule_coverage_n']} "
          f"struct_all={final_cov['structural_coverage_all_n']}")

out = {
    "parse_max_tokens": PARSE_MAX_TOKENS,
    "real008_fallback_cleared": "real008" not in fallback_ids,
    "fallback_ids": fallback_ids,
    "any_diff_from_v3": any_diff,
    "final_label": final_label,
    "coverage": {
        lib_id: {
            "v3": {k: v for k, v in d["v3"].items() if k.endswith("_n")},
            "final": {k: v for k, v in d["final"].items() if k.endswith("_n")},
        } for lib_id, d in coverage_report.items()
    },
}
(REPO_ROOT / "analysis" / "phase2_taskF_output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskF_output.json")
