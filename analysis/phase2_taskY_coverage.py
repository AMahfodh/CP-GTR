"""Rule coverage on unseen drafts and on the final deployment hosts.

Supports: the coverage figures of the deployment-realism and reach-of-the-
guarantee sections (host, rule and structural coverage of the reviewed
eight-rule library and the full 31-rule library).

(a) The ten unique Unconstrained drafts (one per prompt, reused across the nine
    contexts) are parsed once each with the canonical LLMParser, giving 90
    (draft, context) coverage evaluations from ten parses. These texts were
    never part of the twelve-host set, so this is the stronger coverage test.
    Libraries: LIB-FAITHFUL-P-v2 and LIB-FULL.
(b) LIB-FULL and LIB-GUARDED coverage on the final twelve deployment hosts
    (corpus/holdout_hosts_real_canonical_v4.json). Offline, no model calls.

Run (needs an LLM provider key for part (a); see cpgtr/llm.py):
    python analysis/phase2_taskY_coverage.py
Order: after phase2_generation_run.py (it reads the generated drafts).

Inputs : analysis/phase2_generation_pairs.json (LLM-generated drafts),
         docs/phase2_prompts_v1.md, corpus/holdout_hosts_real_canonical_v4.json,
         analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl,
         analysis/realization_templates_corrections.json.
Outputs: analysis/phase2_taskY_output.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
from cpgtr.llm import make_complete
from cpgtr.graph import Graph
from cpgtr.admit import CONTEXTS
from analysis.phase1a_run import coverage_metrics_v3
from analysis.phase2_harness import build_rules_for_library, apply_template_corrections
from analysis.phase2_generation_lib import load_prompts, assert_config

cfg = assert_config()
cpgtr_cert, ungated_cert = cfg["cpgtr_cert"], cfg["ungated_cert"]

eval_prompts, _ = load_prompts()
pairs = json.loads((REPO_ROOT / "analysis" / "phase2_generation_pairs.json").read_text(encoding="utf-8"))

# ---- (a) 10 unique Unconstrained drafts, parsed once each ----
print("=== (a) Coverage on the 10 unconstrained drafts, canonical parse, all 9 contexts each ===")
complete = make_complete()
parser = LLMParser(complete, enforce_topology=True, negative_example=False)

draft_graphs = {}   # prompt_id -> Graph
for i in range(1, 11):
    prompt_id = f"P{i}"
    # Reuse the Unconstrained text already generated for this prompt (it is
    # identical across the prompt's nine pairs) rather than re-drafting.
    sample_pair = next(e for pid, e in pairs.items() if e["prompt_id"] == prompt_id)
    text = sample_pair["Unconstrained"]
    doc = parser.parse(text)
    draft_graphs[prompt_id] = doc.graph
    print(f"  {prompt_id}: N={len(doc.graph.nodes)} E={len(doc.graph.edges)}  "
          f"fallback={doc.parse_fallback}  negation_drops={len(doc.negation_drops)}")

coverage_a = {}
for lib_id, cert in (("LIB-FAITHFUL-P-v2", cpgtr_cert), ("LIB-FULL", ungated_cert)):
    graphs_with_ctx = []
    for prompt_id, g in draft_graphs.items():
        for ctx in CONTEXTS:
            graphs_with_ctx.append((f"{prompt_id}_{ctx[0]}_{ctx[1]}", g, ctx))
    assert len(graphs_with_ctx) == 90
    cov = coverage_metrics_v3(cert, graphs_with_ctx)
    coverage_a[lib_id] = cov
    print(f"\n{lib_id} (n={len(cert)}) on 90 (draft, context) pairs:")
    print(f"  host={cov['host_coverage_n']} rule={cov['rule_coverage_n']} "
          f"struct_all={cov['structural_coverage_all_n']} struct_nonef={cov['structural_coverage_nonedgefree_n']}")

# ---- (b) LIB-FULL / LIB-GUARDED on v4 holdout hosts ----
print("\n=== (b) LIB-FULL / LIB-GUARDED coverage on v4 holdout hosts ===")


def host_to_graph(h):
    g = Graph()
    for n in h["nodes"]:
        g.add_node(n["id"], n["type"])
    for e in h["edges"]:
        g.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return g


v4 = json.loads((REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v4.json").read_text(encoding="utf-8"))
v4_graphs = [(h["id"], host_to_graph(h), (h["context"]["A"], h["context"]["J"])) for h in v4]

guarded_cert, guarded_rec = build_rules_for_library("LIB-GUARDED")
apply_template_corrections(guarded_cert)

coverage_b = {}
for lib_id, cert in (("LIB-FULL", ungated_cert), ("LIB-GUARDED", guarded_cert)):
    cov = coverage_metrics_v3(cert, v4_graphs)
    coverage_b[lib_id] = cov
    print(f"{lib_id} (n={len(cert)}) on v4 hosts: host={cov['host_coverage_n']} "
          f"rule={cov['rule_coverage_n']} struct_all={cov['structural_coverage_all_n']} "
          f"struct_nonef={cov['structural_coverage_nonedgefree_n']}")

out = {
    "a_90_drafts": {lib: {k: v for k, v in cov.items() if k.endswith("_n")}
                     for lib, cov in coverage_a.items()},
    "b_v4_hosts": {lib: {k: v for k, v in cov.items() if k.endswith("_n")}
                    for lib, cov in coverage_b.items()},
}
(REPO_ROOT / "analysis" / "phase2_taskY_output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskY_output.json")
