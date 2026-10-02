"""Task M (pasted task list, 2026-09-23), BLOCKING: resolve the apparent
contradiction between Task E (0/10 g09 hallucinations without the negative
instruction) and Task K (production config, 12-entry precision 0.889 vs
11-entry 0.985 with ZERO variance over 5 runs -- implying g09 produced a
false positive in all 5 production runs).

Runs g09's text through the PRODUCTION path exactly as Task K did: plain
`LLMParser(complete, enforce_topology=True)`, no monkeypatching of any
cpgtr.parse module global. Investigative only -- report, do not act.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser, PARSE_MAX_TOKENS, MAX_PARSE_ATTEMPTS
from cpgtr.llm import make_complete

gold = json.loads((REPO_ROOT / "corpus" / "gold_parses.json").read_text(encoding="utf-8"))
g09_text = [e for e in gold if e["id"] == "g09"][0]["text"]

print(f"PARSE_MAX_TOKENS = {PARSE_MAX_TOKENS}")
print(f"MAX_PARSE_ATTEMPTS = {MAX_PARSE_ATTEMPTS}")
print(f"g09 text: {g09_text!r}")

complete = make_complete()
N = 10
results = []
for i in range(N):
    parser = LLMParser(complete, enforce_topology=True)   # plain production construction
    doc = parser.parse(g09_text)
    nodes = list(doc.graph.nodes.values())
    edges = list(doc.graph.edges.values())
    hallucinated = bool(nodes or edges)
    results.append({"trial": i + 1, "nodes": nodes, "edges": edges,
                     "hallucinated": hallucinated, "parse_fallback": doc.parse_fallback,
                     "parse_attempts": doc.parse_attempts})
    print(f"  trial {i+1}: nodes={nodes} edges={edges} "
          f"{'HALLUCINATED' if hallucinated else 'empty (correct)'} "
          f"(attempts={doc.parse_attempts}, fallback={doc.parse_fallback})")

n_halluc = sum(1 for r in results if r["hallucinated"])
print(f"\n{n_halluc}/{N} trials hallucinated (production path, PARSE_MAX_TOKENS={PARSE_MAX_TOKENS})")

(REPO_ROOT / "analysis" / "phase2_taskM_g09_production.json").write_text(
    json.dumps({"parse_max_tokens": PARSE_MAX_TOKENS, "max_parse_attempts": MAX_PARSE_ATTEMPTS,
                "n_hallucinated": n_halluc, "N": N, "results": results}, indent=2),
    encoding="utf-8")
print("Written: analysis/phase2_taskM_g09_production.json")
