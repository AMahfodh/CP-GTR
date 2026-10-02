"""Task Q(c): g09 through the production path 10 times under each condition
(negative_example=False / True). No monkeypatching -- the real constructor
parameter.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser, PARSE_MAX_TOKENS
from cpgtr.llm import make_complete

gold = json.loads((REPO_ROOT / "corpus" / "gold_parses.json").read_text(encoding="utf-8"))
g09_text = [e for e in gold if e["id"] == "g09"][0]["text"]
print(f"PARSE_MAX_TOKENS = {PARSE_MAX_TOKENS}")
print(f"g09 text: {g09_text!r}")

complete = make_complete()
N = 10


def run(label, negative_example):
    results = []
    for i in range(N):
        parser = LLMParser(complete, enforce_topology=True, negative_example=negative_example)
        doc = parser.parse(g09_text)
        h = bool(doc.graph.nodes or doc.graph.edges)
        results.append(h)
        print(f"  [{label}] trial {i+1}/{N}: {'HALLUCINATED' if h else 'empty (correct)'}")
    n_h = sum(results)
    print(f"  [{label}] {n_h}/{N} hallucinated")
    return results


print("\n=== Condition A: without negative_example ===")
a = run("A", negative_example=False)
print("\n=== Condition B: WITH negative_example ===")
b = run("B", negative_example=True)

(REPO_ROOT / "analysis" / "phase2_taskQ_c_g09.json").write_text(
    json.dumps({"A": a, "B": b, "A_hallucinated": sum(a), "B_hallucinated": sum(b), "N": N}, indent=2),
    encoding="utf-8")
print("\nWritten: analysis/phase2_taskQ_c_g09.json")
