"""Task A + Task B (pasted task list, 2026-09-23): re-measure the gold
subset once more, now with (a) Task A's retry/fallback tracking wired in and
(b) Task B's negative-example instruction baked into enforce_topology=True's
prompt (cpgtr/parse.py: _NEGATIVE_EXAMPLE_FOR_PARSER). Real API calls.

Reports, for BOTH enforce_topology settings: P/R/F1, how many entries needed
any retry, how many still fell back after exhausting retries, and g09
specifically (the reproduced hallucination-on-negative-example finding).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
from cpgtr.eval_parser import score_corpus
from cpgtr.llm import make_complete

gold = json.loads((REPO_ROOT / "corpus" / "gold_parses.json").read_text(encoding="utf-8"))
print(f"gold entries: {len(gold)}")

complete = make_complete()


class _ReplayParser:
    """Wraps an already-parsed {text: Document} cache so score_corpus()'s
    own internal parser.parse(text) call reuses the SAME parse instead of
    issuing a second real API call per gold entry."""
    def __init__(self, cache):
        self.cache = cache

    def parse(self, text):
        return self.cache[text]


def run(enforce_topology):
    parser = LLMParser(complete, enforce_topology=enforce_topology)
    docs = {}      # id -> Document
    cache = {}     # text -> Document
    for entry in gold:
        d = parser.parse(entry["text"])
        docs[entry["id"]] = d
        cache[entry["text"]] = d
    res = score_corpus(_ReplayParser(cache), gold)
    p, r, f1 = res["overall"]["all"]
    needed_retry = [eid for eid, d in docs.items() if d.parse_attempts > 1]
    still_fell_back = [eid for eid, d in docs.items() if d.parse_fallback]
    g09_doc = docs.get("g09")
    g09_nodes = list(g09_doc.graph.nodes.values()) if g09_doc else None
    g09_edges = [(s, t, ty) for (s, t, ty) in g09_doc.graph.edges.values()] if g09_doc else None
    return {
        "precision": p, "recall": r, "f1": f1,
        "needed_retry": needed_retry, "still_fell_back": still_fell_back,
        "per_item": res["per_item"],
        "g09_node_types": g09_nodes, "g09_edges": g09_edges,
    }


print("\n--- enforce_topology=False ---")
off = run(False)
print(f"P={off['precision']:.3f} R={off['recall']:.3f} F1={off['f1']:.3f}")
print(f"needed retry: {off['needed_retry']}")
print(f"still fell back after exhausting retries: {off['still_fell_back']}")

print("\n--- enforce_topology=True (Task B negative-example instruction included) ---")
on = run(True)
print(f"P={on['precision']:.3f} R={on['recall']:.3f} F1={on['f1']:.3f}")
print(f"needed retry: {on['needed_retry']}")
print(f"still fell back after exhausting retries: {on['still_fell_back']}")
print(f"g09 (gold: empty) -> node types: {on['g09_node_types']}  edges: {on['g09_edges']}")

print("\n=== Four-condition table (Task B) ===")
print("old truncated, no-topology:        P=0.762 R=0.813 F1=0.787")
print("old truncated, enforce_topology:   P=0.985 R=0.853 F1=0.914")
print("fixed, no negative instruction:    P=0.889 R=0.853 F1=0.871  (Task 3a, enforce_topology=True, pre-Task-B)")
print(f"fixed, WITH negative instruction:  P={on['precision']:.3f} R={on['recall']:.3f} F1={on['f1']:.3f}")

out = {
    "off": {k: v for k, v in off.items() if k not in ("per_item",)},
    "on": {k: v for k, v in on.items() if k not in ("per_item",)},
    "off_per_item": off["per_item"],
    "on_per_item": on["per_item"],
}
(REPO_ROOT / "analysis" / "phase2_taskAB_gold_remeasure.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskAB_gold_remeasure.json")
