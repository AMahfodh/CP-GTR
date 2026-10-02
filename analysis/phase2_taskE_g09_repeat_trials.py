"""Follow-up to Task E: the single before/after g09 anecdote (hallucinates
without Task B's negative instruction, empty with it) did NOT reproduce in
Task E's fresh side-by-side run -- g09 came back empty under ALL FOUR
conditions, including "no negative instruction", contradicting the original
Task 3/Task B finding. This means g09's hallucination is not a deterministic
function of the prompt -- it's flaky model behavior (temperature=0 is
"mostly reproducible in practice, not guaranteed" per
run_e2e.py: generate_real_holdout_hosts()'s own documented caveat).

Re-parses g09's text N=10 times under EACH of "no negative instruction" and
"with negative instruction" (same monkeypatch technique as Task E, real
LLMParser.parse() code path) to measure an empirical hallucination RATE
under each, rather than relying on one anecdote in either direction.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cpgtr.parse as parse_mod
from cpgtr.parse import LLMParser
from cpgtr.llm import make_complete

gold = json.loads((REPO_ROOT / "corpus" / "gold_parses.json").read_text(encoding="utf-8"))
g09_text = [e for e in gold if e["id"] == "g09"][0]["text"]
N = 10

complete = make_complete()


@contextmanager
def patched(**kwargs):
    old = {k: getattr(parse_mod, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(parse_mod, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(parse_mod, k, v)


def trial(include_negative):
    with patched(_NEGATIVE_EXAMPLE_FOR_PARSER=(parse_mod._NEGATIVE_EXAMPLE_FOR_PARSER
                                               if include_negative else "")):
        parser = LLMParser(complete, enforce_topology=True)
        results = []
        for i in range(N):
            doc = parser.parse(g09_text)
            n_nodes = len(doc.graph.nodes)
            n_edges = len(doc.graph.edges)
            results.append((n_nodes, n_edges))
            print(f"  trial {i+1}: nodes={n_nodes} edges={n_edges} "
                  f"{'HALLUCINATED' if n_nodes or n_edges else 'empty (correct)'}")
        return results


print(f"=== g09, {N} trials, NO negative instruction ===")
no_neg = trial(include_negative=False)
n_halluc_no_neg = sum(1 for n, e in no_neg if n or e)

print(f"\n=== g09, {N} trials, WITH negative instruction (Task B) ===")
with_neg = trial(include_negative=True)
n_halluc_with_neg = sum(1 for n, e in with_neg if n or e)

print(f"\n=== Summary ===")
print(f"No negative instruction:   {n_halluc_no_neg}/{N} trials hallucinated")
print(f"With negative instruction: {n_halluc_with_neg}/{N} trials hallucinated")

(REPO_ROOT / "analysis" / "phase2_taskE_g09_repeat_trials.json").write_text(
    json.dumps({"no_negative": no_neg, "with_negative": with_neg,
                "n_hallucinated_no_negative": n_halluc_no_neg,
                "n_hallucinated_with_negative": n_halluc_with_neg, "N": N}, indent=2),
    encoding="utf-8")
print("\nWritten: analysis/phase2_taskE_g09_repeat_trials.json")
