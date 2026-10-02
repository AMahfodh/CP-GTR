"""Task M follow-up: isolate WHY Task E's harness (0/10 hallucinated) and
current production (10/10 hallucinated, phase2_taskM_g09_production.py)
disagree. The only functional difference identified between the two
harnesses is PARSE_MAX_TOKENS -- Task E's repeat-trials script ran BEFORE
Task F raised it 4096 -> 8192, so it measured at 4096; Task K and
phase2_taskM_g09_production.py both measure at the current 8192. This
directly A/B tests that one variable, same session, same code otherwise
(temporary monkeypatch of the module global only -- not a permanent code
change, investigative only).
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

complete = make_complete()
N = 5


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


def trial(max_tokens):
    with patched(PARSE_MAX_TOKENS=max_tokens):
        parser = LLMParser(complete, enforce_topology=True)
        out = []
        for i in range(N):
            doc = parser.parse(g09_text)
            h = bool(doc.graph.nodes or doc.graph.edges)
            out.append(h)
            print(f"  max_tokens={max_tokens} trial {i+1}/{N}: "
                  f"{'HALLUCINATED' if h else 'empty (correct)'}")
        return out


print("=== max_tokens=4096 (Task E's actual setting at the time) ===")
r4096 = trial(4096)
print("\n=== max_tokens=8192 (current production, Task F) ===")
r8192 = trial(8192)

print(f"\n4096: {sum(r4096)}/{N} hallucinated")
print(f"8192: {sum(r8192)}/{N} hallucinated")

(REPO_ROOT / "analysis" / "phase2_taskM_maxtokens_ab.json").write_text(
    json.dumps({"4096": r4096, "8192": r8192}, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskM_maxtokens_ab.json")
