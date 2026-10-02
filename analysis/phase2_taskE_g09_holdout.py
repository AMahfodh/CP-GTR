"""Task E (pasted task list, 2026-09-23): the negative-example instruction
(Task B) was written IN RESPONSE to gold entry g09's hallucination, so the
1.000/0.921 headline is measured partly on the item that motivated the
change. Recompute all four gold-subset conditions with g09 EXCLUDED
(11 entries) and report both sets side by side. g09 itself is NOT removed
from corpus/gold_parses.json -- only excluded from this scoring pass.

All four conditions are reproduced using the REAL LLMParser.parse() code
path (retry loop, topology-repair loop included) via temporary monkeypatches
of cpgtr.parse's module-level globals (PARSE_MAX_TOKENS, MAX_PARSE_ATTEMPTS,
_NEGATIVE_EXAMPLE_FOR_PARSER) -- not a reimplementation, and not a permanent
flag added to the shipped class (Task B's "one change, not iterated on"
stays true of the shipped code; this is a one-off diagnostic re-measurement
of what the code already does under each historical setting). Each gold
text is parsed ONCE per condition; the same parsed Document is scored both
ways (12-entry, 11-entry) via score_corpus()'s existing fallback_count
reporting, using a tiny replay parser so no text is parsed twice.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cpgtr.parse as parse_mod
from cpgtr.parse import LLMParser
from cpgtr.eval_parser import score_corpus
from cpgtr.llm import make_complete

gold = json.loads((REPO_ROOT / "corpus" / "gold_parses.json").read_text(encoding="utf-8"))
gold_no_g09 = [e for e in gold if e["id"] != "g09"]
print(f"gold entries: {len(gold)} (with g09), {len(gold_no_g09)} (g09 excluded)")

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


class _ReplayParser:
    def __init__(self, cache):
        self.cache = cache

    def parse(self, text):
        return self.cache[text]


def run_condition(label, enforce_topology, max_tokens, max_attempts, include_negative):
    with patched(PARSE_MAX_TOKENS=max_tokens, MAX_PARSE_ATTEMPTS=max_attempts,
                 _NEGATIVE_EXAMPLE_FOR_PARSER=(parse_mod._NEGATIVE_EXAMPLE_FOR_PARSER
                                               if include_negative else "")):
        parser = LLMParser(complete, enforce_topology=enforce_topology)
        cache = {}
        for entry in gold:
            cache[entry["text"]] = parser.parse(entry["text"])

    res_12 = score_corpus(_ReplayParser(cache), gold)
    res_11 = score_corpus(_ReplayParser(cache), gold_no_g09)
    p12, r12, f12 = res_12["overall"]["all"]
    p11, r11, f11 = res_11["overall"]["all"]
    g09_doc = cache[[e for e in gold if e["id"] == "g09"][0]["text"]]
    print(f"\n--- {label} ---")
    print(f"  12-entry (with g09):    P={p12:.3f} R={r12:.3f} F1={f12:.3f}  fallback={res_12['fallback_count']}/12")
    print(f"  11-entry (g09 excluded): P={p11:.3f} R={r11:.3f} F1={f11:.3f}  fallback={res_11['fallback_count']}/11")
    print(f"  g09 predicted: nodes={list(g09_doc.graph.nodes.values())} edges={list(g09_doc.graph.edges.values())}")
    return {"12": (p12, r12, f12), "11": (p11, r11, f11),
            "fallback_12": res_12["fallback_count"], "fallback_11": res_11["fallback_count"]}


results = {}
results["old_no_topology"] = run_condition(
    "1. Old truncated, no topology (max_tokens=1024, 1 attempt)",
    enforce_topology=False, max_tokens=1024, max_attempts=1, include_negative=False)

results["old_topology"] = run_condition(
    "2. Old truncated, enforce_topology=True, no negative example (max_tokens=1024, 1 attempt)",
    enforce_topology=True, max_tokens=1024, max_attempts=1, include_negative=False)

results["fixed_no_negative"] = run_condition(
    "3. Fixed (Task 1), enforce_topology=True, NO negative instruction (max_tokens=4096, 3 attempts)",
    enforce_topology=True, max_tokens=4096, max_attempts=3, include_negative=False)

results["fixed_with_negative"] = run_condition(
    "4. Fixed (Task 1) + negative instruction (Task B) (max_tokens=4096, 3 attempts)",
    enforce_topology=True, max_tokens=4096, max_attempts=3, include_negative=True)

print("\n=== Side-by-side summary ===")
print(f"{'condition':45s} {'12-entry P/R/F1':25s} {'11-entry (no g09) P/R/F1':25s}")
for key, label in [("old_no_topology", "old truncated, no topology"),
                    ("old_topology", "old truncated, enforce_topology"),
                    ("fixed_no_negative", "fixed, no negative instr."),
                    ("fixed_with_negative", "fixed + negative instr. (Task B)")]:
    r = results[key]
    p12, r12v, f12 = r["12"]
    p11, r11v, f11 = r["11"]
    print(f"{label:45s} {p12:.3f}/{r12v:.3f}/{f12:.3f}          {p11:.3f}/{r11v:.3f}/{f11:.3f}")

(REPO_ROOT / "analysis" / "phase2_taskE_g09_holdout.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskE_g09_holdout.json")
