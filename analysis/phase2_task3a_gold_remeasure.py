"""Task 3a (pasted task list, 2026-09-23): re-measure the gold-subset parser
P/R/F1 for LLMParser, both with and without enforce_topology, now that
PARSE_MAX_TOKENS=4096 (commit 6f6fdf0) fixes the truncation that was active
when analysis/phase2_gate.md's Task 1 numbers (0.762/0.813/0.787 without,
0.985/0.853/0.914 with) were measured. Real API calls (CPGTR_MODE=real).
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

parser_off = LLMParser(complete, enforce_topology=False)
res_off = score_corpus(parser_off, gold)
p_off, r_off, f1_off = res_off["overall"]["all"]
print(f"BEFORE (enforce_topology=False): P={p_off:.3f} R={r_off:.3f} F1={f1_off:.3f}")

parser_on = LLMParser(complete, enforce_topology=True)
res_on = score_corpus(parser_on, gold)
p_on, r_on, f1_on = res_on["overall"]["all"]
print(f"AFTER  (enforce_topology=True):  P={p_on:.3f} R={r_on:.3f} F1={f1_on:.3f}")

print()
print("Comparison against phase2_gate.md's original (pre-fix, truncated) numbers:")
print(f"  BEFORE: old P=0.762 R=0.813 F1=0.787  ->  new P={p_off:.3f} R={r_off:.3f} F1={f1_off:.3f}")
print(f"  AFTER:  old P=0.985 R=0.853 F1=0.914  ->  new P={p_on:.3f} R={r_on:.3f} F1={f1_on:.3f}")

out = {
    "before": {"precision": p_off, "recall": r_off, "f1": f1_off, "per_item": res_off["per_item"]},
    "after": {"precision": p_on, "recall": r_on, "f1": f1_on, "per_item": res_on["per_item"]},
    "old_before": {"precision": 0.762, "recall": 0.813, "f1": 0.787},
    "old_after": {"precision": 0.985, "recall": 0.853, "f1": 0.914},
}
(REPO_ROOT / "analysis" / "phase2_task3a_gold_remeasure.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_task3a_gold_remeasure.json")
