"""Gold-subset parser quality of the production LLMParser, five repeats.

Supports: the parser-quality table (LLMParser row: precision 0.889, recall
0.853, F1 0.871 on the twelve-entry gold subset, mean of five repeats whose
minimum and maximum coincide with the mean) and the 11-entry figure that
excludes g09.

Parses the 12 gold entries five times with the production configuration
(canonical topology enforcement, no negative example in the prompt, negation
filter active) and scores each run against corpus/gold_parses.json with
cpgtr.eval_parser.score_corpus() (a Smatch-style scorer). It reports per-run
P/R/F1, the mean, min and max for the 12-entry set and for the 11-entry set
without g09, the fallback count per run and the total number of
negation-filter drops. It also compares the 11-entry F1 with the recorded
baseline of 0.914 (same configuration before the negation filter existed),
which the filter must not worsen.

Run (needs an LLM provider key; see cpgtr/llm.py):
    python analysis/phase2_taskAD_gold_check.py
Order: independent of the generation scripts; run after the parser
configuration is final.

Inputs : corpus/gold_parses.json (hand-labelled gold triplets).
Outputs: analysis/phase2_taskAD_gold_check.json.
"""
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
import cpgtr.parse as parse_mod
from cpgtr.eval_parser import score_corpus
from cpgtr.llm import make_complete

gold = json.loads((REPO_ROOT / "corpus" / "gold_parses.json").read_text(encoding="utf-8"))
gold_no_g09 = [e for e in gold if e["id"] != "g09"]

complete = make_complete()
N_RUNS = 5


class _Replay:
    def __init__(self, cache):
        self.cache = cache

    def parse(self, text):
        return self.cache[text]


runs_12, runs_11 = [], []
parse_mod.negation_filter_drop_count = 0
for i in range(N_RUNS):
    # Production configuration; the negation filter is active by default.
    parser = LLMParser(complete, enforce_topology=True, negative_example=False)
    cache = {}
    for entry in gold:
        cache[entry["text"]] = parser.parse(entry["text"])
    res12 = score_corpus(_Replay(cache), gold)
    res11 = score_corpus(_Replay(cache), gold_no_g09)
    p12, r12, f12 = res12["overall"]["all"]
    p11, r11, f11 = res11["overall"]["all"]
    runs_12.append((p12, r12, f12))
    runs_11.append((p11, r11, f11))
    print(f"run {i+1}/{N_RUNS}: 12-entry P={p12:.3f} R={r12:.3f} F1={f12:.3f}  "
          f"| 11-entry P={p11:.3f} R={r11:.3f} F1={f11:.3f}  fallback={res12['fallback_count']}/12")

print(f"\nTotal negation-filter drops across gold-subset measurement: {parse_mod.negation_filter_drop_count}")


def summarize(runs, names=("P", "R", "F1")):
    out = {}
    for i, name in enumerate(names):
        vals = [r[i] for r in runs]
        out[name] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    return out

s12, s11 = summarize(runs_12), summarize(runs_11)
print("\n12-entry:", s12)
print("11-entry:", s11)

# 11-entry F1 recorded for the same configuration before the negation filter
# existed.
BASELINE_11_F1 = 0.914
new_f1 = s11["F1"]["mean"]
print(f"\nBaseline 11-entry F1 (pre-filter): {BASELINE_11_F1}")
print(f"New 11-entry F1 (with filter, mean of {N_RUNS} runs): {new_f1:.3f}")
print(f"Clause (i) [11-entry F1 no worse than baseline]: {new_f1 >= BASELINE_11_F1}")

out = {"runs_12": runs_12, "runs_11": runs_11, "summary_12": s12, "summary_11": s11,
       "baseline_11_f1": BASELINE_11_F1, "new_11_f1_mean": new_f1,
       "clause_i_passes": new_f1 >= BASELINE_11_F1,
       "total_negation_drops": parse_mod.negation_filter_drop_count}
(REPO_ROOT / "analysis" / "phase2_taskAD_gold_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskAD_gold_check.json")
