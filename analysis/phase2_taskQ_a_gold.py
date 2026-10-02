"""Task Q(a) (pasted task list, 2026-09-23): A/B the re-applied
negative_example parameter (cpgtr/parse.py: LLMParser(negative_example=...))
on the gold subset, WITH corrected templates in place (not that this
sub-task uses templates -- noted for consistency with the pasted
instruction's framing; gold scoring doesn't invoke realize()). 5 repeats per
condition, mean/min/max P/R/F1, both 12-entry and 11-entry (g09 excluded).
No monkeypatching -- uses the real `negative_example` constructor parameter.
"""
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
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


def run_condition(label, negative_example):
    runs_12, runs_11 = [], []
    for i in range(N_RUNS):
        parser = LLMParser(complete, enforce_topology=True, negative_example=negative_example)
        cache = {}
        for entry in gold:
            cache[entry["text"]] = parser.parse(entry["text"])
        res12 = score_corpus(_Replay(cache), gold)
        res11 = score_corpus(_Replay(cache), gold_no_g09)
        p12, r12, f12 = res12["overall"]["all"]
        p11, r11, f11 = res11["overall"]["all"]
        runs_12.append((p12, r12, f12))
        runs_11.append((p11, r11, f11))
        print(f"  [{label}] run {i+1}/{N_RUNS}: 12-entry P={p12:.3f} R={r12:.3f} F1={f12:.3f}  "
              f"| 11-entry P={p11:.3f} R={r11:.3f} F1={f11:.3f}  fallback={res12['fallback_count']}/12")
    return runs_12, runs_11


def summarize(runs, names=("P", "R", "F1")):
    out = {}
    for i, name in enumerate(names):
        vals = [r[i] for r in runs]
        out[name] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals), "values": vals}
    return out


print(f"=== Condition A: without negative_example ({N_RUNS} runs) ===")
a_12, a_11 = run_condition("A", negative_example=False)

print(f"\n=== Condition B: WITH negative_example ({N_RUNS} runs) ===")
b_12, b_11 = run_condition("B", negative_example=True)

results = {
    "A": {"with_g09": summarize(a_12), "without_g09": summarize(a_11)},
    "B": {"with_g09": summarize(b_12), "without_g09": summarize(b_11)},
}

print("\n" + "=" * 70)
print("SUMMARY (mean [min, max] over 5 runs)")
print("=" * 70)
for cond_label, cond in results.items():
    for subset_label, subset in cond.items():
        p, r, f1 = subset["P"], subset["R"], subset["F1"]
        print(f"{cond_label:12s} {subset_label:12s} "
              f"P={p['mean']:.3f} [{p['min']:.3f},{p['max']:.3f}]  "
              f"R={r['mean']:.3f} [{r['min']:.3f},{r['max']:.3f}]  "
              f"F1={f1['mean']:.3f} [{f1['min']:.3f},{f1['max']:.3f}]")

(REPO_ROOT / "analysis" / "phase2_taskQ_a_gold.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskQ_a_gold.json")
