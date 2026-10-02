"""Repeated gold-subset measurement of LLMParser with and without topology.

Supports: the parser-quality section's statement that only the constrained
configuration is reported. The same configurations had given different numbers
in separate single runs, so this measures each condition five times in one
script to characterise run-to-run variance directly.

Conditions:
  1. no topology: LLMParser(complete, enforce_topology=False)
  2. topology, no negative example: LLMParser(complete, enforce_topology=True),
     the production configuration.
For each it reports mean, min and max P/R/F1 with all 12 gold entries and with
g09 excluded (11), over 5 independent live runs (5 runs x 2 conditions x 12
entries = 120 parse calls, before any repair-loop retries). Scoring uses
cpgtr.eval_parser.score_corpus().

Run (needs an LLM provider key; see cpgtr/llm.py):
    python analysis/phase2_taskK_repeated_gold.py
Order: independent diagnostic of the parser; no other script depends on it.

Inputs : corpus/gold_parses.json (hand-labelled gold triplets).
Outputs: analysis/phase2_taskK_repeated_gold.json.
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


def run_condition(label, enforce_topology):
    runs_12, runs_11 = [], []
    for i in range(N_RUNS):
        parser = LLMParser(complete, enforce_topology=enforce_topology)
        cache = {}
        for entry in gold:
            cache[entry["text"]] = parser.parse(entry["text"])

        class _Replay:
            def parse(self, text):
                return cache[text]

        res12 = score_corpus(_Replay(), gold)
        res11 = score_corpus(_Replay(), gold_no_g09)
        p12, r12, f12 = res12["overall"]["all"]
        p11, r11, f11 = res11["overall"]["all"]
        runs_12.append((p12, r12, f12))
        runs_11.append((p11, r11, f11))
        print(f"  [{label}] run {i+1}/{N_RUNS}: 12-entry P={p12:.3f} R={r12:.3f} F1={f12:.3f}  "
              f"| 11-entry P={p11:.3f} R={r11:.3f} F1={f11:.3f}  "
              f"fallback={res12['fallback_count']}/12")
    return runs_12, runs_11


def summarize(runs, names=("P", "R", "F1")):
    out = {}
    for i, name in enumerate(names):
        vals = [r[i] for r in runs]
        out[name] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals),
                      "values": vals}
    return out


print(f"=== Condition 1: no topology ({N_RUNS} runs) ===")
no_topo_12, no_topo_11 = run_condition("no-topology", enforce_topology=False)

print(f"\n=== Condition 2: topology, no negative instruction / production config ({N_RUNS} runs) ===")
topo_12, topo_11 = run_condition("topology", enforce_topology=True)

results = {
    "no_topology": {"with_g09": summarize(no_topo_12), "without_g09": summarize(no_topo_11)},
    "topology": {"with_g09": summarize(topo_12), "without_g09": summarize(topo_11)},
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

(REPO_ROOT / "analysis" / "phase2_taskK_repeated_gold.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskK_repeated_gold.json")
