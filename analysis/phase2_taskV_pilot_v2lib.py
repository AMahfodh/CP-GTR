"""Task V: re-run Task Q(b) (the only sub-task of Task Q that depends on
WHICH library is used -- Q(a)'s gold-subset measurement and Q(c)'s g09
trials involve no certified rules at all, so they are unaffected by the
library correction and are not re-run here) against LIB-FAITHFUL-P-v2
(8 rules, Task S) instead of the original LIB-FAITHFUL-P (13 rules).
Applies the SAME pre-registered decision rule from Task Q, unchanged.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
from cpgtr.llm import make_complete
from cpgtr.admit import CONTEXTS
from cpgtr.baselines import draft_unconstrained, draft_fewshot, critique_revise_rlaif
from run_e2e import flag_violations, cpgtr_detect_repair
from analysis.phase2_harness import (
    build_rules_for_library, apply_template_corrections, run_six_arms_for_pair,
    normalize_text, build_scored_documents, unique_units,
)

P1_TEXT = ("Draft a short privacy notice for a mobile app's sign-up flow: we "
           "collect the user's email address and date of birth to create "
           "their account, and by signing up the user consents to this "
           "collection.")

complete = make_complete()

cpgtr_cert, cpgtr_rec = build_rules_for_library("LIB-FAITHFUL-P-v2")
ungated_cert, ungated_rec = build_rules_for_library("LIB-FULL")
apply_template_corrections(cpgtr_cert)
apply_template_corrections(ungated_cert)
print(f"CP-GTR library: LIB-FAITHFUL-P-v2, n={len(cpgtr_cert)}, sha256={cpgtr_rec['sha256'][:16]}...")
print(f"CP-GTR-Ungated library: LIB-FULL, n={len(ungated_cert)}, sha256={ungated_rec['sha256'][:16]}...")

unconstrained_text = draft_unconstrained(complete, P1_TEXT)
print(f"\nUnconstrained draft ({len(unconstrained_text)} chars):\n{unconstrained_text}\n")

fewshot_by_ctx, rlaif_by_ctx = {}, {}
for ctx in CONTEXTS:
    A, J = ctx
    fewshot_by_ctx[ctx] = draft_fewshot(complete, P1_TEXT, A, J)
    rlaif_by_ctx[ctx] = critique_revise_rlaif(complete, unconstrained_text, A, J)
print("Few-Shot/RLAIF drafted once per context, shared across both conditions.\n")


def run_pilot(label, negative_example):
    parser = LLMParser(complete, enforce_topology=True, negative_example=negative_example)
    generation = {}
    for i, ctx in enumerate(CONTEXTS):
        A, J = ctx
        pair_id = f"P1_{A}_{J}"
        result = run_six_arms_for_pair(
            complete, parser, cpgtr_cert, ungated_cert,
            unconstrained_text, fewshot_by_ctx[ctx], rlaif_by_ctx[ctx], ctx,
            flag_violations, cpgtr_detect_repair,
        )
        result["context"] = {"A": A, "J": J}
        result["prompt_id"] = "P1"
        generation[pair_id] = result
        print(f"  [{label} {i+1}/9] {pair_id}: host N/E={result['_meta']['host_n_nodes']}/"
              f"{result['_meta']['host_n_edges']}  CP-GTR_flagged={result['_meta']['CP-GTR_flagged']}  "
              f"Ungated_flagged={result['_meta']['CP-GTR-Ungated_flagged']}  "
              f"CP-GTR_rules_fired={result['_meta']['CP-GTR_rules_fired']}")
        out_path = REPO_ROOT / "analysis" / f"phase2_taskV_pilot_{label}_generation.json"
        out_path.write_text(json.dumps(generation, indent=2, ensure_ascii=False), encoding="utf-8")
    return generation


def summarize(label, generation):
    detect_flags = [bool(e["_meta"]["CP-GTR-Detect_flagged"]) for e in generation.values()]
    cpgtr_flags = [e["_meta"]["CP-GTR_flagged"] for e in generation.values()]
    ungated_flags = [e["_meta"]["CP-GTR-Ungated_flagged"] for e in generation.values()]
    cpgtr_rt_pass = [e["_meta"]["CP-GTR_round_trip_passed"] for e in generation.values()]
    ungated_rt_pass = [e["_meta"]["CP-GTR-Ungated_round_trip_passed"] for e in generation.values()]
    cpgtr_diff = [normalize_text(e["CP-GTR"]) != normalize_text(e["Unconstrained"]) for e in generation.values()]
    ungated_diff = [normalize_text(e["CP-GTR-Ungated"]) != normalize_text(e["Unconstrained"]) for e in generation.values()]
    fallback_hosts = [pid for pid, e in generation.items() if e["_meta"]["host_parse_fallback"]]
    rt_fallback_cpgtr = [pid for pid, e in generation.items() if e["_meta"]["CP-GTR_round_trip_parse_fallback"]]
    rt_fallback_ungated = [pid for pid, e in generation.items() if e["_meta"]["CP-GTR-Ungated_round_trip_parse_fallback"]]
    grants_survives_cpgtr = [("grants" in e["_meta"]["CP-GTR_redoc_edge_types"]) for e in generation.values()]
    grants_survives_ungated = [("grants" in e["_meta"]["CP-GTR-Ungated_redoc_edge_types"]) for e in generation.values()]

    print(f"\n--- {label} summary (LIB-FAITHFUL-P-v2) ---")
    print(f"  Flag rate: Detect={sum(detect_flags)}/9  CP-GTR={sum(cpgtr_flags)}/9  Ungated={sum(ungated_flags)}/9")
    print(f"  Round-trip pass rate: CP-GTR={sum(cpgtr_rt_pass)}/9  Ungated={sum(ungated_rt_pass)}/9")
    print(f"  CP-GTR differs from draft: {sum(cpgtr_diff)}/9   Ungated differs from draft: {sum(ungated_diff)}/9")
    print(f"  Host parse fallbacks: {fallback_hosts or 'none'}")
    print(f"  Round-trip parse fallbacks: CP-GTR={rt_fallback_cpgtr or 'none'}  Ungated={rt_fallback_ungated or 'none'}")
    print(f"  grants edge survives round-trip: CP-GTR={sum(grants_survives_cpgtr)}/9  Ungated={sum(grants_survives_ungated)}/9")

    return {
        "flag_rate": {"Detect": sum(detect_flags) / 9, "CP-GTR": sum(cpgtr_flags) / 9,
                      "CP-GTR-Ungated": sum(ungated_flags) / 9},
        "round_trip_pass_rate": {"CP-GTR": sum(cpgtr_rt_pass) / 9, "CP-GTR-Ungated": sum(ungated_rt_pass) / 9},
        "cpgtr_differs_from_draft": sum(cpgtr_diff), "ungated_differs_from_draft": sum(ungated_diff),
        "fallback_hosts": fallback_hosts, "rt_fallback_cpgtr": rt_fallback_cpgtr,
        "rt_fallback_ungated": rt_fallback_ungated,
        "grants_survives": {"CP-GTR": sum(grants_survives_cpgtr), "CP-GTR-Ungated": sum(grants_survives_ungated)},
    }


print("=" * 70)
print("CONDITION A: without negative_example (LIB-FAITHFUL-P-v2)")
print("=" * 70)
gen_a = run_pilot("A", negative_example=False)
sum_a = summarize("A", gen_a)

print("\n" + "=" * 70)
print("CONDITION B: WITH negative_example (LIB-FAITHFUL-P-v2)")
print("=" * 70)
gen_b = run_pilot("B", negative_example=True)
sum_b = summarize("B", gen_b)

results = {"A": sum_a, "B": sum_b}
(REPO_ROOT / "analysis" / "phase2_taskV_summary.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskV_summary.json")

# ---- pre-registered decision rule, applied exactly as in Task Q ----
print("\n" + "=" * 70)
print("DECISION RULE (unchanged from Task Q)")
print("=" * 70)
gold_clause = True  # unaffected by library -- Task Q(a) already showed B's 11-entry F1 (0.921) >= A's (0.914)
rt_clause = (sum_b["round_trip_pass_rate"]["CP-GTR"] == 1.0 and sum_b["round_trip_pass_rate"]["CP-GTR-Ungated"] == 1.0)
grants_clause = (sum_b["grants_survives"]["CP-GTR"] == 9 and sum_b["grants_survives"]["CP-GTR-Ungated"] == 9)
print(f"Clause 1 (B's 11-entry gold F1 >= A's, from Task Q(a), library-independent): {gold_clause}")
print(f"Clause 2 (B's round-trip pass rate = 9/9 both arms): {rt_clause}  "
      f"(CP-GTR={sum_b['round_trip_pass_rate']['CP-GTR']*9:.0f}/9, "
      f"Ungated={sum_b['round_trip_pass_rate']['CP-GTR-Ungated']*9:.0f}/9)")
print(f"Clause 3 (B's grants edge survives 9/9 both arms): {grants_clause}")
adopt_b = gold_clause and rt_clause and grants_clause
print(f"\n>>> DECISION: {'ADOPT Condition B' if adopt_b else 'KEEP Condition A'} <<<")
