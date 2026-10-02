"""Task Q(b) (pasted task list, 2026-09-23): A/B the re-applied
negative_example parameter with a full P1 pilot (9 contexts, six arms) under
each condition, corrected templates in place (Task I). Reports flag rate,
round-trip pass rate, grants-edge survival per context, host node/edge
counts, parse fallbacks, and CP-GTR-differs-from-draft count for BOTH
conditions. Real API calls -- roughly 2x the cost of a single pilot run.
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

cpgtr_cert, cpgtr_rec = build_rules_for_library("LIB-FAITHFUL-P")
ungated_cert, ungated_rec = build_rules_for_library("LIB-FULL")
apply_template_corrections(cpgtr_cert)
apply_template_corrections(ungated_cert)

# Unconstrained is context-blind -- drafted once, shared across BOTH
# conditions and all 9 contexts, so the two conditions differ ONLY in the
# parser, not in the underlying draft.
unconstrained_text = draft_unconstrained(complete, P1_TEXT)
print(f"Unconstrained draft ({len(unconstrained_text)} chars):\n{unconstrained_text}\n")

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
              f"Ungated_redoc={result['_meta']['CP-GTR-Ungated_redoc_edge_types']}")
        out_path = REPO_ROOT / "analysis" / f"phase2_taskQ_b_pilot_{label}_generation.json"
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

    print(f"\n--- {label} summary ---")
    print(f"  Flag rate: Detect={sum(detect_flags)}/9  CP-GTR={sum(cpgtr_flags)}/9  Ungated={sum(ungated_flags)}/9")
    print(f"  Round-trip pass rate: CP-GTR={sum(cpgtr_rt_pass)}/9  Ungated={sum(ungated_rt_pass)}/9")
    print(f"  CP-GTR differs from draft: {sum(cpgtr_diff)}/9   Ungated differs from draft: {sum(ungated_diff)}/9")
    print(f"  Host parse fallbacks: {fallback_hosts or 'none'}")
    print(f"  Round-trip parse fallbacks: CP-GTR={rt_fallback_cpgtr or 'none'}  Ungated={rt_fallback_ungated or 'none'}")
    print(f"  grants edge survives round-trip: CP-GTR={sum(grants_survives_cpgtr)}/9  Ungated={sum(grants_survives_ungated)}/9")
    for pid, e in generation.items():
        print(f"    {pid}: N={e['_meta']['host_n_nodes']} E={e['_meta']['host_n_edges']}  "
              f"CP-GTR_grants={('grants' in e['_meta']['CP-GTR_redoc_edge_types'])}  "
              f"Ungated_grants={('grants' in e['_meta']['CP-GTR-Ungated_redoc_edge_types'])}")

    return {
        "flag_rate": {"Detect": sum(detect_flags) / 9, "CP-GTR": sum(cpgtr_flags) / 9,
                      "CP-GTR-Ungated": sum(ungated_flags) / 9},
        "round_trip_pass_rate": {"CP-GTR": sum(cpgtr_rt_pass) / 9, "CP-GTR-Ungated": sum(ungated_rt_pass) / 9},
        "cpgtr_differs_from_draft": sum(cpgtr_diff), "ungated_differs_from_draft": sum(ungated_diff),
        "fallback_hosts": fallback_hosts, "rt_fallback_cpgtr": rt_fallback_cpgtr,
        "rt_fallback_ungated": rt_fallback_ungated,
        "grants_survives": {"CP-GTR": sum(grants_survives_cpgtr), "CP-GTR-Ungated": sum(grants_survives_ungated)},
        "host_node_edge_by_context": {pid: {"nodes": e["_meta"]["host_n_nodes"], "edges": e["_meta"]["host_n_edges"]}
                                       for pid, e in generation.items()},
    }


print("=" * 70)
print("CONDITION A: without negative_example")
print("=" * 70)
gen_a = run_pilot("A", negative_example=False)
sum_a = summarize("A", gen_a)

print("\n" + "=" * 70)
print("CONDITION B: WITH negative_example")
print("=" * 70)
gen_b = run_pilot("B", negative_example=True)
sum_b = summarize("B", gen_b)

results = {"A": sum_a, "B": sum_b}
(REPO_ROOT / "analysis" / "phase2_taskQ_b_summary.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskQ_b_summary.json")
