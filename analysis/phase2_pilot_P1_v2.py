"""Task L (pasted task list, 2026-09-23): re-run the P1 pilot with Task I
(corrected realization templates) and Task J (negative-example instruction
reverted) applied. Same 9 contexts, six arms, same stop rules as the first
pilot (analysis/phase2_pilot_P1.py / phase2_pilot_P1_report.md).

STOP RULES (unchanged):
  - CP-GTR's output identical to the draft in ALL 9 contexts
  - flag rate (any CP-GTR-family arm) exceeds 50%
  - any parse falls back
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
parser = LLMParser(complete, enforce_topology=True)

cpgtr_cert, cpgtr_rec = build_rules_for_library("LIB-FAITHFUL-P")
ungated_cert, ungated_rec = build_rules_for_library("LIB-FULL")
apply_template_corrections(cpgtr_cert)
apply_template_corrections(ungated_cert)
print(f"LIB-FAITHFUL-P: {len(cpgtr_cert)} rules, sha256={cpgtr_rec['sha256'][:16]}... (Task I templates applied)")
print(f"LIB-FULL:       {len(ungated_cert)} rules, sha256={ungated_rec['sha256'][:16]}... (Task I templates applied)")

unconstrained_text = draft_unconstrained(complete, P1_TEXT)
print(f"\nUnconstrained draft ({len(unconstrained_text)} chars):\n{unconstrained_text}\n")

generation = {}
for i, ctx in enumerate(CONTEXTS):
    A, J = ctx
    pair_id = f"P1_{A}_{J}"
    fewshot_text = draft_fewshot(complete, P1_TEXT, A, J)
    rlaif_text = critique_revise_rlaif(complete, unconstrained_text, A, J)

    result = run_six_arms_for_pair(
        complete, parser, cpgtr_cert, ungated_cert,
        unconstrained_text, fewshot_text, rlaif_text, ctx,
        flag_violations, cpgtr_detect_repair,
    )
    result["context"] = {"A": A, "J": J}
    result["prompt_id"] = "P1"
    generation[pair_id] = result
    print(f"[{i+1}/9] {pair_id}: host N/E={result['_meta']['host_n_nodes']}/"
          f"{result['_meta']['host_n_edges']}  "
          f"Detect_flagged={bool(result['_meta']['CP-GTR-Detect_flagged'])}  "
          f"CP-GTR_flagged={result['_meta']['CP-GTR_flagged']}  "
          f"Ungated_flagged={result['_meta']['CP-GTR-Ungated_flagged']}  "
          f"host_fallback={result['_meta']['host_parse_fallback']}  "
          f"Ungated_redoc_edges={result['_meta']['CP-GTR-Ungated_redoc_edge_types']}")

    out_path = REPO_ROOT / "analysis" / "phase2_pilot_P1_v2_generation.json"
    out_path.write_text(json.dumps(generation, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nWritten: {out_path}")

# --------------------------------------------------------------- reporting
print("\n" + "=" * 70)
print("REPORT")
print("=" * 70)

doc_lengths = {}
for arm in ("Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)",
            "CP-GTR-Detect", "CP-GTR", "CP-GTR-Ungated"):
    lens = [len(entry[arm]) for entry in generation.values()]
    doc_lengths[arm] = lens
    print(f"{arm:30s} lengths: {lens}  (min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.0f})")

print("\nFlag rate per CP-GTR-family arm (of 9 contexts):")
detect_flags = [bool(e["_meta"]["CP-GTR-Detect_flagged"]) for e in generation.values()]
cpgtr_flags = [e["_meta"]["CP-GTR_flagged"] for e in generation.values()]
ungated_flags = [e["_meta"]["CP-GTR-Ungated_flagged"] for e in generation.values()]
print(f"  CP-GTR-Detect:  {sum(detect_flags)}/9 ({100*sum(detect_flags)/9:.0f}%)")
print(f"  CP-GTR:         {sum(cpgtr_flags)}/9 ({100*sum(cpgtr_flags)/9:.0f}%)")
print(f"  CP-GTR-Ungated: {sum(ungated_flags)}/9 ({100*sum(ungated_flags)/9:.0f}%)")

print("\nRound-trip PASS rate per arm (of 9 contexts) -- new in v2:")
cpgtr_rt_pass = [e["_meta"]["CP-GTR_round_trip_passed"] for e in generation.values()]
ungated_rt_pass = [e["_meta"]["CP-GTR-Ungated_round_trip_passed"] for e in generation.values()]
print(f"  CP-GTR:         {sum(cpgtr_rt_pass)}/9 ({100*sum(cpgtr_rt_pass)/9:.0f}%)")
print(f"  CP-GTR-Ungated: {sum(ungated_rt_pass)}/9 ({100*sum(ungated_rt_pass)/9:.0f}%)")

print("\nHow many CP-GTR outputs differ from the draft (Unconstrained):")
cpgtr_diff = [normalize_text(e["CP-GTR"]) != normalize_text(e["Unconstrained"]) for e in generation.values()]
ungated_diff = [normalize_text(e["CP-GTR-Ungated"]) != normalize_text(e["Unconstrained"]) for e in generation.values()]
print(f"  CP-GTR differs from draft:         {sum(cpgtr_diff)}/9")
print(f"  CP-GTR-Ungated differs from draft:  {sum(ungated_diff)}/9")

scored_docs = build_scored_documents(generation)
units = unique_units(scored_docs)
print(f"\nScored documents: {len(scored_docs)}  Unique annotation units: {len(units)}")

print("\nCP-GTR firing rate per context (pre-registered prediction 2: LIB-FAITHFUL-P is COPPA-heavy, low activity expected):")
for pid, e in generation.items():
    fired = e["_meta"]["CP-GTR_rules_fired"]
    print(f"  {pid} (A={e['context']['A']}, J={e['context']['J']}): fired={fired or '[]'}")
print(f"  CP-GTR fired in {sum(1 for e in generation.values() if e['_meta']['CP-GTR_rules_fired'])}/9 contexts")

print("\nTask H instrumentation summary:")
fallback_hosts = [pid for pid, e in generation.items() if e["_meta"]["host_parse_fallback"]]
rt_fallback_cpgtr = [pid for pid, e in generation.items() if e["_meta"]["CP-GTR_round_trip_parse_fallback"]]
rt_fallback_ungated = [pid for pid, e in generation.items() if e["_meta"]["CP-GTR-Ungated_round_trip_parse_fallback"]]
print(f"  host parse fallbacks: {fallback_hosts or 'none'}")
print(f"  CP-GTR round-trip parse fallbacks: {rt_fallback_cpgtr or 'none'}")
print(f"  CP-GTR-Ungated round-trip parse fallbacks: {rt_fallback_ungated or 'none'}")
print(f"  host node/edge counts by context:")
for pid, e in generation.items():
    print(f"    {pid}: N={e['_meta']['host_n_nodes']} E={e['_meta']['host_n_edges']} "
          f"attempts={e['_meta']['host_parse_attempts']}")

print("\nCP-GTR-Ungated round-trip redoc edge types by context (does 'grants' survive?):")
grants_survives = []
for pid, e in generation.items():
    edges = e["_meta"]["CP-GTR-Ungated_redoc_edge_types"]
    survives = "grants" in edges
    grants_survives.append(survives)
    print(f"  {pid}: redoc_edges={edges}  grants_survives={survives}")
print(f"  grants edge survives round-trip re-parse in {sum(grants_survives)}/9 contexts")

any_fallback = bool(fallback_hosts or rt_fallback_cpgtr or rt_fallback_ungated)
max_flag_rate = max(sum(detect_flags), sum(cpgtr_flags), sum(ungated_flags)) / 9
cpgtr_never_differs = not any(cpgtr_diff)

print("\n" + "=" * 70)
print("STOP RULES")
print("=" * 70)
print(f"CP-GTR identical to draft in ALL 9 contexts? {cpgtr_never_differs}")
print(f"Any CP-GTR-family flag rate > 50%? {max_flag_rate > 0.5}  (max observed: {max_flag_rate:.0%})")
print(f"Any parse fell back? {any_fallback}")

triggered = cpgtr_never_differs or (max_flag_rate > 0.5) or any_fallback
print(f"\n>>> STOP TRIGGERED: {triggered} <<<")

# ----------------------------------------------------- three verbatim docs
sample_pair_id = next(iter(generation))
sample = generation[sample_pair_id]
print(f"\n=== Three verbatim documents at {sample_pair_id} (A={sample['context']['A']}, J={sample['context']['J']}) ===")
print(f"\n--- Unconstrained ---\n{sample['Unconstrained']}")
print(f"\n--- CP-GTR ---\n{sample['CP-GTR']}")
print(f"\n--- CP-GTR-Ungated ---\n{sample['CP-GTR-Ungated']}")

summary = {
    "doc_lengths": doc_lengths,
    "flag_rates": {"CP-GTR-Detect": sum(detect_flags) / 9, "CP-GTR": sum(cpgtr_flags) / 9,
                   "CP-GTR-Ungated": sum(ungated_flags) / 9},
    "round_trip_pass_rates": {"CP-GTR": sum(cpgtr_rt_pass) / 9, "CP-GTR-Ungated": sum(ungated_rt_pass) / 9},
    "cpgtr_differs_from_draft": sum(cpgtr_diff), "ungated_differs_from_draft": sum(ungated_diff),
    "n_scored_documents": len(scored_docs), "n_unique_units": len(units),
    "cpgtr_firing_rate": sum(1 for e in generation.values() if e["_meta"]["CP-GTR_rules_fired"]) / 9,
    "grants_survives_roundtrip_count": sum(grants_survives),
    "fallback_hosts": fallback_hosts, "rt_fallback_cpgtr": rt_fallback_cpgtr,
    "rt_fallback_ungated": rt_fallback_ungated,
    "stop_rules": {
        "cpgtr_identical_all_9": cpgtr_never_differs,
        "max_flag_rate": max_flag_rate,
        "any_parse_fallback": any_fallback,
        "triggered": triggered,
    },
    "sample_pair_id": sample_pair_id,
}
(REPO_ROOT / "analysis" / "phase2_pilot_P1_v2_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_pilot_P1_v2_summary.json")
