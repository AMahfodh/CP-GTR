"""Task AB: regenerate the SAME 27 pairs (P1, P2, P3 x 9 contexts) under the
IDENTICAL config as Task W, now with Task AA's provenance-verification fix
live in cpgtr/realize.py. Writes to SEPARATE files from Task W's
(phase2_generation_pairs.json / phase2_generation.json are untouched) so
both runs stay in the record for comparison.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cpgtr.realize as realize_mod
from cpgtr.admit import CONTEXTS
from cpgtr.baselines import draft_unconstrained, draft_fewshot, critique_revise_rlaif
from run_e2e import flag_violations, cpgtr_detect_repair
from analysis.phase2_harness import run_six_arms_for_pair, normalize_text, text_hash
from analysis.phase2_generation_lib import load_prompts, assert_config

PROMPTS_THIS_RUN = ["P1", "P2", "P3"]

PAIRS_PATH = REPO_ROOT / "analysis" / "phase2_generation_pairs_AB.json"
DOCS_PATH = REPO_ROOT / "analysis" / "phase2_generation_AB.json"

print("Asserting configuration (identical to Task W; refuses to run on any hash/count mismatch)...")
cfg = assert_config()
for k, v in cfg["config_summary"].items():
    print(f"  {k}: {v}")
print("Configuration asserted OK.\n")

eval_prompts, practice_prompts = load_prompts()

pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8")) if PAIRS_PATH.exists() else {}
print(f"{len(pairs)} pairs already present (resumed).")

complete, parser = cfg["complete"], cfg["parser"]
cpgtr_cert, ungated_cert = cfg["cpgtr_cert"], cfg["ungated_cert"]

realize_mod.verification_call_count = 0   # Task AA: count added calls for THIS run only

for prompt_id in PROMPTS_THIS_RUN:
    prompt_text = eval_prompts[prompt_id]
    pair_ids_for_prompt = [f"{prompt_id}_{A}_{J}" for (A, J) in CONTEXTS]
    if all(pid in pairs for pid in pair_ids_for_prompt):
        print(f"{prompt_id}: all 9 contexts already present, skipping.")
        continue

    print(f"=== {prompt_id}: {prompt_text[:70]}... ===")
    unconstrained_text = draft_unconstrained(complete, prompt_text)

    for i, ctx in enumerate(CONTEXTS):
        A, J = ctx
        pair_id = f"{prompt_id}_{A}_{J}"
        if pair_id in pairs:
            continue
        fewshot_text = draft_fewshot(complete, prompt_text, A, J)
        rlaif_text = critique_revise_rlaif(complete, unconstrained_text, A, J)

        result = run_six_arms_for_pair(
            complete, parser, cpgtr_cert, ungated_cert,
            unconstrained_text, fewshot_text, rlaif_text, ctx,
            flag_violations, cpgtr_detect_repair,
        )
        result["context"] = {"A": A, "J": J}
        result["prompt_id"] = prompt_id
        pairs[pair_id] = result
        print(f"  [{i+1}/9] {pair_id}: host N/E={result['_meta']['host_n_nodes']}/"
              f"{result['_meta']['host_n_edges']}  Detect={result['_meta']['CP-GTR-Detect_flagged']}  "
              f"CP-GTR={result['_meta']['CP-GTR_flagged']}  Ungated={result['_meta']['CP-GTR-Ungated_flagged']}  "
              f"host_fallback={result['_meta']['host_parse_fallback']}  "
              f"verif_calls_so_far={realize_mod.verification_call_count}")

        PAIRS_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nWritten: {PAIRS_PATH} ({len(pairs)} pairs total)")
print(f"Task AA verification calls made this run: {realize_mod.verification_call_count}")

# --------------------------------------------------------- flatten to per-document
ARM_LIB_HASH = {
    "Unconstrained": None, "Few-Shot Prompted": None, "Constitutional AI (RLAIF)": None,
    "CP-GTR-Detect": cfg["cpgtr_rec"]["sha256"], "CP-GTR": cfg["cpgtr_rec"]["sha256"],
    "CP-GTR-Ungated": cfg["ungated_rec"]["sha256"],
}

documents = []
for pair_id, entry in pairs.items():
    meta = entry["_meta"]
    for arm in ("Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)",
                "CP-GTR-Detect", "CP-GTR", "CP-GTR-Ungated"):
        text = entry[arm]
        flag_status = {
            "CP-GTR-Detect": meta["CP-GTR-Detect_flagged"],
            "CP-GTR": meta["CP-GTR_flagged"],
            "CP-GTR-Ungated": meta["CP-GTR-Ungated_flagged"],
        }.get(arm)
        rules_fired = {
            "CP-GTR-Detect": meta["CP-GTR-Detect_rules_fired"],
            "CP-GTR": meta["CP-GTR_rules_fired"],
            "CP-GTR-Ungated": meta["CP-GTR-Ungated_rules_fired"],
        }.get(arm)
        round_trip = {
            "CP-GTR": {"passed": meta["CP-GTR_round_trip_passed"],
                       "parse_fallback": meta["CP-GTR_round_trip_parse_fallback"]},
            "CP-GTR-Ungated": {"passed": meta["CP-GTR-Ungated_round_trip_passed"],
                                "parse_fallback": meta["CP-GTR-Ungated_round_trip_parse_fallback"]},
        }.get(arm)
        documents.append({
            "pair_id": pair_id, "prompt_id": entry["prompt_id"], "context": entry["context"],
            "arm": arm, "library_hash": ARM_LIB_HASH[arm], "flag_status": flag_status,
            "rules_fired": rules_fired, "round_trip": round_trip,
            "host_parse_fallback": meta["host_parse_fallback"],
            "host_parse_attempts": meta["host_parse_attempts"],
            "host_n_nodes": meta["host_n_nodes"], "host_n_edges": meta["host_n_edges"],
            "text": text, "text_hash": text_hash(text),
        })

DOCS_PATH.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Written: {DOCS_PATH} ({len(documents)} documents)")
