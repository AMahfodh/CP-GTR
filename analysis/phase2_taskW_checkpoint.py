"""Task W checkpoint report: parse fallback count, flag rate per CP-GTR arm,
differs-from-draft counts, host node/edge counts, round-trip pass rates,
unique-unit count so far, and any document whose realized text doesn't read
as notice text. Pure offline analysis of the already-generated
analysis/phase2_generation_pairs.json -- no new API calls.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import normalize_text, build_scored_documents, unique_units, inheritance_report

pairs = json.loads((REPO_ROOT / "analysis" / "phase2_generation_pairs.json").read_text(encoding="utf-8"))
print(f"Pairs so far: {len(pairs)} ({len(pairs)*6} documents)")

# ---- parse fallback count ----
host_fallbacks = [pid for pid, e in pairs.items() if e["_meta"]["host_parse_fallback"]]
rt_fallback_cpgtr = [pid for pid, e in pairs.items() if e["_meta"]["CP-GTR_round_trip_parse_fallback"]]
rt_fallback_ungated = [pid for pid, e in pairs.items() if e["_meta"]["CP-GTR-Ungated_round_trip_parse_fallback"]]
print(f"\nParse fallbacks:")
print(f"  host: {len(host_fallbacks)}/{len(pairs)}  {host_fallbacks or ''}")
print(f"  CP-GTR round-trip: {len(rt_fallback_cpgtr)}/{len(pairs)}  {rt_fallback_cpgtr or ''}")
print(f"  CP-GTR-Ungated round-trip: {len(rt_fallback_ungated)}/{len(pairs)}  {rt_fallback_ungated or ''}")
total_fallbacks = len(host_fallbacks) + len(rt_fallback_cpgtr) + len(rt_fallback_ungated)
print(f"  TOTAL: {total_fallbacks}")

# ---- flag rate per CP-GTR arm ----
detect_flags = [bool(e["_meta"]["CP-GTR-Detect_flagged"]) for e in pairs.values()]
cpgtr_flags = [e["_meta"]["CP-GTR_flagged"] for e in pairs.values()]
ungated_flags = [e["_meta"]["CP-GTR-Ungated_flagged"] for e in pairs.values()]
print(f"\nFlag rate (of {len(pairs)} pairs):")
print(f"  CP-GTR-Detect:  {sum(detect_flags)}/{len(pairs)} ({100*sum(detect_flags)/len(pairs):.0f}%)")
print(f"  CP-GTR:         {sum(cpgtr_flags)}/{len(pairs)} ({100*sum(cpgtr_flags)/len(pairs):.0f}%)")
print(f"  CP-GTR-Ungated: {sum(ungated_flags)}/{len(pairs)} ({100*sum(ungated_flags)/len(pairs):.0f}%)")

# ---- round-trip pass rates ----
cpgtr_rt_pass = [e["_meta"]["CP-GTR_round_trip_passed"] for e in pairs.values()]
ungated_rt_pass = [e["_meta"]["CP-GTR-Ungated_round_trip_passed"] for e in pairs.values()]
print(f"\nRound-trip pass rate:")
print(f"  CP-GTR:         {sum(cpgtr_rt_pass)}/{len(pairs)} ({100*sum(cpgtr_rt_pass)/len(pairs):.0f}%)")
print(f"  CP-GTR-Ungated: {sum(ungated_rt_pass)}/{len(pairs)} ({100*sum(ungated_rt_pass)/len(pairs):.0f}%)")

# ---- differs from draft ----
cpgtr_diff = [normalize_text(e["CP-GTR"]) != normalize_text(e["Unconstrained"]) for e in pairs.values()]
ungated_diff = [normalize_text(e["CP-GTR-Ungated"]) != normalize_text(e["Unconstrained"]) for e in pairs.values()]
print(f"\nCP-GTR differs from draft:         {sum(cpgtr_diff)}/{len(pairs)}")
print(f"CP-GTR-Ungated differs from draft:  {sum(ungated_diff)}/{len(pairs)}")

# ---- host node/edge counts ----
print(f"\nHost node/edge counts by pair:")
by_prompt = {}
for pid, e in pairs.items():
    by_prompt.setdefault(e["prompt_id"], []).append((pid, e["_meta"]["host_n_nodes"], e["_meta"]["host_n_edges"]))
for prompt_id, rows in by_prompt.items():
    ns = set((n, ed) for _, n, ed in rows)
    print(f"  {prompt_id}: {rows[0][1]}/{rows[0][2]} (N/E)"
          f"{' -- CONSTANT across all 9 contexts' if len(ns) == 1 else ' -- VARIES: ' + str(ns)}")

# ---- unique-unit count so far ----
scored_docs = build_scored_documents(pairs)
units = unique_units(scored_docs)
report = inheritance_report(scored_docs)
print(f"\nScored documents so far: {len(scored_docs)}  Unique annotation units: {len(units)}")
print(f"Per-arm inherited (share a unit with an already-counted doc): {report['per_arm_inherited']}")

# ---- flag any realized document that doesn't read as notice text ----
# Heuristic screen: a drafter-instruction/graph-edit sentence typically
# starts with an imperative verb ("Add", "Remove", "Delete", "Replace",
# "Prohibit", "Do not allow", "Ensure") addressed to a drafter, not a reader.
# Every CP-GTR/CP-GTR-Ungated document is checked by scanning its text for
# these patterns -- flagged for human eyeballing, not auto-rejected.
IMPERATIVE_MARKERS = ("Add a ", "Add an ", "Remove any", "Delete the", "Replace any",
                      "Replace indefinite", "Replace an indefinite", "Prohibit ",
                      "Do not allow", "Do not treat", "Ensure the ConsentProcess")
flagged_docs = []
for pid, e in pairs.items():
    for arm in ("CP-GTR", "CP-GTR-Ungated", "CP-GTR-Detect"):
        text = e[arm]
        for marker in IMPERATIVE_MARKERS:
            if marker in text:
                flagged_docs.append((pid, arm, marker, text))
                break

print(f"\nDocuments whose realized text may not read as notice text: {len(flagged_docs)}")
for pid, arm, marker, text in flagged_docs:
    print(f"  {pid} / {arm}  (matched {marker!r})")
    print(f"    {text}")

out = {
    "n_pairs": len(pairs), "n_documents": len(pairs) * 6,
    "parse_fallbacks": {"host": host_fallbacks, "cpgtr_round_trip": rt_fallback_cpgtr,
                        "ungated_round_trip": rt_fallback_ungated, "total": total_fallbacks},
    "flag_rate": {"CP-GTR-Detect": sum(detect_flags) / len(pairs), "CP-GTR": sum(cpgtr_flags) / len(pairs),
                  "CP-GTR-Ungated": sum(ungated_flags) / len(pairs)},
    "round_trip_pass_rate": {"CP-GTR": sum(cpgtr_rt_pass) / len(pairs),
                              "CP-GTR-Ungated": sum(ungated_rt_pass) / len(pairs)},
    "differs_from_draft": {"CP-GTR": sum(cpgtr_diff), "CP-GTR-Ungated": sum(ungated_diff)},
    "n_scored_documents": len(scored_docs), "n_unique_units": len(units),
    "inheritance_report": report,
    "non_notice_flagged": [{"pair_id": pid, "arm": arm, "marker": marker} for pid, arm, marker, _ in flagged_docs],
}
(REPO_ROOT / "analysis" / "phase2_taskW_checkpoint.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWritten: analysis/phase2_taskW_checkpoint.json")
