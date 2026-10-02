"""Violations per document, paired comparisons, and the scoped CCR.

Supports: the violations-per-document table (overall and by age band) and its
paired bootstrap comparisons against CP-GTR, plus a secondary scoped CCR. No
new data: everything is computed from the adjudicated final labels and the
generated pairs.

(a) Violation count. Mean violations per document per arm, overall, per age
    band and per (age, jurisdiction) cell, with prompt-level cluster bootstrap
    CIs (the same method as the CCR comparisons in
    phase2_final_results_part2.py), and paired CP-GTR-versus-each-other-arm
    differences with Holm correction within each scope's family of five.
(b) Scoped CCR (secondary, post hoc, not the headline). The same
    compliance_rate() computation, but each document's violation list is
    filtered to the codes LIB-FAITHFUL-P-v2 targets (from
    phase2_coverage_map.json) before scoring, reported per age band for all six
    arms. Those targeted codes only ever apply at (age 9, US); every other
    (age, jurisdiction) cell is vacuously compliant once scoped, so only the
    age-9 column, and within it the 9-US cell, carries signal. The script
    reports the 9-US cell separately and prints this caveat.

Run:
    python analysis/phase2_final_results_part3.py
Order: after phase2_final_results.py (final labels) and
phase2_final_results_part2.py (coverage map).

Inputs : analysis/phase2_final_labels.json (adjudicated human labels, not part
         of the public release), analysis/phase2_coverage_map.json,
         analysis/phase2_blind_annotation_sheet.csv,
         analysis/phase2_generation_pairs.json (LLM-generated text),
         schema/codebook_v2.json.
Outputs: analysis/phase2_final_mean_violations.json,
         analysis/phase2_final_mean_violation_bootstrap.json,
         analysis/phase2_final_scoped_ccr.json.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import build_scored_documents, propagate_labels, text_hash
from cpgtr.eval_compliance import compliance_rate

CODEBOOK_V2_PATH = REPO_ROOT / "schema" / "codebook_v2.json"
CSV_PATH = REPO_ROOT / "analysis" / "phase2_blind_annotation_sheet.csv"
PAIRS_PATH = REPO_ROOT / "analysis" / "phase2_generation_pairs.json"
COVERAGE_MAP_PATH = REPO_ROOT / "analysis" / "phase2_coverage_map.json"
FINAL_LABELS_PATH = REPO_ROOT / "analysis" / "phase2_final_labels.json"

codebook = json.loads(CODEBOOK_V2_PATH.read_text(encoding="utf-8"))
CODES = codebook["codes"] + [codebook["over_requirement"]]

final_labels = json.loads(FINAL_LABELS_PATH.read_text(encoding="utf-8"))
csv_rows = list(__import__("csv").DictReader(open(CSV_PATH, encoding="utf-8")))
unit_id_to_tuple = {f"U{int(r['row_id']):03d}": (text_hash(r["text"]), int(r["A"]), r["J"]) for r in csv_rows}
unit_labels = {unit_id_to_tuple[uid]: [c for c, v in row.items() if v == "yes"] for uid, row in final_labels.items()}

pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
scored_docs = build_scored_documents(pairs)
doc_labels = propagate_labels(scored_docs, unit_labels)
assert all((d["pair_id"], d["arm"]) in doc_labels for d in scored_docs)

coverage_map = json.loads(COVERAGE_MAP_PATH.read_text(encoding="utf-8"))
TARGETED_CODES = sorted(c for c, info in coverage_map["LIB-FAITHFUL-P-v2"].items() if info["targeted_at_all"])
print(f"LIB-FAITHFUL-P-v2-targeted codes (scoped CCR basis): {TARGETED_CODES}")

ALL_ARMS = ["Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)",
            "CP-GTR-Detect", "CP-GTR", "CP-GTR-Ungated"]
COMPARISON_ARMS = ["Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)",
                   "CP-GTR-Detect", "CP-GTR-Ungated"]
AGE_LABELS = [(9, "Child"), (15, "Teen"), (21, "Adult")]
Js = ["US", "EU", "UK"]
N_BOOT = 5000
BOOT_SEED = 2026

# =========================================================================== #
# (a) Mean violation count per document, per arm.
# =========================================================================== #


def viol_count(pid, arm):
    return len(doc_labels[(pid, arm)])


def prompt_grouped_counts(arm, age=None, J=None):
    groups = defaultdict(list)
    for d in scored_docs:
        if d["arm"] != arm:
            continue
        if age is not None and d["A"] != age:
            continue
        if J is not None and d["J"] != J:
            continue
        prompt_id = pairs[d["pair_id"]]["prompt_id"]
        groups[prompt_id].append(viol_count(d["pair_id"], arm))
    return groups


def mean_and_boot_ci(groups, n_boot=N_BOOT, seed=BOOT_SEED):
    prompt_ids = sorted(groups.keys())

    def mean_of(ids):
        vals = [v for pid in ids for v in groups[pid]]
        return sum(vals) / len(vals)

    point = mean_of(prompt_ids)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        resampled = [prompt_ids[rng.randrange(len(prompt_ids))] for _ in range(len(prompt_ids))]
        boots.append(mean_of(resampled))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, (lo, hi)


def paired_bootstrap_mean_diff(arm_a, arm_b, age=None, J=None, n_boot=N_BOOT, seed=BOOT_SEED):
    ga = prompt_grouped_counts(arm_a, age, J)
    gb = prompt_grouped_counts(arm_b, age, J)
    prompt_ids = sorted(ga.keys())
    assert prompt_ids == sorted(gb.keys())

    def mean_of(groups, ids):
        vals = [v for pid in ids for v in groups[pid]]
        return sum(vals) / len(vals)

    point = mean_of(ga, prompt_ids) - mean_of(gb, prompt_ids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        resampled = [prompt_ids[rng.randrange(len(prompt_ids))] for _ in range(len(prompt_ids))]
        diffs.append(mean_of(ga, resampled) - mean_of(gb, resampled))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    p_below = sum(1 for x in diffs if x <= 0) / n_boot
    p_above = sum(1 for x in diffs if x >= 0) / n_boot
    p_value = min(1.0, 2 * min(p_below, p_above))
    return {"point_diff": point, "ci95": [lo, hi], "p_value": p_value}


def holm_correct(results):
    order = sorted(range(len(results)), key=lambda i: results[i]["p_value"])
    m = len(results)
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * results[i]["p_value"]
        running_max = max(running_max, adj)
        results[i]["p_holm"] = min(1.0, running_max)
    for r in results:
        r["significant_holm_0.05"] = r["p_holm"] < 0.05
    return results


print("\n=== Mean violations/document per arm ===\n")
mean_viol_output = {"overall": {}, "per_age": {}, "per_aj": {}}

print("--- Overall (all 90) ---")
for arm in ALL_ARMS:
    point, ci = mean_and_boot_ci(prompt_grouped_counts(arm))
    mean_viol_output["overall"][arm] = {"mean": point, "ci95": list(ci)}
    print(f"  {arm:<28} mean={point:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

print("\n--- Per age band ---")
for age, lbl in AGE_LABELS:
    print(f" age {age} ({lbl}):")
    mean_viol_output["per_age"][lbl] = {}
    for arm in ALL_ARMS:
        point, ci = mean_and_boot_ci(prompt_grouped_counts(arm, age=age))
        mean_viol_output["per_age"][lbl][arm] = {"mean": point, "ci95": list(ci)}
        print(f"    {arm:<28} mean={point:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

print("\n--- Per (A,J) cell ---")
for age, _ in AGE_LABELS:
    for J in Js:
        key = f"{age},{J}"
        mean_viol_output["per_aj"][key] = {}
        cells = []
        for arm in ALL_ARMS:
            point, ci = mean_and_boot_ci(prompt_grouped_counts(arm, age=age, J=J))
            mean_viol_output["per_aj"][key][arm] = {"mean": point, "ci95": list(ci)}
            cells.append(f"{arm}={point:.2f}")
        print(f"  {key:<8}" + "  ".join(cells))

with open(REPO_ROOT / "analysis" / "phase2_final_mean_violations.json", "w", encoding="utf-8") as f:
    json.dump(mean_viol_output, f, indent=2)
print("\nWritten: analysis/phase2_final_mean_violations.json")

print("\n=== Paired mean-violation-count differences, CP-GTR vs each other arm, Holm-corrected ===\n")
bootstrap_meandiff_output = {}
for scope_name, age in [("OVERALL (all 90)", None)] + [(f"age {a}", a) for a, _ in AGE_LABELS]:
    print(f"--- {scope_name} ---")
    family = []
    for other in COMPARISON_ARMS:
        res = paired_bootstrap_mean_diff("CP-GTR", other, age=age)
        res["comparison"] = f"CP-GTR vs {other}"
        family.append(res)
    holm_correct(family)
    for r in family:
        sig = "*" if r["significant_holm_0.05"] else " "
        print(f"  {r['comparison']:<32} diff={r['point_diff']:+.3f}  "
              f"95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"p={r['p_value']:.4f}  p_holm={r['p_holm']:.4f} {sig}")
    bootstrap_meandiff_output[scope_name] = family
    print()
print("(negative diff = CP-GTR has FEWER violations, i.e. favours CP-GTR; * = significant at Holm alpha=0.05)")

with open(REPO_ROOT / "analysis" / "phase2_final_mean_violation_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump(bootstrap_meandiff_output, f, indent=2)
print("\nWritten: analysis/phase2_final_mean_violation_bootstrap.json")

# =========================================================================== #
# (b) Scoped CCR: only the codes LIB-FAITHFUL-P-v2 targets.
# =========================================================================== #
print("\n\n=== SCOPED CCR (only LIB-FAITHFUL-P-v2-targeted codes: "
      f"{TARGETED_CODES}) -- SECONDARY, post-hoc, not the headline ===\n")


def scoped_labeled_outputs_for(arm):
    out = []
    for d in scored_docs:
        if d["arm"] != arm:
            continue
        prompt_id = pairs[d["pair_id"]]["prompt_id"]
        scoped_viol = [v for v in doc_labels[(d["pair_id"], arm)] if v in TARGETED_CODES]
        out.append({"prompt_id": prompt_id, "age": d["A"], "jurisdiction": d["J"], "violations": scoped_viol})
    return out


hdr = f"{'Arm':<28}" + "".join(f"{lbl:>22}" for _, lbl in AGE_LABELS)
print(hdr); print("-" * len(hdr))
scoped_ccr = {}
for arm in ALL_ARMS:
    items = scoped_labeled_outputs_for(arm)
    row, cells = {}, []
    for age, lbl in AGE_LABELS:
        rate, (lo, hi) = compliance_rate(items, age)
        row[lbl] = {"rate_pct": rate, "ci95": [lo, hi]}
        cells.append(f"{rate:>7.1f} [{lo:.1f},{hi:.1f}]".rjust(22))
    scoped_ccr[arm] = row
    print(f"{arm:<28}" + "".join(cells))

print("\nCaveat: US_VPC/US_RETENTION/US_PARENT_REVIEW only ever apply at (age 9, jurisdiction US) -- "
      "every EU/UK item and every age-15/21 item is vacuously scored 'compliant' once scoped (no "
      "applicable targeted code = no possible violation). Only the Child column, and within it only "
      "the 9,US third of the pooled-over-J items, carries real signal; Teen/Adult columns here are "
      "artifacts of pooling, not evidence the mechanism works there -- reported in full per instruction, "
      "not pre-filtered away.")

print(f"\n--- For reference: scoped CCR restricted to (age 9, US) ONLY (n=10/arm, the only cell where "
      f"a targeted code applies) ---")
scoped_9us = {}
for arm in ALL_ARMS:
    items = [e for e in scoped_labeled_outputs_for(arm) if e["age"] == 9 and e["jurisdiction"] == "US"]
    assert len(items) == 10, (arm, len(items))
    rate, (lo, hi) = compliance_rate(items, 9)
    scoped_9us[arm] = {"rate_pct": rate, "ci95": [lo, hi]}
    print(f"  {arm:<28} {rate:>7.1f} [{lo:.1f},{hi:.1f}]")

with open(REPO_ROOT / "analysis" / "phase2_final_scoped_ccr.json", "w", encoding="utf-8") as f:
    json.dump({"targeted_codes": TARGETED_CODES, "per_age_band_pooled_over_J": scoped_ccr,
               "age9_US_only_meaningful_cell": scoped_9us}, f, indent=2)
print("\nWritten: analysis/phase2_final_scoped_ccr.json")
