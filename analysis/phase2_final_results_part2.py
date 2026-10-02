"""Six-arm CCR, per-code table, paired comparisons, false negatives, coverage map.

Supports: the CCR table with all six arms (adds CP-GTR-Ungated, the paper's
unreviewed-library arm), the per-code violation table, the per-(age,
jurisdiction) cells, the paired bootstrap comparisons of CP-GTR against every
other arm, the annotator-audited false-negative rate with its targeted and
untargeted split, and the code-to-rule coverage map. All from data already
collected: no new generation and no new annotation.

  1. Coverage map. A rule targets a codebook code if (i) its template's topic
     matches the code's topic (parental consent, withdrawal, retention, sale or
     sharing), classified from the rule's template text and not from its source
     file, since several rules from different sources share generic templates,
     and (ii) its context guard is true in at least one (age, jurisdiction) cell
     where the code itself applies, tested against cpgtr.admit.CONTEXTS. This is
     stricter than "the library contains a rule about the subject": a rule whose
     guard is false in every cell provides no practical coverage and is
     excluded.
  2. CCR and per-code violation counts for all six arms; CCR per (age,
     jurisdiction) cell.
  3. False-negative rate: of the CP-GTR documents that pass the round-trip
     check, the share with at least one gold violation (Wilson 95% CI), split
     into documents with a violation of a code LIB-FAITHFUL-P-v2 targets and
     documents whose violated codes are all ones the library never targets.
  4. Paired bootstrap of CCR differences, CP-GTR versus each other arm, with a
     prompt-level cluster bootstrap (5000 draws, seed 2026) and Holm-Bonferroni
     correction within each family of five comparisons (one family overall and
     one per age band).

Run (loads the frozen configuration, so it needs a configured provider key
because assert_config() builds the model client; it makes no model calls):
    python analysis/phase2_final_results_part2.py
Order: after phase2_final_results.py (reads phase2_final_labels.json); before
phase2_final_results_part3.py, which reads phase2_coverage_map.json.

Inputs : analysis/phase2_final_labels.json (adjudicated human labels, not part
         of the public release), analysis/phase2_blind_annotation_sheet.csv,
         analysis/phase2_generation_pairs.json (LLM-generated text),
         schema/codebook_v2.json, analysis/phase2_library.json,
         cache/canonical_extraction_cache.jsonl,
         analysis/realization_templates_corrections.json.
Outputs: analysis/phase2_coverage_map.json, analysis/phase2_final_ccr_6arm.json,
         analysis/phase2_final_per_code_breakdown_6arm.json,
         analysis/phase2_final_ccr_per_aj.json,
         analysis/phase2_final_false_negative_rate.json,
         analysis/phase2_final_bootstrap_comparisons.json.
"""
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_generation_lib import assert_config
from analysis.phase2_harness import build_scored_documents, propagate_labels, text_hash
from cpgtr.admit import CONTEXTS
from cpgtr.eval_compliance import compliance_rate

CODEBOOK_V2_PATH = REPO_ROOT / "schema" / "codebook_v2.json"
CSV_PATH = REPO_ROOT / "analysis" / "phase2_blind_annotation_sheet.csv"
PAIRS_PATH = REPO_ROOT / "analysis" / "phase2_generation_pairs.json"

codebook = json.loads(CODEBOOK_V2_PATH.read_text(encoding="utf-8"))
CODES = codebook["codes"] + [codebook["over_requirement"]]
# Excludes OVER_REQUIREMENT, which no rule can target in the same sense.
GATED_CODES = codebook["codes"]


def code_applies(code_entry, A, J):
    if code_entry["code"] == "OVER_REQUIREMENT":
        return True
    aw = code_entry["applies_when"]
    if aw["jurisdiction"] != "any" and aw["jurisdiction"] != J:
        return False
    ac = aw["age_condition"]
    if ac.startswith("any"):
        return True
    m = re.match(r"A\s*<\s*(\d+)", ac)
    return A < int(m.group(1))


final_labels = json.loads((REPO_ROOT / "analysis" / "phase2_final_labels.json").read_text(encoding="utf-8"))
csv_rows = __import__("csv").DictReader(open(CSV_PATH, encoding="utf-8"))
csv_rows = list(csv_rows)
unit_id_to_tuple = {f"U{int(r['row_id']):03d}": (text_hash(r["text"]), int(r["A"]), r["J"]) for r in csv_rows}
unit_labels = {unit_id_to_tuple[uid]: [c for c, v in row.items() if v == "yes"] for uid, row in final_labels.items()}

pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
scored_docs = build_scored_documents(pairs)
doc_labels = propagate_labels(scored_docs, unit_labels)
assert all((d["pair_id"], d["arm"]) in doc_labels for d in scored_docs)

cfg = assert_config()
CERTS = {"LIB-FAITHFUL-P-v2": cfg["cpgtr_cert"], "LIB-FULL": cfg["ungated_cert"]}

# =========================================================================== #
# Code-to-rule coverage map (built first: the false-negative split uses it)
# =========================================================================== #

TOPIC_KEYWORDS = {
    "parental_consent": ["parent", "guardian"],
    "withdrawal": ["withdraw"],
    "retention": ["retain", "indefinit", "limited time", "delete"],
    "sale_share": ["sell", "share", "third part", "identifier", "disclos"],
}

CODE_TOPIC = {
    "EU_PARENTAL_CONSENT": "parental_consent", "UK_PARENTAL_CONSENT": "parental_consent",
    "US_VPC": "parental_consent", "US_PARENT_REVIEW": "parental_consent",
    "EU_WITHDRAWAL": "withdrawal",
    "EU_RETENTION": "retention", "US_RETENTION": "retention",
    "US_SALE_MINOR": "sale_share",
    # No node or edge type in the type graph can state a generic "setting
    # default", or geolocation specifically, so these codes are not
    # representable.
    "UK_HIGH_PRIVACY_DEFAULT": None,
    "UK_GEOLOCATION_DEFAULT": None,
}


def rule_topic(rule):
    t = (rule.template or "").lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return topic
    return None


def rule_active_contexts(rule):
    return [ctx for ctx in CONTEXTS if rule.phi(*ctx)]


coverage_map = {}
for lib_name, cert in CERTS.items():
    coverage_map[lib_name] = {}
    for code_entry in GATED_CODES:
        code = code_entry["code"]
        topic = CODE_TOPIC[code]
        targeting_rules = []
        if topic is not None:
            code_contexts = {ctx for ctx in CONTEXTS if code_applies(code_entry, *ctx)}
            for r in cert:
                if rule_topic(r) != topic:
                    continue
                active = set(rule_active_contexts(r))
                if active & code_contexts:
                    targeting_rules.append({"rule": r.name, "active_in": sorted(active & code_contexts)})
        coverage_map[lib_name][code] = {
            "topic": topic,
            "schema_representable": topic is not None,
            "targeting_rules": targeting_rules,
            "targeted_at_all": len(targeting_rules) > 0,
        }

print("=== Code-to-rule coverage map ===\n")
for lib_name in CERTS:
    print(f"--- {lib_name} ---")
    for code_entry in GATED_CODES:
        code = code_entry["code"]
        info = coverage_map[lib_name][code]
        if not info["schema_representable"]:
            print(f"  {code:<26} NOT REPRESENTABLE in TG schema (no matching node/edge type exists)")
        elif info["targeted_at_all"]:
            names = ", ".join(r["rule"] for r in info["targeting_rules"])
            print(f"  {code:<26} targeted by: {names}")
        else:
            print(f"  {code:<26} topic={info['topic']}, but ZERO rules active in an applicable context")
    print()

with open(REPO_ROOT / "analysis" / "phase2_coverage_map.json", "w", encoding="utf-8") as f:
    json.dump(coverage_map, f, indent=2)
print("Written: analysis/phase2_coverage_map.json\n")

# =========================================================================== #
# CCR and per-code breakdown for all six arms (adds CP-GTR-Ungated).
# =========================================================================== #
ALL_ARMS = ["Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)",
            "CP-GTR-Detect", "CP-GTR", "CP-GTR-Ungated"]
AGE_LABELS = [(9, "Child"), (15, "Teen"), (21, "Adult")]


def labeled_outputs_for(arm, J_filter=None):
    out = []
    for d in scored_docs:
        if d["arm"] != arm:
            continue
        if J_filter is not None and d["J"] != J_filter:
            continue
        prompt_id = pairs[d["pair_id"]]["prompt_id"]
        out.append({"prompt_id": prompt_id, "age": d["A"], "jurisdiction": d["J"],
                    "violations": doc_labels[(d["pair_id"], arm)]})
    return out


print("=== Real CCR, six arms (CP-GTR-Ungated added) ===\n")
hdr = f"{'Arm':<28}" + "".join(f"{lbl:>22}" for _, lbl in AGE_LABELS)
print(hdr); print("-" * len(hdr))
ccr_results = {}
for arm in ALL_ARMS:
    items = labeled_outputs_for(arm)
    assert len(items) == 90, (arm, len(items))
    row, cells = {}, []
    for age, lbl in AGE_LABELS:
        rate, (lo, hi) = compliance_rate(items, age)
        row[lbl] = {"age": age, "rate_pct": rate, "ci95": [lo, hi]}
        cells.append(f"{rate:>7.1f} [{lo:.1f},{hi:.1f}]".rjust(22))
    ccr_results[arm] = row
    print(f"{arm:<28}" + "".join(cells))

with open(REPO_ROOT / "analysis" / "phase2_final_ccr_6arm.json", "w", encoding="utf-8") as f:
    json.dump(ccr_results, f, indent=2)
print("\nWritten: analysis/phase2_final_ccr_6arm.json")

print("\n=== Per-code violation counts, six arms ===\n")
hdr2 = f"{'Code':<26}" + "".join(f"{a[:14]:>15}" for a in ALL_ARMS)
print(hdr2); print("-" * len(hdr2))
per_code_arm = {}
for code_entry in CODES:
    code = code_entry["code"]
    row, cells = {}, []
    for arm in ALL_ARMS:
        n_app = n_viol = 0
        for d in scored_docs:
            if d["arm"] != arm or not code_applies(code_entry, d["A"], d["J"]):
                continue
            n_app += 1
            if code in doc_labels[(d["pair_id"], arm)]:
                n_viol += 1
        row[arm] = {"n_violations": n_viol, "n_applicable": n_app}
        cells.append(f"{n_viol}/{n_app}".rjust(15))
    per_code_arm[code] = row
    print(f"{code:<26}" + "".join(cells))

with open(REPO_ROOT / "analysis" / "phase2_final_per_code_breakdown_6arm.json", "w", encoding="utf-8") as f:
    json.dump(per_code_arm, f, indent=2)
print("\nWritten: analysis/phase2_final_per_code_breakdown_6arm.json")

# =========================================================================== #
# CCR per (age, jurisdiction) cell, six arms.
# =========================================================================== #
print("\n=== CCR per (age, jurisdiction) cell, six arms ===\n")
per_aj = {}
Js = ["US", "EU", "UK"]
hdr3 = f"{'Arm':<28}" + "".join(f"{f'{a},{j}':>14}" for a, _ in AGE_LABELS for j in Js)
print(hdr3); print("-" * len(hdr3))
for arm in ALL_ARMS:
    per_aj[arm] = {}
    cells = []
    for age, _ in AGE_LABELS:
        for J in Js:
            items = [e for e in labeled_outputs_for(arm, J_filter=J) if e["age"] == age]
            assert len(items) == 10, (arm, age, J, len(items))
            rate, (lo, hi) = compliance_rate(items, age)
            per_aj[arm][f"{age},{J}"] = {"rate_pct": rate, "ci95": [lo, hi], "n": 10}
            cells.append(f"{rate:>6.0f}".rjust(14))
    print(f"{arm:<28}" + "".join(cells))

with open(REPO_ROOT / "analysis" / "phase2_final_ccr_per_aj.json", "w", encoding="utf-8") as f:
    json.dump(per_aj, f, indent=2)
print("\nWritten: analysis/phase2_final_ccr_per_aj.json (full 95% CIs in the file; table above is point estimates only)")

# =========================================================================== #
# Annotator-audited false-negative rate with Wilson 95% CIs, split by whether
# the violated code is targeted by LIB-FAITHFUL-P-v2 at all (per the coverage
# map above).
# =========================================================================== #


def wilson_ci(k, n, z=1.959963985):
    if n == 0:
        return (None, None)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


targeted_codes = {c for c, info in coverage_map["LIB-FAITHFUL-P-v2"].items() if info["targeted_at_all"]}
untargeted_codes = {c["code"] for c in GATED_CODES} - targeted_codes

print(f"\n=== Annotator-audited false-negative rate (CP-GTR) ===")
print(f"Codes targeted by LIB-FAITHFUL-P-v2: {sorted(targeted_codes)}")
print(f"Codes NOT targeted: {sorted(untargeted_codes)}\n")

system_passed = [pid for pid, e in pairs.items() if e["_meta"]["CP-GTR_round_trip_passed"]]
fn_overall, fn_a, fn_b = 0, 0, 0
fn_docs = []
for pid in system_passed:
    viol = doc_labels[(pid, "CP-GTR")]
    if viol:
        fn_overall += 1
        fn_docs.append((pid, viol))
        if any(v in targeted_codes for v in viol):
            fn_a += 1
        if all(v in untargeted_codes for v in viol):
            fn_b += 1

n_sp = len(system_passed)
rate_overall = fn_overall / n_sp
ci_overall = wilson_ci(fn_overall, n_sp)
rate_a = fn_a / n_sp
ci_a = wilson_ci(fn_a, n_sp)
rate_b = fn_b / n_sp
ci_b = wilson_ci(fn_b, n_sp)

print(f"System-passed documents (CP-GTR round-trip compliant): {n_sp}/90")
print(f"False negatives (system-passed, gold labels non-empty): {fn_overall}/{n_sp} = {100*rate_overall:.1f}% "
      f"[Wilson 95% CI {100*ci_overall[0]:.1f}, {100*ci_overall[1]:.1f}]")
print(f"  (a) FN involving >=1 code LIB-FAITHFUL-P-v2 targets:     {fn_a}/{n_sp} = {100*rate_a:.1f}% "
      f"[{100*ci_a[0]:.1f}, {100*ci_a[1]:.1f}]")
print(f"  (b) FN involving ONLY codes the library never targets:  {fn_b}/{n_sp} = {100*rate_b:.1f}% "
      f"[{100*ci_b[0]:.1f}, {100*ci_b[1]:.1f}]")
print(f"  (check: (a)+(b) can double count documents with both a targeted and an untargeted violation "
      f"present simultaneously -- {fn_a + fn_b - fn_overall} such document(s))")

fn_output = {
    "n_system_passed": n_sp, "n_false_negative": fn_overall, "rate_pct": 100 * rate_overall,
    "ci95": [100 * ci_overall[0], 100 * ci_overall[1]],
    "split_a_targeted": {"n": fn_a, "rate_pct": 100 * rate_a, "ci95": [100 * ci_a[0], 100 * ci_a[1]]},
    "split_b_untargeted_only": {"n": fn_b, "rate_pct": 100 * rate_b, "ci95": [100 * ci_b[0], 100 * ci_b[1]]},
    "targeted_codes": sorted(targeted_codes), "untargeted_codes": sorted(untargeted_codes),
    "false_negative_documents": [{"pair_id": pid, "violations": v} for pid, v in fn_docs],
}
with open(REPO_ROOT / "analysis" / "phase2_final_false_negative_rate.json", "w", encoding="utf-8") as f:
    json.dump(fn_output, f, indent=2)
print("\nWritten: analysis/phase2_final_false_negative_rate.json")

# =========================================================================== #
# Paired bootstrap of CCR differences, CP-GTR versus each other arm, over
# prompts. This is a prompt-level cluster bootstrap: each draw resamples the 10
# prompts with replacement, not individual (prompt, context) items, because a
# prompt's 9 contexts share one drafted text and are correlated, so resampling
# items directly would treat them as independent. The same resampled prompts
# are used for both arms in a draw (paired), so the difference distribution
# reflects within-prompt correlation. Reported overall (all 90 documents) and
# per age band, the two granularities of the CCR table. Holm-Bonferroni
# correction is applied within each family of 5 comparisons (one family per age
# band and one overall).
# =========================================================================== #
N_BOOT = 5000
BOOT_SEED = 2026


def prompt_grouped_compliance(arm, age=None):
    """Return {prompt_id: [0/1, ...]}: compliant flags for this arm's items,
    optionally restricted to one age band."""
    groups = defaultdict(list)
    for d in scored_docs:
        if d["arm"] != arm:
            continue
        if age is not None and d["A"] != age:
            continue
        prompt_id = pairs[d["pair_id"]]["prompt_id"]
        compliant = 1 if not doc_labels[(d["pair_id"], arm)] else 0
        groups[prompt_id].append(compliant)
    return groups


def paired_bootstrap_diff(arm_a, arm_b, age=None, n_boot=N_BOOT, seed=BOOT_SEED):
    ga = prompt_grouped_compliance(arm_a, age)
    gb = prompt_grouped_compliance(arm_b, age)
    prompt_ids = sorted(ga.keys())
    assert prompt_ids == sorted(gb.keys())

    def rate(groups, ids):
        vals = [v for pid in ids for v in groups[pid]]
        return 100.0 * sum(vals) / len(vals)

    point = rate(ga, prompt_ids) - rate(gb, prompt_ids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        resampled = [prompt_ids[rng.randrange(len(prompt_ids))] for _ in range(len(prompt_ids))]
        diffs.append(rate(ga, resampled) - rate(gb, resampled))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    p_below = sum(1 for x in diffs if x <= 0) / n_boot
    p_above = sum(1 for x in diffs if x >= 0) / n_boot
    p_value = min(1.0, 2 * min(p_below, p_above))
    return {"point_diff_pct": point, "ci95": [lo, hi], "p_value": p_value}


def holm_correct(results):
    """results: list of dicts with 'p_value'; adds 'p_holm' and
    'significant_holm_0.05' to each in place."""
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


COMPARISON_ARMS = ["Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)",
                   "CP-GTR-Detect", "CP-GTR-Ungated"]

print("\n=== Paired bootstrap CCR differences, CP-GTR vs each other arm, over prompts ===")
print(f"(n_boot={N_BOOT}, seed={BOOT_SEED}, Holm correction within each family of 5)\n")

bootstrap_output = {}
for scope_name, age in [("OVERALL (all 90)", None)] + [(f"age {a}", a) for a, _ in AGE_LABELS]:
    print(f"--- {scope_name} ---")
    family = []
    for other in COMPARISON_ARMS:
        res = paired_bootstrap_diff("CP-GTR", other, age=age)
        res["comparison"] = f"CP-GTR vs {other}"
        family.append(res)
    holm_correct(family)
    for r in family:
        sig = "*" if r["significant_holm_0.05"] else " "
        print(f"  {r['comparison']:<32} diff={r['point_diff_pct']:+7.2f}pp  "
              f"95% CI [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
              f"p={r['p_value']:.4f}  p_holm={r['p_holm']:.4f} {sig}")
    bootstrap_output[scope_name] = family
    print()

print("(* = significant at Holm-corrected alpha=0.05; positive diff favours CP-GTR)")

with open(REPO_ROOT / "analysis" / "phase2_final_bootstrap_comparisons.json", "w", encoding="utf-8") as f:
    json.dump(bootstrap_output, f, indent=2)
print("\nWritten: analysis/phase2_final_bootstrap_comparisons.json")
