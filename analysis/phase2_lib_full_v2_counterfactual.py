"""Offline counterfactual for LIB-FULL-v2, by label inheritance only (2026-09-26
instruction, session_report_2026-09-26b.md). NOT a new arm, NOT new generation,
NOT new annotation -- CP-GTR-Guarded remains withdrawn (docs/REVISION_SPEC.md
sec:2.8) and no result is claimed for it. This is strictly: for each of the 90
already-generated CP-GTR-Ungated documents, would LIB-FULL-v2 (LIB-FULL minus
the 13 rules Prof. Ali recommended removing) have changed the output at all --
and where the answer is provably "no, it would have left the draft untouched,"
inherit that unit's ALREADY-EXISTING label via the exact sec:2.6 mechanism
(same text hash -> same annotation unit -> same label), not a fresh judgement.

Logic: cpgtr.rules.normalize_trace() fires every rule whose L-pattern matches
and whose Phi is true, in a stratified order, until no further rule matches.
If a document's ACTUAL recorded CP-GTR-Ungated_rules_fired (drawn from all 31
LIB-FULL rules) is a SUBSET of the 13 rules being removed, that means none of
the surviving 18 rules ever matched the host graph during the real run, even
though all 31 were available to try. Removing OTHER rules cannot cause a rule
that didn't match before to start matching now -- DPO rewriting depends only
on the current graph state, and the 18 survivors already had their chance
against the same host graph other rules were firing on. So under LIB-FULL-v2,
zero rules fire, normalize_trace() returns an empty trace immediately, and
realize() produces IDENTICAL text to the unconstrained draft (no edits to
make). This is verified below by checking text_hash equality against the
paired Unconstrained document wherever the fired set is empty (the directly-
checkable case), and reported as a stated logical consequence (not an
independent measurement) for the nonempty-but-fully-removed-subset case, where
the counterfactual text was never actually generated to hash-compare against.

Where the fired set is NOT a subset of the 13 (>=1 surviving rule fired), the
counterfactual output is text nobody has annotated -- left UNSCORED, not
estimated, interpolated, or assumed.
"""
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import build_scored_documents, propagate_labels, text_hash
from cpgtr.eval_compliance import compliance_rate

PAIRS_PATH = REPO_ROOT / "analysis" / "phase2_generation_pairs.json"
LIBRARY_PATH = REPO_ROOT / "analysis" / "phase2_library.json"
FINAL_LABELS_PATH = REPO_ROOT / "analysis" / "phase2_final_labels.json"
CSV_PATH = REPO_ROOT / "analysis" / "phase2_blind_annotation_sheet.csv"

pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
lib_data = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
REMOVED_RULE_NAMES = {r["name"] for r in lib_data["LIB-FULL-v2"]["removed_from_LIB-FULL"]}
assert len(REMOVED_RULE_NAMES) == 13

final_labels = json.loads(FINAL_LABELS_PATH.read_text(encoding="utf-8"))
csv_rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
unit_id_to_tuple = {f"U{int(r['row_id']):03d}": (text_hash(r["text"]), int(r["A"]), r["J"]) for r in csv_rows}
unit_labels = {unit_id_to_tuple[uid]: [c for c, v in row.items() if v == "yes"] for uid, row in final_labels.items()}

scored_docs = build_scored_documents(pairs)
doc_labels = propagate_labels(scored_docs, unit_labels)

# =========================================================================== #
# Classify all 90 pairs.
# =========================================================================== #
scorable = []      # (pair_id, inherited_from='Unconstrained', hash_verified: bool)
unscorable = []     # pair_id, surviving rules that fired

empty_fired_hash_checked = 0
empty_fired_hash_mismatches = []
nonempty_subset_count = 0

for pid, e in pairs.items():
    fired = set(e["_meta"]["CP-GTR-Ungated_rules_fired"])
    if fired <= REMOVED_RULE_NAMES:
        # LIB-FULL-v2 output == Unconstrained draft, by the DPO argument above.
        if not fired:
            # Directly checkable case: verify by text hash against the actual
            # recorded CP-GTR-Ungated text for this pair (also zero-fired real
            # output -- should already equal Unconstrained verbatim).
            ungated_text = e["CP-GTR-Ungated"]
            unconstrained_text = e["Unconstrained"]
            empty_fired_hash_checked += 1
            if text_hash(ungated_text) != text_hash(unconstrained_text):
                empty_fired_hash_mismatches.append(pid)
        else:
            nonempty_subset_count += 1
        scorable.append({"pair_id": pid, "fired": sorted(fired), "empty": not fired})
    else:
        surviving_fired = sorted(fired - REMOVED_RULE_NAMES)
        unscorable.append({"pair_id": pid, "fired": sorted(fired), "surviving_rules_fired": surviving_fired})

print(f"Scorable (fired set subset of the 13 removed rules): {len(scorable)}/90")
print(f"  of which fired=[] (directly text-hash-checked against actual CP-GTR-Ungated output): "
      f"{empty_fired_hash_checked}, mismatches: {len(empty_fired_hash_mismatches)} {empty_fired_hash_mismatches}")
print(f"  of which fired=nonempty-but-fully-removed-subset (counterfactual text never generated, "
      f"inferred by the DPO argument, not independently hash-verified): {nonempty_subset_count}")
print(f"Unscorable (>=1 surviving LIB-FULL-v2 rule fired -- counterfactual text nobody annotated): "
      f"{len(unscorable)}/90")
for u in unscorable:
    print(f"  {u['pair_id']}: surviving rules fired = {u['surviving_rules_fired']}")

assert not empty_fired_hash_mismatches, \
    "text-hash verification FAILED for an empty-fired pair -- the DPO argument's premise doesn't hold, stop"

# =========================================================================== #
# Score the scorable subset: LIB-FULL-v2 (inherited label = Unconstrained's
# label) vs CP-GTR-Ungated (actual, already-scored) on the SAME subset.
# =========================================================================== #
scorable_ids = {s["pair_id"] for s in scorable}
print(f"\n=== Scorable fraction: {len(scorable_ids)}/90 ({100*len(scorable_ids)/90:.1f}%) — "
      f"STATED PROMINENTLY: this is a counterfactual over PART of the sample, weaker evidence than a run ===\n")

AGE_LABELS = [(9, "Child"), (15, "Teen"), (21, "Adult")]


def labeled_outputs_libfullv2(subset_ids):
    out = []
    for pid in subset_ids:
        e = pairs[pid]
        A, J = e["context"]["A"], e["context"]["J"]
        # LIB-FULL-v2's counterfactual violations == Unconstrained's actual violations
        # (identical text -> identical unit -> identical label, sec:2.6).
        viol = doc_labels[(pid, "Unconstrained")]
        out.append({"prompt_id": e["prompt_id"], "age": A, "jurisdiction": J, "violations": viol})
    return out


def labeled_outputs_ungated(subset_ids):
    out = []
    for pid in subset_ids:
        e = pairs[pid]
        A, J = e["context"]["A"], e["context"]["J"]
        viol = doc_labels[(pid, "CP-GTR-Ungated")]
        out.append({"prompt_id": e["prompt_id"], "age": A, "jurisdiction": J, "violations": viol})
    return out


print(f"{'Arm':<24}{'Child':>22}{'Teen':>22}{'Adult':>22}   n(9,15,21)")
for label, fn in [("LIB-FULL-v2 (counterfactual)", labeled_outputs_libfullv2),
                  ("CP-GTR-Ungated (actual, same subset)", labeled_outputs_ungated)]:
    items = fn(scorable_ids)
    cells = []
    ns = []
    for age, _ in AGE_LABELS:
        age_items = [x for x in items if x["age"] == age]
        ns.append(len(age_items))
        if not age_items:
            cells.append("no docs".rjust(22))
            continue
        rate, (lo, hi) = compliance_rate(items, age)
        cells.append(f"{rate:>7.1f} [{lo:.1f},{hi:.1f}]".rjust(22))
    print(f"{label:<24}" + "".join(cells) + f"   {ns}")

# Mean violations, scorable subset only.
print()
for label, fn in [("LIB-FULL-v2 (counterfactual)", labeled_outputs_libfullv2),
                  ("CP-GTR-Ungated (actual, same subset)", labeled_outputs_ungated)]:
    items = fn(scorable_ids)
    mean_v = sum(len(x["violations"]) for x in items) / len(items)
    n_over = sum(1 for x in items if "OVER_REQUIREMENT" in x["violations"])
    print(f"{label:<38} mean violations/doc = {mean_v:.3f}   OVER_REQUIREMENT = {n_over}/{len(items)}")

# =========================================================================== #
# Write output
# =========================================================================== #
out = {
    "n_total": 90,
    "n_scorable": len(scorable_ids),
    "scorable_fraction_pct": 100 * len(scorable_ids) / 90,
    "n_unscorable": len(unscorable),
    "scorable_detail": scorable,
    "unscorable_detail": unscorable,
    "empty_fired_hash_checked": empty_fired_hash_checked,
    "empty_fired_hash_mismatches": empty_fired_hash_mismatches,
    "nonempty_subset_inferred_not_hash_verified": nonempty_subset_count,
}
with open(REPO_ROOT / "analysis" / "phase2_lib_full_v2_counterfactual.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\nWritten: analysis/phase2_lib_full_v2_counterfactual.json")
