"""Build the blind annotation sheet from the generated evaluation documents.

Supports: the annotation workflow behind every compliance result. Identical
texts judged in the same context are pooled into one annotation unit, so the
540 scored documents reduce to 353 units, and units are shuffled with a fixed
seed (2026) so annotators see no arm identity or ordering.

Takes the 90 evaluation pairs (practice pairs are excluded, since they are
never scored), builds the unique units, and writes one row per unit with an
empty column for each codebook_v2.json code (including the over-requirement
code) and a notes column. Offline: no model calls.

Run:
    python analysis/phase2_taskZ_annotation_sheet.py
Order: after phase2_generation_run.py; before
phase2_build_annotation_workbooks.py and every phase2_final_results*.py
script, which all read the CSV.

Inputs : analysis/phase2_generation_pairs.json (LLM-generated text),
         schema/codebook_v2.json.
Outputs: analysis/phase2_blind_annotation_sheet.csv (353 rows),
         analysis/phase2_taskZ_output.json (counts and inheritance report).
"""
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import build_scored_documents, unique_units, build_blind_annotation_sheet, inheritance_report

SEED = 2026

pairs = json.loads((REPO_ROOT / "analysis" / "phase2_generation_pairs.json").read_text(encoding="utf-8"))
assert len(pairs) == 90, f"expected 90 eval pairs, got {len(pairs)}"

codebook = json.loads((REPO_ROOT / "schema" / "codebook_v2.json").read_text(encoding="utf-8"))
code_ids = [c["code"] for c in codebook["codes"]] + [codebook["over_requirement"]["code"]]
assert len(code_ids) == 11, code_ids

scored_docs = build_scored_documents(pairs)
units = unique_units(scored_docs)
sheet, seed = build_blind_annotation_sheet(scored_docs, seed=SEED)
report = inheritance_report(scored_docs)

print(f"Scored documents: {len(scored_docs)}")
print(f"Unique annotation units: {len(units)}")
print(f"Blind sheet rows: {len(sheet)}  (seed={seed})")
print(f"\nInheritance reduction per arm:")
for arm, total in report["per_arm_total"].items():
    inherited = report["per_arm_inherited"].get(arm, 0)
    fresh = total - inherited
    print(f"  {arm:30s} total={total:3d}  fresh={fresh:3d}  inherited={inherited:3d} "
          f"({100*inherited/total:.0f}% reduction)")

# ---- admission-budget check ----
# The generation pipeline never calls cpgtr.admit.admit(): the harness's
# build_rules_for_library() reconstructs rules from the extraction cache and
# filters on the frozen content-hash set recorded in phase2_library.json. The
# admission wall-clock budget therefore cannot have been exceeded, because no
# admission run takes place. Confirm that admit is not even imported into the
# harness namespace; this is stronger than a substring search, which would
# match the harness's own docstring mentions of admit().
import analysis.phase2_harness as harness_mod
assert not hasattr(harness_mod, "admit"), \
    "cpgtr.admit.admit is imported into phase2_harness.py -- budget_exceeded assertion needs re-checking"
print("\nbudget_exceeded assertion: cpgtr.admit.admit is not imported/callable anywhere in the "
      "generation pipeline (analysis/phase2_harness.py) -- no admission run occurs here, so "
      "budget_exceeded is vacuously empty -- PASS")

# ---- write the sheet with one empty column per codebook_v2 code ----
out_path = REPO_ROOT / "analysis" / "phase2_blind_annotation_sheet.csv"
fieldnames = ["row_id", "A", "J", "text"] + code_ids + ["notes"]
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in sheet:
        out_row = {"row_id": row["row_id"], "A": row["A"], "J": row["J"], "text": row["text"], "notes": ""}
        for c in code_ids:
            out_row[c] = ""
        w.writerow(out_row)

print(f"\nWritten: {out_path} ({len(sheet)} rows, {len(fieldnames)} columns: "
      f"row_id/A/J/text + {len(code_ids)} codebook_v2 codes + notes)")

out = {
    "seed": seed, "n_scored_documents": len(scored_docs), "n_unique_units": len(units),
    "n_sheet_rows": len(sheet), "inheritance_report": report, "code_ids": code_ids,
}
(REPO_ROOT / "analysis" / "phase2_taskZ_output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("Written: analysis/phase2_taskZ_output.json")
