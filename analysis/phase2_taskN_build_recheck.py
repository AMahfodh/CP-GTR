"""Build the fidelity re-check sheet for the 13 rules of LIB-FAITHFUL-P.

Supports: the rule-fidelity review (fidelity table) and the derivation of the
reviewed deployed library (LIB-FAITHFUL-P-v2).

The first fidelity review judged each rule's original, drafter-style template
text against its source provision. After the templates were rewritten as
reader-facing notice sentences (analysis/realization_templates_corrections.json),
the 13 rules of LIB-FAITHFUL-P need a fresh judgement against the new text.
This script writes analysis/fidelity_recheck.csv: one row per rule, pairing the
old and the new template with the reviewer's original supported_by_span
verdict, in shuffled order (seed 7), with empty columns for the re-check
verdict and notes. Offline: no model calls.

Run:
    python analysis/phase2_taskN_build_recheck.py
Order: after the template corrections exist; before
phase2_build_fidelity_recheck_xlsx.py (which turns the CSV into the reviewer
workbook) and phase2_build_faithful_p_v2.py.

Inputs : analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl,
         analysis/realization_templates_corrections.json (rule templates),
         docs/fidelity_audit_Ali_filled2.csv (human annotation data: the first
         fidelity review; not part of the public release).
Outputs: analysis/fidelity_recheck.csv.
"""
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import build_rules_for_library

SEED = 7  # seed for row-order randomisation (the first audit sheet used seed 1)

rules, _ = build_rules_for_library("LIB-FAITHFUL-P")
faithful_p_ids = {r.content_hash() for r in rules}
assert len(faithful_p_ids) == 13

corrections = json.loads(
    (REPO_ROOT / "analysis" / "realization_templates_corrections.json").read_text(encoding="utf-8"))

with open(REPO_ROOT / "docs" / "fidelity_audit_Ali_filled2.csv", encoding="utf-8-sig") as f:
    audit_rows = {row["rule_id"]: row for row in csv.DictReader(f)}

rows = []
for rh in faithful_p_ids:
    audit = audit_rows.get(rh)
    if audit is None:
        raise KeyError(f"LIB-FAITHFUL-P rule {rh} not found in fidelity_audit_Ali_filled2.csv")
    rows.append({
        "rule_id": rh,
        "source_file": audit["source_file"],
        "source_provision_text": audit["source_provision_text"],
        "old_template": audit["obligation_template"],
        "new_template": corrections[rh],
        "supported_by_span": audit["supported_by_span"],
        "verdict_still_holds": "",
        "notes": "",
    })

assert len(rows) == 13
rng = random.Random(SEED)
rng.shuffle(rows)

out_path = REPO_ROOT / "analysis" / "fidelity_recheck.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["rule_id", "source_file", "source_provision_text",
                                       "old_template", "new_template", "supported_by_span",
                                       "verdict_still_holds", "notes"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path} (seed={SEED})")
for r in rows:
    print(f"  {r['rule_id'][:16]}... old_verdict={r['supported_by_span']}")
