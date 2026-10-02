"""Builds LIB-FULL-v2 in analysis/phase2_library.json: LIB-FULL (31 rules)
minus the 13 unconditional rules Prof. Ali recommended removing outright
(annotation/guard_verification_workbook_filled_byAli.xlsx -- 13/13
guard_correct=yes, 13/13 recommended_disposition=remove, session_report_2026-
09-26.md). No new rules are added here -- Ali separately identified 6 of the
13's underlying BEHAVIOURS (bounded retention, consent withdrawal, no-sale-
without-opt-in) as legitimate and worth re-deriving from their correct
citations (GDPR Arts. 5(1)(e)/7(3)/13(2)(a), 16 C.F.R. Sec.312.10, Cal. Civ.
Code Sec.1798.120(c)) -- that is a separate rule-authoring task, out of scope
here, and NOT done in this script.

Non-destructive: LIB-FULL (31 rules) is left untouched and verified
byte-identical after this write, consistent with every prior library
revision this project has made (LIB-FAITHFUL-v2, LIB-FAITHFUL-P-v2).
"""
import json
import sys
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import _rows_sha256, PHASE2_LIBRARY_PATH

FILLED_XLSX = REPO_ROOT / "annotation" / "guard_verification_workbook_filled_byAli.xlsx"

wb = openpyxl.load_workbook(FILLED_XLSX, data_only=True)
ws = wb["Guard Verification"]

verdicts = {}   # content_hash -> (guard_correct, disposition, notes)
for r in range(2, ws.max_row + 1):
    rh = ws.cell(row=r, column=1).value
    if not rh:
        continue
    verdicts[rh] = {
        "guard_correct": ws.cell(row=r, column=9).value,
        "disposition": ws.cell(row=r, column=11).value,
        "notes": ws.cell(row=r, column=12).value,
    }
assert len(verdicts) == 13, len(verdicts)
assert all(v["disposition"] == "remove" for v in verdicts.values()), \
    "not all 13 are disposition=remove -- script assumes unanimous removal, check the filled workbook"

lib_data = json.loads(PHASE2_LIBRARY_PATH.read_text(encoding="utf-8"))
old_entry = lib_data["LIB-FULL"]
old_rules = old_entry["rules"]
old_sha256_before = old_entry["sha256"]
assert len(old_rules) == 31, len(old_rules)

kept, removed = [], []
for row in old_rules:
    ch = row["content_hash"]
    if ch in verdicts:
        removed.append({"name": row["name"], "content_hash": ch, "notes": verdicts[ch]["notes"]})
    else:
        kept.append(row)

assert len(removed) == 13, len(removed)
assert len(kept) == 18, len(kept)

print(f"Kept: {len(kept)}  Removed: {len(removed)}")
for r in removed:
    print(f"  REMOVE  {r['name']}")

new_sha256 = _rows_sha256(kept)

new_entry = {
    "library_id": "LIB-FULL-v2",
    "sha256": new_sha256,
    "n_rules": len(kept),
    "membership_source": "EXTERNAL EXPERT REVIEW (Prof. Ali, independent judicial expert) -- "
                          "annotation/guard_verification_workbook_filled_byAli.xlsx. All 13 "
                          "unconditional (Phi always-True) rules in LIB-FULL reviewed for a "
                          "derivable context guard; his verdict was NOT 'guard them' but 'remove "
                          "all thirteen' -- their cited UN CRC source provisions do not support "
                          "their claimed data-privacy subject at all (e.g. a retention rule sourced "
                          "from military-recruitment-age text), so no guard is legitimately "
                          "derivable and attaching one anyway would make an unfaithful rule LOOK "
                          "audited.",
    "definition": "LIB-FULL (31 rules) minus the 13 unconditional rules Ali recommended removing "
                  "outright. No replacement rules added -- see this file's own module docstring "
                  "for the separate, out-of-scope rule-authoring task his review also identified "
                  "(6 of the 13 rules' underlying behaviours are legitimate but need re-deriving "
                  "from GDPR Arts. 5(1)(e)/7(3)/13(2)(a), 16 C.F.R. Sec.312.10, Cal. Civ. Code "
                  "Sec.1798.120(c) instead of their current UN CRC citations).",
    "superseded": "LIB-FULL",
    "removed_from_LIB-FULL": removed,
    "rules": kept,
}

lib_data["LIB-FULL-v2"] = new_entry
PHASE2_LIBRARY_PATH.write_text(json.dumps(lib_data, indent=2, ensure_ascii=False), encoding="utf-8")

# Non-destructive check: LIB-FULL itself must be byte-identical to before.
reloaded = json.loads(PHASE2_LIBRARY_PATH.read_text(encoding="utf-8"))
assert reloaded["LIB-FULL"]["sha256"] == old_sha256_before, "LIB-FULL was modified!"
assert reloaded["LIB-FULL"]["n_rules"] == 31

print(f"\nWritten: LIB-FULL-v2 (n_rules={len(kept)}, sha256={new_sha256})")
print(f"Original LIB-FULL entry (31 rules, sha256={old_sha256_before[:16]}...) verified UNTOUCHED, both coexist.")
