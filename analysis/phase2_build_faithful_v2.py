"""Task S: recompute LIB-FAITHFUL (strict, supported_by_span == yes AND
guard_correct != no) against Prof. Ali's fresh fidelity-recheck verdicts.
Mirrors analysis/phase2_build_faithful_p_v2.py's logic exactly (same fresh
classification map, same source data) but with the strict yes-only filter
instead of LIB-FAITHFUL-P's lenient yes-or-partial one.

LIB-FAITHFUL's original 6 rules are ALL among the 13 rules Task N re-checked
(verified: LIB-FAITHFUL's definition is a strict subset of LIB-FAITHFUL-P's
lenient one, so every LIB-FAITHFUL rule is necessarily also a LIB-FAITHFUL-P
rule) -- no new rechecking needed, only re-filtering the same fresh verdicts.

Writes "LIB-FAITHFUL-v2" into analysis/phase2_library.json as a NEW entry;
does not touch the original "LIB-FAITHFUL" (6 rules) or "LIB-FAITHFUL-P-v2"
(already written) entries.
"""
import csv
import json
import sys
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import _rows_sha256, PHASE2_LIBRARY_PATH

FILLED_XLSX = REPO_ROOT / "docs" / "fidelity_recheck_filled__Prof_Ali.xlsx"
AUDIT_CSV = REPO_ROOT / "docs" / "fidelity_audit_Ali_filled2.csv"

# Same fresh classifications as phase2_build_faithful_p_v2.py, read from
# Prof. Ali's notes text (the xlsx dropdown only records "changed", not a
# fresh yes/no/partial value itself).
FRESH_CLASSIFICATION = {
    "3e62a311cc627fa1a3ed097e14fffdef25bd758e6d2b70645b524f4949122af6": "no",
    "e477c6fddc4b07408eb10410afc84324d517c0e099c6a34910a53edf644d116d": "no",
    "1a2490f719af18c502c5c8628292897d082710a143cd6192a1311adacf349ceb": "no",
    "9db8f19ab63934788a6bace3665dae54ed661e28a6cc9d6a5639ae271776985e": "no",
    "ddd030d02f418324db87eab951301a08b72d601557d88423bfec5c8c03a655ec": "no",
    "6b907952ebe4b4f225932dee0334b9618da9ac394c6e9899855ea9d971c32abb": "partial",
}

wb = openpyxl.load_workbook(FILLED_XLSX)
ws = wb["Fidelity Recheck"]
verdicts = {}
for r in range(2, ws.max_row + 1):
    rh = ws[f"A{r}"].value
    if rh:
        verdicts[rh] = (ws[f"G{r}"].value, ws[f"H{r}"].value or "")
assert len(verdicts) == 13

with open(AUDIT_CSV, encoding="utf-8-sig") as f:
    orig_span = {row["rule_id"]: row["supported_by_span"] for row in csv.DictReader(f)}

lib_data = json.loads(PHASE2_LIBRARY_PATH.read_text(encoding="utf-8"))
old_entry = lib_data["LIB-FAITHFUL"]
old_rules = old_entry["rules"]
assert len(old_rules) == 6

# Sanity: every LIB-FAITHFUL rule must be in the 13 re-checked rules.
missing = [r["name"] for r in old_rules if r["content_hash"] not in verdicts]
if missing:
    raise RuntimeError(f"LIB-FAITHFUL rule(s) not in the 13 re-checked rules: {missing}")

kept, dropped, disposition = [], [], {}
for row in old_rules:
    ch = row["content_hash"]
    verdict, notes = verdicts[ch]
    fresh = FRESH_CLASSIFICATION[ch] if verdict == "changed" else orig_span[ch]
    qualifies = fresh == "yes"   # STRICT: yes only, not partial
    disposition[row["name"]] = {"content_hash": ch, "fresh_classification": fresh,
                                 "verdict_still_holds": verdict, "qualifies": qualifies, "notes": notes}
    (kept if qualifies else dropped).append(row)

print(f"LIB-FAITHFUL (strict) recompute: kept={len(kept)} dropped={len(dropped)}")
for name, d in disposition.items():
    print(f"  {'KEEP' if d['qualifies'] else 'DROP':5s} {name:40s} fresh={d['fresh_classification']}")

new_sha256 = _rows_sha256(kept)
new_entry = {
    "library_id": "LIB-FAITHFUL-v2",
    "sha256": new_sha256,
    "n_rules": len(kept),
    "membership_source": "EXTERNAL EXPERT AUDIT (Prof. Ali), RE-VERIFIED AGAINST TASK I'S REWRITTEN "
                          "TEMPLATES -- docs/fidelity_recheck_filled__Prof_Ali.xlsx. Supersedes "
                          "LIB-FAITHFUL (6 rules) for any use where the rewritten templates are what "
                          "gets deployed.",
    "definition": "Same rule as LIB-FAITHFUL: supported_by_span == yes AND guard_correct != no -- "
                  "applied to fresh verdicts. Of LIB-FAITHFUL's original 6 rules, 2 "
                  "(StatuteOfLimitations_AdminAction, Provision2) were re-judged 'no' support for the "
                  "new template's claim and dropped; the other 4 are unchanged.",
    "superseded": "LIB-FAITHFUL",
    "limitation": "Only the 13 rules that were also LIB-FAITHFUL-P members were re-checked "
                   "(analysis/fidelity_recheck.csv). The other 18 of LIB-FULL's 31 rules retain their "
                   "ORIGINAL (pre-Task-I-template) fidelity verdicts -- they were never candidates for "
                   "LIB-FAITHFUL/LIB-FAITHFUL-P either way (supported_by_span was already 'no' or "
                   "'partial'-excluded for them under the strict rule, or they're simply outside both "
                   "libraries' definitions), so this doesn't affect membership here, but it IS a real "
                   "limitation on how much of LIB-FULL has been re-verified against the new templates.",
    "dropped_from_LIB-FAITHFUL": [
        {"name": name, "content_hash": d["content_hash"], "fresh_classification": d["fresh_classification"],
         "notes": d["notes"]}
        for name, d in disposition.items() if not d["qualifies"]
    ],
    "rules": kept,
}

lib_data["LIB-FAITHFUL-v2"] = new_entry
PHASE2_LIBRARY_PATH.write_text(json.dumps(lib_data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWritten: LIB-FAITHFUL-v2 (n_rules={len(kept)}, sha256={new_sha256})")
print(f"Original LIB-FAITHFUL (6 rules, sha256={old_entry['sha256'][:16]}...) left untouched.")
