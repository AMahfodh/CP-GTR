"""Build the workbook for re-annotating US_RETENTION under codebook v3.

Supports: the final label set behind the compliance results. The primary
annotator alone re-judges US_RETENTION under the revised marking criterion of
schema/codebook_v3.json, for all 50 applicable units in the full 353 (age
below 13, jurisdiction US), not only the 82-unit overlap. The other two
annotators are not re-asked, so their US_RETENTION judgements on the overlap
remain under the v2 wording (that code had no positive case from any rater, so
no wording difference could change its agreement statistic). The workbook is
blind to the annotator's earlier answer on this code, to avoid anchoring.

Row order is the order these units already have in
analysis/phase2_blind_annotation_sheet.csv (shuffle seed 2026), filtered, not
re-shuffled. The workbook uses the same conventions as the other annotation
workbooks (see phase2_build_annotation_workbooks.py) and has Annotation, Codes
(v2 and v3 criteria side by side) and About sheets.

Run:
    python analysis/phase2_build_v3_reannotation_workbook.py
Order: after phase2_build_codebook_v3.py; before phase2_final_results.py, which
applies the filled workbook.

Inputs : analysis/phase2_blind_annotation_sheet.csv (LLM-generated text),
         schema/codebook_v3.json, schema/codebook_v2.json.
Outputs: annotation/annotator_primary_v3_us_retention.xlsx (blank workbook).
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.protection import SheetProtection

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = REPO_ROOT / "analysis" / "phase2_blind_annotation_sheet.csv"
CODEBOOK_V3_PATH = REPO_ROOT / "schema" / "codebook_v3.json"
CODEBOOK_V2_PATH = REPO_ROOT / "schema" / "codebook_v2.json"
OUT_PATH = REPO_ROOT / "annotation" / "annotator_primary_v3_us_retention.xlsx"

NAVY = "1F4E78"
ORANGE = "E36C09"
WHITE = "FFFFFF"
STRIPE = "F2F2F2"

csv_rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
assert len(csv_rows) == 353, len(csv_rows)
csv_sha256 = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

codebook_v3 = json.loads(CODEBOOK_V3_PATH.read_text(encoding="utf-8"))
codebook_v3_sha256_wholefile = hashlib.sha256(CODEBOOK_V3_PATH.read_bytes()).hexdigest()
us_retention = next(c for c in codebook_v3["codes"] if c["code"] == "US_RETENTION")

# US_RETENTION-applicable rows in the full 353 (not just the 82-unit overlap):
# jurisdiction US and age < 13, per the codebook's own applies_when.
rows = [r for r in csv_rows if r["J"] == "US" and int(r["A"]) < 13]
assert len(rows) == 50, len(rows)

# ------------------------------------------------------------------ styling
header_font = Font(name="Arial", size=11, bold=True, color=WHITE)
header_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
data_font = Font(name="Arial", size=10, bold=False)
data_align_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
data_align_nowrap = Alignment(horizontal="left", vertical="top", wrap_text=False)
data_align_center = Alignment(horizontal="center", vertical="top", wrap_text=False)
thin = Side(style="thin", color="D9D9D9")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
ref_fill = PatternFill("solid", fgColor=NAVY)
judge_fill = PatternFill("solid", fgColor=ORANGE)
stripe_fill = PatternFill("solid", fgColor=STRIPE)

REF_COLS = [
    ("unit_id", 14, data_align_nowrap),
    ("reader_age", 11, data_align_center),
    ("jurisdiction", 12, data_align_center),
    ("document_text", 100, data_align_wrap),
]

wb = openpyxl.Workbook()
wb.remove(wb.active)

# ------------------------------------------------------------------ Annotation
ws = wb.create_sheet("Annotation")
for i, (header, width, _align) in enumerate(REF_COLS, start=1):
    col = get_column_letter(i)
    cell = ws[f"{col}1"]
    cell.value = header
    cell.font, cell.alignment, cell.fill = header_font, header_align, ref_fill
    cell.protection = Protection(locked=True)
    ws.column_dimensions[col].width = width

us_ret_col = get_column_letter(len(REF_COLS) + 1)
cell = ws[f"{us_ret_col}1"]
cell.value = "US_RETENTION"
cell.font, cell.alignment, cell.fill = header_font, header_align, judge_fill
cell.protection = Protection(locked=True)
cell.comment = Comment(
    "codebook_v3 (revised 2026-09-25):\n" + us_retention["marking_criterion"] +
    "\n\nApplies when: jurisdiction=US, age_condition=A < 13" +
    f"\nCitation: {us_retention['citation']}" +
    "\n\nCHANGED from codebook_v2: v2 only caught an affirmative 'indefinite retention' "
    "claim; v3 also catches a document that omits any stated retention period.",
    "codebook_v3",
)
ws.column_dimensions[us_ret_col].width = 16

notes_col = get_column_letter(len(REF_COLS) + 2)
cell = ws[f"{notes_col}1"]
cell.value = "notes"
cell.font, cell.alignment, cell.fill = header_font, header_align, judge_fill
cell.protection = Protection(locked=True)
ws.column_dimensions[notes_col].width = 40
ws.row_dimensions[1].height = 40

r_idx = 2
for stripe_i, row in enumerate(rows):
    unit_id = f"U{int(row['row_id']):03d}"
    values = [unit_id, int(row["A"]), row["J"], row["text"]]
    stripe = stripe_fill if stripe_i % 2 == 1 else None
    max_lines = 1
    for c_idx, (val, (_h, width, align)) in enumerate(zip(values, REF_COLS), start=1):
        col = get_column_letter(c_idx)
        cell = ws[f"{col}{r_idx}"]
        cell.value = val
        cell.font, cell.alignment, cell.protection, cell.border = data_font, align, Protection(locked=True), border
        if stripe:
            cell.fill = stripe
        if c_idx == 4 and val:
            chars_per_line = max(10, width - 2)
            max_lines = max(max_lines, math.ceil(len(str(val)) / chars_per_line))

    cell = ws[f"{us_ret_col}{r_idx}"]
    cell.value = None
    cell.font, cell.alignment, cell.protection, cell.border = data_font, data_align_center, Protection(locked=False), border
    if stripe:
        cell.fill = stripe

    cell = ws[f"{notes_col}{r_idx}"]
    cell.value = None
    cell.font, cell.alignment, cell.protection, cell.border = data_font, data_align_wrap, Protection(locked=False), border
    if stripe:
        cell.fill = stripe

    ws.row_dimensions[r_idx].height = max(40, max_lines * 14 + 10)
    r_idx += 1

last_row = r_idx - 1
dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True, showDropDown=False)
dv.error = "Choose one of: yes, no"
dv.errorTitle = "Invalid entry"
dv.prompt = "yes / no"
dv.promptTitle = "US_RETENTION (codebook_v3)?"
ws.add_data_validation(dv)
dv.add(f"{us_ret_col}2:{us_ret_col}{last_row}")

ws.freeze_panes = "E2"
ws.protection = SheetProtection(
    sheet=True, password=None,
    selectLockedCells=False, selectUnlockedCells=False,
    insertRows=True, insertHyperlinks=True, autoFilter=True,
    deleteColumns=True, insertColumns=True, pivotTables=True, deleteRows=True,
    formatCells=False, formatColumns=False, formatRows=False, sort=False,
)

# ------------------------------------------------------- Codes (US_RETENTION v2/v3)
codes_ws = wb.create_sheet("Codes")
headers = ["code_id", "applies_when", "marking_criterion (v3, CURRENT)", "marking_criterion (v2, for reference)", "citation"]
widths = [16, 40, 70, 70, 20]
for i, (h, w) in enumerate(zip(headers, widths), start=1):
    col = get_column_letter(i)
    cell = codes_ws[f"{col}1"]
    cell.value = h
    cell.font, cell.alignment, cell.fill = header_font, header_align, ref_fill
    cell.protection = Protection(locked=True)
    codes_ws.column_dimensions[col].width = w
codes_ws.row_dimensions[1].height = 22

codebook_v2 = json.loads(CODEBOOK_V2_PATH.read_text(encoding="utf-8"))
us_retention_v2 = next(c for c in codebook_v2["codes"] if c["code"] == "US_RETENTION")
row_vals = [
    "US_RETENTION",
    f"jurisdiction={us_retention['applies_when']['jurisdiction']}, age_condition={us_retention['applies_when']['age_condition']}",
    us_retention["marking_criterion"],
    us_retention_v2["marking_criterion"],
    us_retention["citation"],
]
for c_idx, (val, w) in enumerate(zip(row_vals, widths), start=1):
    col = get_column_letter(c_idx)
    cell = codes_ws[f"{col}2"]
    cell.value = val
    cell.font, cell.alignment, cell.protection, cell.border = data_font, data_align_wrap, Protection(locked=True), border
codes_ws.row_dimensions[2].height = 90
codes_ws.freeze_panes = "A2"
codes_ws.protection = SheetProtection(sheet=True, password=None, selectLockedCells=False, selectUnlockedCells=False)

# ------------------------------------------------------------------ About
about = wb.create_sheet("About")
about.column_dimensions["A"].width = 30
about.column_dimensions["B"].width = 100
label_font = Font(name="Arial", size=10, bold=True)
value_font = Font(name="Arial", size=10, bold=False)
value_align = Alignment(horizontal="left", vertical="top", wrap_text=True)
label_align = Alignment(horizontal="left", vertical="top")

about_rows = [
    ("Purpose", "Re-annotation of US_RETENTION ONLY, under codebook_v3.json's revised "
                 "marking_criterion (Decision 2, session_report_2026-09-25c.md). Covers all 50 "
                 "US_RETENTION-applicable units in the full 353 (age < 13, jurisdiction US) -- "
                 "not just the 82-unit overlap sample."),
    ("Who", "Prof. Ali ONLY. Prof. Ahmed and Prof. Eid are NOT re-asked -- their US_RETENTION "
            "judgements on the 82-unit overlap remain on record under v2 wording. This is "
            "disclosed, not hidden: agreement for US_RETENTION on the overlap is therefore "
            "measured under v2 wording for all three raters, though it was degenerate anyway (all "
            "three marked 0/15 -- zero variance -- so no wording difference could have changed the "
            "pre-adjudication agreement statistic for this one code)."),
    ("Why this code changed", "codebook_v2's US_RETENTION wording caught only an AFFIRMATIVE claim "
                                "of indefinite retention, not a document that simply omits any stated "
                                "retention period. Evidenced twice, independently: "
                                "session_report_2026-09-24c.md's Task 2 (9/50, 18%, of applicable "
                                "documents mention retention/deletion but state no period) and Prof. "
                                "Eid's note on unit U021 (session_report_2026-09-25b.md), which raised "
                                "the identical point re: 16 C.F.R. Sec.312.10 independently."),
    ("codebook_v3.json SHA-256 (whole file)", codebook_v3_sha256_wholefile),
    ("codebook_v3.json embedded sha256 field", codebook_v3["sha256"] +
     " (basis: {codes, over_requirement} only, not a whole-file hash)"),
    ("phase2_blind_annotation_sheet.csv SHA-256 (source of unit_id/text/order)", csv_sha256),
    ("Row order", "The same order these 50 units already have in "
                   "phase2_blind_annotation_sheet.csv (seed=2026's shuffle) -- filtered to US_RETENTION-"
                   "applicable rows only, NOT re-shuffled."),
    ("Blind to prior answer", "Yes, deliberately -- Prof. Ali's own v2-wording answer for this code is "
                                "not shown here, to avoid anchoring the re-judgement to the earlier one, "
                                "matching every other annotation sheet's blindness convention on this "
                                "project."),
    ("Units", "50"),
]
for i, (label, value) in enumerate(about_rows, start=1):
    a, b = about[f"A{i}"], about[f"B{i}"]
    a.value, b.value = label, value
    a.font, b.font = label_font, value_font
    a.alignment, b.alignment = label_align, value_align
    lines = math.ceil(len(str(value)) / 98) if value else 1
    about.row_dimensions[i].height = max(18, lines * 14 + 8)

wb.save(OUT_PATH)
print(f"Written: {OUT_PATH}")
print(f"Annotation sheet: {last_row - 1} data rows, {ws.max_column} columns")
print(f"codebook_v3.json embedded sha256: {codebook_v3['sha256']}")
