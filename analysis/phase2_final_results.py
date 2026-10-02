"""Inter-annotator agreement, the final label set, CCR, and the per-code table.

Supports: Fleiss' kappa and the pairwise Cohen's kappa on the overlap, the
adjudicated final labels, the Contextual Compliance Rate table (five arms
here; phase2_final_results_part2.py adds the sixth) and the per-code violation
table.

  (a) Agreement. Three pairwise Cohen's kappa and Fleiss' kappa across the
      three annotators on the 82 overlap units, before adjudication, using the
      v2 wording of every code (including US_RETENTION). Reports observed
      agreement, a prevalence-adjusted statistic (Randolph's free-marginal
      kappa, equal to 2*Po - 1 for binary categories), and flags codes whose
      marginals are degenerate (all raters give the same single category).
      There the standard kappa is 0/0 although observed agreement is 1, and
      the free-marginal statistic remains defined.
  (b) Final labels. The primary annotator's labels for all 353 units, with the
      majority vote of the three annotators replacing them on the 82 overlap
      units for the ten unchanged codes, and the primary annotator's v3
      US_RETENTION answer replacing the v2 answer on all units (the v3 overlay
      is applied last and so takes precedence over the majority vote for that
      code). The other two annotators never answered under the v3 wording, and
      a vote mixing v2 and v3 answers would not answer a single question.
  (c) CCR. The Contextual Compliance Rate and a bootstrap CI per arm and age
      band, from cpgtr.eval_compliance.compliance_rate(), which scores
      already-labelled outputs and does not generate or judge anything.

Run (after the three annotators and the re-annotation workbook are filled in):
    python analysis/phase2_final_results.py
Order: after phase2_build_annotation_workbooks.py and
phase2_build_v3_reannotation_workbook.py; before phase2_final_results_part2.py
and phase2_final_results_part3.py, which read phase2_final_labels.json.

Inputs : the four filled annotation workbooks in annotation/ (primary
         annotator, the two overlap annotators, and the primary annotator's v3
         US_RETENTION re-annotation; the exact file names are the *_PATH
         constants below). They are human annotation data and are not part of
         the public release.
         Also:
         annotation/overlap_sample.json, schema/codebook_v2.json,
         analysis/phase2_blind_annotation_sheet.csv (LLM-generated text),
         analysis/phase2_generation_pairs.json (LLM-generated text).
Outputs: analysis/phase2_taskA_kappa.json, analysis/phase2_final_labels.json,
         analysis/phase2_final_ccr.json,
         analysis/phase2_final_per_code_breakdown.json.
"""
import json
import sys
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase2_harness import build_scored_documents, propagate_labels, text_hash
from cpgtr.eval_compliance import compliance_rate

CODEBOOK_V2_PATH = REPO_ROOT / "schema" / "codebook_v2.json"
CSV_PATH = REPO_ROOT / "analysis" / "phase2_blind_annotation_sheet.csv"
PAIRS_PATH = REPO_ROOT / "analysis" / "phase2_generation_pairs.json"
OVERLAP_SAMPLE_PATH = REPO_ROOT / "annotation" / "overlap_sample.json"

ALI_PRIMARY_PATH = REPO_ROOT / "annotation" / "annotator_primary_filled_by_prof_Ali.xlsx"
AHMED_OVERLAP_PATH = REPO_ROOT / "annotation" / "annotator_overlap_filled_by_colleague_Ahmed.xlsx"
EID_OVERLAP_PATH = REPO_ROOT / "annotation" / "annotator_overlap_filled_by_colleague_Eid.xlsx"
ALI_V3_PATH = REPO_ROOT / "annotation" / "annotator_primary_v3_us_retention_filled_by_ProfAli.xlsx"

codebook = json.loads(CODEBOOK_V2_PATH.read_text(encoding="utf-8"))
CODES = codebook["codes"] + [codebook["over_requirement"]]
CODE_IDS = [c["code"] for c in CODES]
assert len(CODE_IDS) == 11

overlap_sample = json.loads(OVERLAP_SAMPLE_PATH.read_text(encoding="utf-8"))
OVERLAP_UNIT_IDS = set(overlap_sample["sampled_unit_ids"])
assert len(OVERLAP_UNIT_IDS) == 82


def read_annotation_sheet(path, n_code_cols=11):
    """Return {unit_id: {"A": int, "J": str, "text": str, codes...:
    'yes'/'no'/'n-a'/None}} for one filled annotation workbook."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Annotation"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    assert headers[:4] == ["unit_id", "reader_age", "jurisdiction", "document_text"], headers[:4]
    code_headers = headers[4:4 + n_code_cols]
    out = {}
    for r in range(2, ws.max_row + 1):
        unit_id = ws.cell(row=r, column=1).value
        A = ws.cell(row=r, column=2).value
        J = ws.cell(row=r, column=3).value
        text = ws.cell(row=r, column=4).value
        rec = {"A": int(A), "J": J, "text": text}
        for i, code in enumerate(code_headers):
            rec[code] = ws.cell(row=r, column=5 + i).value
        out[unit_id] = rec
    return out


# v2 wording throughout: 353 units from the primary annotator, 82 from each
# overlap annotator, all 11 codes.
ali_v2 = read_annotation_sheet(ALI_PRIMARY_PATH)
ahmed_v2 = read_annotation_sheet(AHMED_OVERLAP_PATH)
eid_v2 = read_annotation_sheet(EID_OVERLAP_PATH)
assert len(ali_v2) == 353, len(ali_v2)
assert set(ahmed_v2) == OVERLAP_UNIT_IDS == set(eid_v2)

# US_RETENTION-only v3 re-annotation (50 units, primary annotator only).
ali_v3_wb = openpyxl.load_workbook(ALI_V3_PATH, data_only=True)
ali_v3_ws = ali_v3_wb["Annotation"]
ali_v3_us_retention = {}
for r in range(2, ali_v3_ws.max_row + 1):
    unit_id = ali_v3_ws.cell(row=r, column=1).value
    val = ali_v3_ws.cell(row=r, column=5).value
    ali_v3_us_retention[unit_id] = val
assert len(ali_v3_us_retention) == 50

print(f"Loaded: Ali v2 (353 units), Ahmed v2 (82), Eid v2 (82), Ali v3 US_RETENTION (50).")

# =========================================================================== #
# (a) Pairwise Cohen's kappa and Fleiss' kappa on the 82 overlap units, before
#     adjudication, with v2 wording throughout.
# =========================================================================== #


def cohens_kappa(pairs):
    """pairs: list of (rater1_val, rater2_val) over applicable items only.
    Returns (Po, kappa_or_None, pabak). kappa is None if Pe == 1, the
    degenerate case where one category has all the marginal mass."""
    n = len(pairs)
    if n == 0:
        return None, None, None
    agree = sum(1 for a, b in pairs if a == b)
    Po = agree / n
    cats = set(x for pair in pairs for x in pair)
    p1 = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    p2 = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    Pe = sum(p1[c] * p2[c] for c in cats)
    kappa = None if Pe >= 1.0 else (Po - Pe) / (1 - Pe)
    pabak = 2 * Po - 1  # Randolph free-marginal (Brennan-Prediger) kappa, binary
    return Po, kappa, pabak


def fleiss_kappa(rows):
    """rows: list of {category: count} per item, each item rated by the same
    number of raters. Returns (Po, kappa_or_None, free_marginal)."""
    n_items = len(rows)
    if n_items == 0:
        return None, None, None
    n_raters = sum(rows[0].values())
    cats = sorted(set(c for row in rows for c in row))
    P_i = []
    for row in rows:
        s = sum(row.get(c, 0) * (row.get(c, 0) - 1) for c in cats)
        P_i.append(s / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items
    total_ratings = n_items * n_raters
    p_j = {c: sum(row.get(c, 0) for row in rows) / total_ratings for c in cats}
    Pe_bar = sum(p_j[c] ** 2 for c in cats)
    kappa = None if Pe_bar >= 1.0 else (P_bar - Pe_bar) / (1 - Pe_bar)
    # The free-marginal statistic assumes at least two possible categories, even
    # when only one was observed.
    q = len(cats) if len(cats) > 1 else 2
    free_marginal = (P_bar - 1 / q) / (1 - 1 / q)
    return P_bar, kappa, free_marginal


def code_applies(code_entry, A, J):
    if code_entry["code"] == "OVER_REQUIREMENT":
        return True
    aw = code_entry["applies_when"]
    if aw["jurisdiction"] != "any" and aw["jurisdiction"] != J:
        return False
    age_cond = aw["age_condition"]
    if age_cond.startswith("any"):
        return True
    import re
    m = re.match(r"A\s*<\s*(\d+)", age_cond)
    return A < int(m.group(1))


per_code_results = {}
pooled_pairs = {"ali_ahmed": [], "ali_eid": [], "ahmed_eid": []}
pooled_fleiss_rows = []

for code_entry in CODES:
    code = code_entry["code"]
    pairs_aa, pairs_ae, pairs_he = [], [], []
    fleiss_rows = []
    for uid in sorted(OVERLAP_UNIT_IDS, key=lambda x: int(x[1:])):
        A, J = ali_v2[uid]["A"], ali_v2[uid]["J"]
        if not code_applies(code_entry, A, J):
            continue
        a_val, h_val, e_val = ali_v2[uid][code], ahmed_v2[uid][code], eid_v2[uid][code]
        assert a_val in ("yes", "no") and h_val in ("yes", "no") and e_val in ("yes", "no"), \
            (uid, code, a_val, h_val, e_val)
        pairs_aa.append((a_val, h_val))
        pairs_ae.append((a_val, e_val))
        pairs_he.append((h_val, e_val))
        fleiss_rows.append({"yes": [a_val, h_val, e_val].count("yes"),
                             "no": [a_val, h_val, e_val].count("no")})
        pooled_pairs["ali_ahmed"].append((a_val, h_val))
        pooled_pairs["ali_eid"].append((a_val, e_val))
        pooled_pairs["ahmed_eid"].append((h_val, e_val))
        pooled_fleiss_rows.append({"yes": [a_val, h_val, e_val].count("yes"),
                                    "no": [a_val, h_val, e_val].count("no")})

    n = len(pairs_aa)
    res = {"n_applicable": n}
    for label, pairs in (("Ali_vs_Ahmed", pairs_aa), ("Ali_vs_Eid", pairs_ae), ("Ahmed_vs_Eid", pairs_he)):
        Po, kappa, pabak = cohens_kappa(pairs)
        res[label] = {"Po": Po, "kappa": kappa, "free_marginal_kappa": pabak,
                      "degenerate": (kappa is None and n > 0)}
    P_bar, fk, fmk = fleiss_kappa(fleiss_rows) if n > 0 else (None, None, None)
    res["Fleiss_all3"] = {"Po": P_bar, "kappa": fk, "free_marginal_kappa": fmk,
                          "degenerate": (fk is None and n > 0)}
    per_code_results[code] = res

# Pooled (all codes combined)
POOLED_LABELS = {"ali_ahmed": "Ali_vs_Ahmed", "ali_eid": "Ali_vs_Eid", "ahmed_eid": "Ahmed_vs_Eid"}
pooled = {}
for key, pairs in pooled_pairs.items():
    Po, kappa, pabak = cohens_kappa(pairs)
    pooled[POOLED_LABELS[key]] = {"Po": Po, "kappa": kappa, "free_marginal_kappa": pabak, "n_applicable": len(pairs)}
P_bar, fk, fmk = fleiss_kappa(pooled_fleiss_rows)
pooled["Fleiss_all3"] = {"Po": P_bar, "kappa": fk, "free_marginal_kappa": fmk, "n_applicable": len(pooled_fleiss_rows)}

print("\n=== (a) Pre-adjudication agreement, 82 overlap units, v2 wording ===\n")
print(f"{'Code':<26}{'n':>4} {'AliAhmed Po/k/fmk':>26} {'AliEid Po/k/fmk':>26} {'AhmedEid Po/k/fmk':>26} {'Fleiss Po/k/fmk':>26}")
for code, res in per_code_results.items():
    def fmt(d):
        k = "UNDEF" if d["kappa"] is None else f"{d['kappa']:.2f}"
        return f"{d['Po']:.2f}/{k}/{d['free_marginal_kappa']:.2f}"
    print(f"{code:<26}{res['n_applicable']:>4} {fmt(res['Ali_vs_Ahmed']):>26} {fmt(res['Ali_vs_Eid']):>26} "
          f"{fmt(res['Ahmed_vs_Eid']):>26} {fmt(res['Fleiss_all3']):>26}")
print(f"\n{'POOLED (all codes)':<26}{pooled['Ali_vs_Ahmed']['n_applicable']:>4} "
      f"{pooled['Ali_vs_Ahmed']['Po']:.3f}/{pooled['Ali_vs_Ahmed']['kappa']:.3f}/{pooled['Ali_vs_Ahmed']['free_marginal_kappa']:.3f}  "
      f"{pooled['Ali_vs_Eid']['Po']:.3f}/{pooled['Ali_vs_Eid']['kappa']:.3f}/{pooled['Ali_vs_Eid']['free_marginal_kappa']:.3f}  "
      f"{pooled['Ahmed_vs_Eid']['Po']:.3f}/{pooled['Ahmed_vs_Eid']['kappa']:.3f}/{pooled['Ahmed_vs_Eid']['free_marginal_kappa']:.3f}  "
      f"{pooled['Fleiss_all3']['Po']:.3f}/{pooled['Fleiss_all3']['kappa']:.3f}/{pooled['Fleiss_all3']['free_marginal_kappa']:.3f}")

degenerate_codes = [c for c, r in per_code_results.items() if r["Fleiss_all3"]["degenerate"]]
print(f"\nDegenerate/undefined codes (zero marginal variance across all 3 raters): {degenerate_codes}")

with open(REPO_ROOT / "analysis" / "phase2_taskA_kappa.json", "w", encoding="utf-8") as f:
    json.dump({"per_code": per_code_results, "pooled": pooled, "degenerate_codes": degenerate_codes}, f, indent=2)
print("\nWritten: analysis/phase2_taskA_kappa.json")

# =========================================================================== #
# (b) Final label set: the primary annotator's v2 labels for all 353 units, the
#     majority vote replacing them on the 82 overlap units for the ten unchanged
#     codes, and the v3 US_RETENTION answer replacing the v2 answer on all 353
#     units. The v3 overlay is applied last, so it also overrides the majority
#     vote for that code on the overlap units (see the module docstring).
# =========================================================================== #


def majority(a, h, e):
    votes = [a, h, e]
    return "yes" if votes.count("yes") > votes.count("no") else "no"


final_labels = {}   # unit_id -> {code: 'yes'/'no'/'n-a'}
overlay_majority_count = 0
overlay_v3_count = 0
majority_changed_from_ali = 0

for uid, rec in ali_v2.items():
    A, J = rec["A"], rec["J"]
    row = {}
    for code_entry in CODES:
        code = code_entry["code"]
        if not code_applies(code_entry, A, J):
            row[code] = "n-a"
            continue
        val = rec[code]   # base: the primary annotator's v2 answer
        if uid in OVERLAP_UNIT_IDS and code != "US_RETENTION":
            maj = majority(ali_v2[uid][code], ahmed_v2[uid][code], eid_v2[uid][code])
            if maj != val:
                majority_changed_from_ali += 1
            val = maj
            overlay_majority_count += 1
        if code == "US_RETENTION" and uid in ali_v3_us_retention:
            val = ali_v3_us_retention[uid]
            overlay_v3_count += 1
        row[code] = val
    final_labels[uid] = row

print(f"\n=== (b) Final label set built: {len(final_labels)} units x {len(CODE_IDS)} codes ===")
print(f"  majority-vote overlay applied: {overlay_majority_count} (unit,code) cells "
      f"({majority_changed_from_ali} of which changed the value from Ali's own v2 answer)")
print(f"  v3 US_RETENTION overlay applied: {overlay_v3_count} (unit,code) cells (all 50 applicable units)")

with open(REPO_ROOT / "analysis" / "phase2_final_labels.json", "w", encoding="utf-8") as f:
    json.dump(final_labels, f, indent=2)
print("Written: analysis/phase2_final_labels.json")

# =========================================================================== #
# (c) CCR table (tab:ccr) via cpgtr.eval_compliance.compliance_rate(), for five
#     arms, using the final label set above.
# =========================================================================== #
csv_rows_raw = __import__("csv").DictReader(open(CSV_PATH, encoding="utf-8"))
csv_rows_raw = list(csv_rows_raw)
unit_id_to_tuple = {}   # unit_id -> (text_hash, A, J)
for row in csv_rows_raw:
    uid = f"U{int(row['row_id']):03d}"
    unit_id_to_tuple[uid] = (text_hash(row["text"]), int(row["A"]), row["J"])
assert set(unit_id_to_tuple) == set(final_labels), \
    (set(final_labels) - set(unit_id_to_tuple), set(unit_id_to_tuple) - set(final_labels))

unit_labels = {}   # (text_hash, A, J) -> [violation codes]
for uid, row in final_labels.items():
    tup = unit_id_to_tuple[uid]
    unit_labels[tup] = [code for code, val in row.items() if val == "yes"]

pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
scored_docs = build_scored_documents(pairs)
assert len(set(d["unit"] for d in scored_docs)) == 353

doc_labels = propagate_labels(scored_docs, unit_labels)   # {(pair_id, arm): violations}
missing = [d for d in scored_docs if (d["pair_id"], d["arm"]) not in doc_labels]
assert not missing, f"{len(missing)} scored documents have no label -- e.g. {missing[:3]}"

CCR_ARMS = ["Unconstrained", "Few-Shot Prompted", "Constitutional AI (RLAIF)", "CP-GTR-Detect", "CP-GTR"]
AGE_LABELS = [(9, "Child"), (15, "Teen"), (21, "Adult")]

ccr_results = {}
print("\n=== (c) Real CCR (Table 5 / tab:ccr) ===\n")
hdr = f"{'Arm':<28}" + "".join(f"{lbl:>22}" for _, lbl in AGE_LABELS)
print(hdr); print("-" * len(hdr))
for arm in CCR_ARMS:
    labeled_outputs = []
    for d in scored_docs:
        if d["arm"] != arm:
            continue
        prompt_id = pairs[d["pair_id"]]["prompt_id"]
        labeled_outputs.append({"prompt_id": prompt_id, "age": d["A"], "jurisdiction": d["J"],
                                 "violations": doc_labels[(d["pair_id"], arm)]})
    assert len(labeled_outputs) == 90, (arm, len(labeled_outputs))
    row = {}
    cells = []
    for age, lbl in AGE_LABELS:
        rate, (lo, hi) = compliance_rate(labeled_outputs, age)
        row[lbl] = {"age": age, "rate_pct": rate, "ci95": [lo, hi]}
        cells.append(f"{rate:>7.1f} [{lo:.1f},{hi:.1f}]".rjust(22))
    ccr_results[arm] = row
    print(f"{arm:<28}" + "".join(cells))

with open(REPO_ROOT / "analysis" / "phase2_final_ccr.json", "w", encoding="utf-8") as f:
    json.dump(ccr_results, f, indent=2)
print("\nWritten: analysis/phase2_final_ccr.json")

# --------------------------------------------------------------------------- #
# Per-code violation counts per arm. This table is needed to interpret the CCR
# grid above: CP-GTR has the same compliant/non-compliant status as
# CP-GTR-Detect on all 90 documents even though its text differs on most of
# them, and the per-code counts show where the violations remain.
# --------------------------------------------------------------------------- #
print("\n=== Per-code violation counts (n_violations/n_applicable), all 90 docs per arm ===\n")
hdr2 = f"{'Code':<26}" + "".join(f"{a[:15]:>16}" for a in CCR_ARMS)
print(hdr2); print("-" * len(hdr2))
per_code_arm = {}
for code_entry in CODES:
    code = code_entry["code"]
    row = {}
    cells = []
    for arm in CCR_ARMS:
        n_app = n_viol = 0
        for d in scored_docs:
            if d["arm"] != arm or not code_applies(code_entry, d["A"], d["J"]):
                continue
            n_app += 1
            if code in doc_labels[(d["pair_id"], arm)]:
                n_viol += 1
        row[arm] = {"n_violations": n_viol, "n_applicable": n_app}
        cells.append(f"{n_viol}/{n_app}".rjust(16))
    per_code_arm[code] = row
    print(f"{code:<26}" + "".join(cells))

with open(REPO_ROOT / "analysis" / "phase2_final_per_code_breakdown.json", "w", encoding="utf-8") as f:
    json.dump(per_code_arm, f, indent=2)
print("\nWritten: analysis/phase2_final_per_code_breakdown.json")
