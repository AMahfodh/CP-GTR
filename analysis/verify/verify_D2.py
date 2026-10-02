import sys, json, csv
sys.path.insert(0, '.')
import openpyxl
from analysis.phase2_harness import build_scored_documents, text_hash

REPO_ROOT = '.'
ALI_PRIMARY_PATH = 'annotation/annotator_primary_filled_by_prof_Ali.xlsx'
AHMED_OVERLAP_PATH = 'annotation/annotator_overlap_filled_by_colleague_Ahmed.xlsx'
EID_OVERLAP_PATH = 'annotation/annotator_overlap_filled_by_colleague_Eid.xlsx'


def read_annotation_sheet(path, n_code_cols=11):
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


ali_v2 = read_annotation_sheet(ALI_PRIMARY_PATH)
ahmed_v2 = read_annotation_sheet(AHMED_OVERLAP_PATH)
eid_v2 = read_annotation_sheet(EID_OVERLAP_PATH)

overlap = json.load(open('annotation/overlap_sample.json', encoding='utf-8'))
overlap_uids = sorted(overlap['sampled_unit_ids'])
assert len(overlap_uids) == 82

CODES = [c for c in ali_v2[overlap_uids[0]].keys() if c not in ('A', 'J', 'text')]
print('codes:', CODES)


def majority(a, h, e):
    votes = [a, h, e]
    return "yes" if votes.count("yes") > votes.count("no") else "no"


# Build unit -> (text_hash, A, J) so we can cross-reference against the
# "modified by CP-GTR / CP-GTR-Ungated" document sets.
csv_rows = list(csv.DictReader(open('analysis/phase2_blind_annotation_sheet.csv', encoding='utf-8')))
uid_to_tuple = {f"U{int(row['row_id']):03d}": (text_hash(row['text']), int(row['A']), row['J'])
                for row in csv_rows}

pairs = json.load(open('analysis/phase2_generation_pairs.json', encoding='utf-8'))
scored = build_scored_documents(pairs)
by_pair_arm = {(d['pair_id'], d['arm']): d for d in scored}
modified_tuples = set()
for arm in ('CP-GTR', 'CP-GTR-Ungated'):
    for pid, entry in pairs.items():
        if entry[arm] != entry['Unconstrained']:
            modified_tuples.add(by_pair_arm[(pid, arm)]['unit'])

# --- overall agreement, and split by modified/other ---
n_total = n_agree = 0
n_mod_total = n_mod_agree = 0
n_other_total = n_other_agree = 0
for uid in overlap_uids:
    a_rec, h_rec, e_rec = ali_v2[uid], ahmed_v2[uid], eid_v2[uid]
    tup = uid_to_tuple[uid]
    is_modified = tup in modified_tuples
    for code in CODES:
        av, hv, ev = a_rec[code], h_rec[code], e_rec[code]
        if av == 'n-a' or av is None:
            continue  # code not applicable at this (A,J) -- skip, matches project convention
        maj = majority(av, hv, ev)
        n_total += 1
        agree = (av == maj)
        n_agree += agree
        if is_modified:
            n_mod_total += 1
            n_mod_agree += agree
        else:
            n_other_total += 1
            n_other_agree += agree

print()
print(f'Overall: {n_agree}/{n_total} = {n_agree/n_total:.4f}')
print(f'Modified-by-CP-GTR/-Ungated units: {n_mod_agree}/{n_mod_total} = {n_mod_agree/n_mod_total:.4f}' if n_mod_total else 'no modified units in overlap')
print(f'Other units: {n_other_agree}/{n_other_total} = {n_other_agree/n_other_total:.4f}')

# --- OVER_REQUIREMENT specific ---
code = 'OVER_REQUIREMENT'
n_pos = n_agree_or = n_total_or = 0
rows_for_kappa = []
for uid in overlap_uids:
    a_rec, h_rec, e_rec = ali_v2[uid], ahmed_v2[uid], eid_v2[uid]
    av, hv, ev = a_rec[code], h_rec[code], e_rec[code]
    if av in ('yes', 'no'):
        n_total_or += 1
        maj = majority(av, hv, ev)
        n_agree_or += (av == maj)
    n_pos += sum(1 for v in (av, hv, ev) if v == 'yes')
    rows_for_kappa.append((av, hv, ev))

print()
print(f'OVER_REQUIREMENT: applicable units={n_total_or}, total yes-votes across 3 raters={n_pos}')
print(f'  primary agreement with majority: {n_agree_or}/{n_total_or} = {n_agree_or/n_total_or:.4f}' if n_total_or else '  n/a')


def fleiss_kappa(rows):
    """rows: list of (a,h,e) in {'yes','no'}. Standard Fleiss kappa for n=3 raters, 2 categories."""
    N = len(rows)
    n_raters = 3
    cats = ['yes', 'no']
    P_i = []
    n_j = {c: 0 for c in cats}
    for row in rows:
        counts = {c: row.count(c) for c in cats}
        for c in cats:
            n_j[c] += counts[c]
        P_i.append((sum(counts[c] ** 2 for c in cats) - n_raters) / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / N
    p_j = {c: n_j[c] / (N * n_raters) for c in cats}
    Pe_bar = sum(p_j[c] ** 2 for c in cats)
    kappa = None if Pe_bar >= 1.0 else (P_bar - Pe_bar) / (1 - Pe_bar)
    free_marginal = (P_bar - 0.5) / 0.5
    return P_bar, kappa, free_marginal, p_j


P_bar, kappa, fm, p_j = fleiss_kappa(rows_for_kappa)
print(f'  Fleiss P_bar (observed agreement) = {P_bar:.4f}')
print(f'  Fleiss kappa = {kappa}')
print(f'  free-marginal (Randolph) kappa = {fm:.4f}')
print(f'  category marginals: {p_j}')
