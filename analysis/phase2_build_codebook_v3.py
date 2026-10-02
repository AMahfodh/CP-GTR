"""Build schema/codebook_v3.json from schema/codebook_v2.json.

Supports: the final label set behind the compliance results. The codebook
changes exactly one code, US_RETENTION, whose v2 marking criterion fired only
on an affirmative claim of indefinite retention and could not fire on a notice
that is silent about any retention period. The v3 criterion also marks a
notice that omits a retention period (or the criteria used to determine one).
The primary annotator re-annotates this code under the v3 wording (see
phase2_build_v3_reannotation_workbook.py) and phase2_final_results.py applies
that re-annotation to all units.

Criteria audit: every v2 code's marking criterion was checked for the specific
gap pattern above, a criterion phrased to catch only an affirmative violating
statement.
  - US_RETENTION is changed. The evidence is two independent signals: of 50
    applicable documents, 9 (18%) mention retention or deletion but state no
    period, and an annotator note on one unit independently raised the same
    point about 16 C.F.R. 312.10.
  - EU_PARENTAL_CONSENT, UK_PARENTAL_CONSENT, US_VPC and US_SALE_MINOR also
    require a described act to fire, but there is no evidence that silence
    occurs in the corpus, so they are not changed.
  - UK_HIGH_PRIVACY_DEFAULT and UK_GEOLOCATION_DEFAULT ("states ... switched on
    by default") have a similar bucket (10 of 81 documents mention the topic
    without a default-state statement) but only one signal, so they are not
    changed; the open question is recorded in verification_notes.
  - EU_WITHDRAWAL and US_PARENT_REVIEW are already phrased as omission tests.
  - EU_RETENTION is also an omission test; one annotator note raises a narrower
    ambiguity (whether a period stated only for a post-withdrawal event
    satisfies the criterion). That is an interpretation question about existing
    wording rather than a proven gap, so it is recorded, not changed.

codebook_v2.json is not modified (checked by hash at the end of the run).

Run:
    python analysis/phase2_build_codebook_v3.py
Order: after the first annotation round; before
phase2_build_v3_reannotation_workbook.py and phase2_final_results.py.

Inputs : schema/codebook_v2.json.
Outputs: schema/codebook_v3.json.
"""
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_PATH = REPO_ROOT / "schema" / "codebook_v2.json"
V3_PATH = REPO_ROOT / "schema" / "codebook_v3.json"

v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
v2_sha256 = hashlib.sha256(V2_PATH.read_bytes()).hexdigest()

OLD_US_RETENTION = "Mark if the notice states indefinite retention of a child's personal information."
NEW_US_RETENTION = ("Mark if the notice states indefinite retention of a child's personal "
                     "information, or omits a retention period (or the criteria used to "
                     "determine one) for that information.")

v3 = json.loads(json.dumps(v2))  # deep copy, byte-for-byte identical until edited below
v3["codebook_id"] = "codebook_v3"
v3["version"] = "v3"
v3["frozen_at"] = "2026-09-25T00:00:00Z"
v3["source"] = v2["source"]
v3["verification_status"] = (
    "Self-conducted criteria audit (2026-09-25), NOT independently reviewed by Prof. Ali for "
    "wording quality before use (unlike codebook_v1 -> codebook_v2, which Prof. Ali reviewed "
    "directly) -- his re-annotation of the changed code IS the practical check this round, per "
    "explicit instruction (session_report_2026-09-25c.md, Decision 2). Exactly one code changed: "
    "US_RETENTION. Evidence and audit method in this file's own docstring "
    "(analysis/phase2_build_codebook_v3.py) and session_report_2026-09-25c.md."
)

changed = False
for code_entry in v3["codes"]:
    if code_entry["code"] == "US_RETENTION":
        assert code_entry["marking_criterion"] == OLD_US_RETENTION, code_entry["marking_criterion"]
        code_entry["marking_criterion"] = NEW_US_RETENTION
        changed = True
assert changed, "US_RETENTION not found in codebook_v2.json's codes list"

v3["verification_notes"] = list(v2["verification_notes"]) + [
    {
        "topic": "US_RETENTION.marking_criterion (CHANGE, codebook_v2 -> codebook_v3)",
        "note": "v2 wording caught only an AFFIRMATIVE claim of indefinite retention, not a "
                "document that omits any stated retention period -- a real, dual-evidenced gap "
                "(session_report_2026-09-24c.md Task 2: 9/50 (18%) of applicable documents mention "
                "retention/deletion but state no period; Prof. Eid's U021 note, session_report_2026-"
                "09-25b.md, independently raised the identical point re: 16 C.F.R. Sec.312.10). "
                "Re-annotation of this one code, for Prof. Ali only, covers all 50 US_RETENTION-"
                "applicable units in the 353 (annotation/annotator_primary_v3_us_retention.xlsx). "
                "Prof. Ahmed and Prof. Eid are NOT re-asked -- their US_RETENTION judgements remain "
                "on record under v2 wording, disclosed as such; all three marked 0/15 on the 82-unit "
                "overlap's US_RETENTION column, so pre-adjudication agreement on this code was "
                "degenerate (zero variance) regardless of which wording was used.",
        "old_wording": OLD_US_RETENTION,
        "new_wording": NEW_US_RETENTION,
    },
    {
        "topic": "EU_RETENTION -- considered, NOT changed (open question for the user)",
        "note": "Prof. Eid's U308 note (session_report_2026-09-25b.md) flags a real ambiguity: a "
                "document stating a retention period only for a post-withdrawal/deletion-trigger "
                "event (not for the period while consent/active processing stands) was marked 'no' "
                "under v2's 'omits a retention period' wording, since SOME period is technically "
                "stated. This is an application/interpretation question about which PHASE the "
                "criterion's 'a retention period' must cover, not a proven wording gap on the scale "
                "of US_RETENTION's -- one note, not a quantified pattern -- so it was not changed "
                "unilaterally. Flagged for a decision: clarify EU_RETENTION's wording to specify the "
                "active-processing phase explicitly, or leave as an annotator-training/interpretation "
                "matter."
    },
    {
        "topic": "UK_HIGH_PRIVACY_DEFAULT / UK_GEOLOCATION_DEFAULT -- considered, NOT changed",
        "note": "Both are 'states ... switched on by default' (affirmative-only), structurally the "
                "same pattern as US_RETENTION's v2 gap. session_report_2026-09-24c.md's Task 2 found "
                "UK_GEOLOCATION_DEFAULT has a parallel 10/81 (12%) bucket of documents that mention "
                "the topic without stating a default state -- but unlike retention, no annotator "
                "independently flagged this as a criterion gap. Not changed for lack of a second, "
                "independent signal; noted here so it isn't lost."
    },
    {
        "topic": "EU_PARENTAL_CONSENT / UK_PARENTAL_CONSENT / US_VPC / US_SALE_MINOR -- audited, NOT changed",
        "note": "All four require a DESCRIBED act (relying on the child's own consent; collecting "
                "without VPC; selling/sharing without opt-in) to fire, so a notice silent about the "
                "underlying mechanism entirely would not trigger any of them either -- the same "
                "theoretical shape as the changed gap, but with zero supporting evidence (no "
                "annotator note, no quantified finding) that this actually occurs in the corpus. Not "
                "changed."
    },
]

v3_basis = json.dumps({"codes": v3["codes"], "over_requirement": v3["over_requirement"]}, sort_keys=True)
v3["sha256"] = hashlib.sha256(v3_basis.encode("utf-8")).hexdigest()

V3_PATH.write_text(json.dumps(v3, indent=2), encoding="utf-8")

# Non-destructive check: codebook_v2.json itself must be byte-identical to before.
assert hashlib.sha256(V2_PATH.read_bytes()).hexdigest() == v2_sha256, "codebook_v2.json was modified!"

print(f"Written: {V3_PATH}")
print(f"codebook_v2.json SHA-256 (whole file, unchanged): {v2_sha256}")
print(f"codebook_v3.json embedded sha256 (codes+over_requirement basis): {v3['sha256']}")
print(f"codebook_v3.json whole-file SHA-256: {hashlib.sha256(V3_PATH.read_bytes()).hexdigest()}")
print(f"Codes changed: US_RETENTION only (1 of 11)")
