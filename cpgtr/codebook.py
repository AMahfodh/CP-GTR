"""Machine-readable mirror of table4_pilot/codebook's four violation
codes and their "Applies when" conditions. Kept in sync by hand with that
file (the .md is the human-readable annotator reference; this module is what
annotate.py uses to only ask about codes that actually apply at a given
(age, jurisdiction) context, matching the codebook's own annotation
instructions: "a code that isn't applicable at this context is never
marked, regardless of the text").
"""
from __future__ import annotations

CODES = {
    "VIOLATION_PARENTAL_CONSENT": {
        "rule": "r_PC",
        "citation": "GDPR Art. 8",
        "applies": lambda age, jurisdiction: age < 16 and jurisdiction == "EU",
        "description": (
            "the text describes collecting personal data or obtaining consent "
            "from the user, but does not state that a minor's consent must be "
            "given or authorised by a holder of parental responsibility"
        ),
    },
    "VIOLATION_NO_WITHDRAWAL": {
        "rule": "r_WD",
        "citation": "GDPR Art. 7(3)",
        "applies": lambda age, jurisdiction: jurisdiction == "EU",
        "description": (
            "the text describes a consent process but does not state that the "
            "user may withdraw consent at any time"
        ),
    },
    "VIOLATION_UNLAWFUL_SALE": {
        "rule": "r_SALE",
        "citation": "COPPA (16 C.F.R. Part 312); GDPR analogue",
        "applies": lambda age, jurisdiction: age < 16 and jurisdiction in ("EU", "US"),
        "description": (
            "the text states or implies that personal information is sold or "
            "shared with third parties for their own marketing purposes, "
            "without a lawful basis/exception being stated"
        ),
    },
    "VIOLATION_INDEFINITE_RETENTION": {
        "rule": "r_RET",
        "citation": "GDPR Art. 5(1)(e)",
        "applies": lambda age, jurisdiction: jurisdiction == "EU",
        "description": (
            "the text states or implies data is retained indefinitely, \"as "
            "long as needed\" with no bound, or does not state a bounded "
            "retention period at all despite discussing retention/storage"
        ),
    },
}


def applicable_codes(age: int, jurisdiction: str) -> list[str]:
    """Codes whose "Applies when" condition holds at (age, jurisdiction),
    in codebook.md's own order."""
    return [code for code, spec in CODES.items() if spec["applies"](age, jurisdiction)]
