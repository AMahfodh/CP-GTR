"""Machine-readable violation codebook used for compliance annotation.

Mirrors the four violation codes and their "Applies when" conditions in
``docs/table4_pilot/codebook.md`` (the human-readable annotator reference; the
two must be kept in sync by hand). ``annotate.py`` uses `applicable_codes` to
ask only about codes that apply at a given (age, jurisdiction) context: a code
that does not apply in a context is never marked, whatever the text says.

Each entry in `CODES` has the certified rule it corresponds to (``rule``), the
legal ``citation``, an ``applies(age, jurisdiction)`` predicate and a plain
``description`` of the violation.
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
    """Return the codes whose "Applies when" condition holds at (age, jurisdiction),
    in the codebook's own order."""
    return [code for code, spec in CODES.items() if spec["applies"](age, jurisdiction)]
