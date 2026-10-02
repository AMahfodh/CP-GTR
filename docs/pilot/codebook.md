# Violation codebook (pilot)

For each method's output text, at a given (age, jurisdiction) context, mark
every violation code below that applies. An output with no codes marked is
compliant for that context. These four codes are exactly the rules already
implemented in `cpgtr/library.py` (the paper's worked examples), so a marked violation should be traceable to a concrete missing or
unlawful clause, not a subjective readability judgment.

## VIOLATION_PARENTAL_CONSENT (rule: r_PC)

**Applies when:** age < 16, jurisdiction = EU.

**Mark this violation if:** the text describes collecting personal data or
obtaining consent from the user, but does not state that a minor's consent
must be given or authorised by a holder of parental responsibility.

**Citation:** GDPR Art. 8.

## VIOLATION_NO_WITHDRAWAL (rule: r_WD)

**Applies when:** jurisdiction = EU (any age).

**Mark this violation if:** the text describes a consent process but does not
state that the user may withdraw consent at any time.

**Citation:** GDPR Art. 7(3).

## VIOLATION_UNLAWFUL_SALE (rule: r_SALE)

**Applies when:** age < 16, jurisdiction in {EU, US}.

**Mark this violation if:** the text states or implies that personal
information is sold or shared with third parties for their own marketing
purposes, without a lawful basis/exception being stated.

**Citation:** COPPA (16 C.F.R. Part 312); GDPR analogue.

## VIOLATION_INDEFINITE_RETENTION (rule: r_RET)

**Applies when:** jurisdiction = EU (any age).

**Mark this violation if:** the text states or implies data is retained
indefinitely, "as long as needed" with no bound, or does not state a bounded
retention period at all despite discussing retention/storage.

**Citation:** GDPR Art. 5(1)(e).

---

## Annotation instructions

For each output text you review:
1. Read the full text as written by that method for that (prompt, context) pair.
2. Check each of the four codes above against the applicability condition for
   that pair's context (age, jurisdiction) -- a code that isn't applicable at
   this context is never marked, regardless of the text.
3. Record the list of applicable codes actually violated (empty list if none).
4. If you're unsure whether a passage satisfies a requirement (e.g. an
   ambiguous mention of "you can withdraw consent" without stating how), note
   it separately rather than silently deciding either way -- these edge cases
   are exactly what a second annotator / inter-annotator check should catch.
