"""Unconstrained LLM, Few-Shot Prompted, and Constitutional AI (RLAIF) critique-and-revise
(CP-GTR_V2 sec:baselines, arms i-iii). CP-GTR-Detect (arm iv) already
lives in run_e2e.py (flag_violations()/cpgtr_detect_repair()); CP-GTR itself
(arm v) is cpgtr/realize.py: process_document().

Each function takes a `complete` client (see cpgtr.llm.make_complete) and
returns plain output text -- no JSON schema, unlike extract.py/refine.py,
since these arms produce free-text policy drafts, not structured rules.
"""
from __future__ import annotations

# Two short, generic compliant-policy exemplars for the Few-Shot arm -- not
# copied from any real company policy, and deliberately phrased as GENERAL
# patterns (parental consent + withdrawal; retention + no third-party sale)
# rather than quoting library.py's rule templates verbatim, so this baseline
# gets a realistic in-context hint, not an unfair preview of the codebook.
FEWSHOT_EXAMPLES = """Example 1: If our service determines a user is under the \
age of digital consent in their jurisdiction, we require verifiable consent \
from a parent or legal guardian before collecting any personal data, and this \
consent may be withdrawn at any time by contacting our support team.

Example 2: We retain personal data only as long as necessary to provide our \
service, and delete it within 24 months of account closure. We do not sell or \
share personal data with third parties for their own marketing purposes."""

# General Constitutional-AI-style principles (Bai et al. 2022) for the RLAIF
# arm's critique-and-revise step. Deliberately phrased as broad principles,
# not a restatement of docs/table4_pilot/codebook.md's four specific codes --
# a real constitutional-AI baseline critiques against general values, not
# against the evaluator's own answer key.
RLAIF_PRINCIPLES = """1. Protect the privacy of children and minors with extra \
care appropriate to their age.
2. Give users meaningful control over their consent, including the ability to \
withdraw it at any time.
3. Avoid retaining personal data longer than necessary.
4. Avoid selling or sharing personal data with third parties without a clear \
lawful basis."""

_UNCONSTRAINED_PROMPT = """%s

Return only the drafted text, no commentary."""

_FEWSHOT_PROMPT = """Here are two examples of well-written privacy-policy \
sections:

%s

Now complete this task, following a similar style:

%s

Return only the drafted text, no commentary."""

_RLAIF_PROMPT = """You are reviewing a draft privacy-policy section against \
the following principles:

%s

DRAFT:
%s

First, briefly critique the draft against each principle that applies. Then, \
on a new line starting with exactly "REVISED:", write the full revised draft \
text that addresses any problems you found while keeping the same purpose. If \
the draft already satisfies every principle, the revised text may be \
unchanged. Do not include anything after the revised text."""


def draft_unconstrained(complete, prompt_text: str) -> str:
    """Arm (i): draft directly from the prompt, no compliance guidance."""
    return complete(_UNCONSTRAINED_PROMPT % prompt_text).strip()


def draft_fewshot(complete, prompt_text: str) -> str:
    """Arm (ii): draft with two in-context exemplars of compliant language."""
    return complete(_FEWSHOT_PROMPT % (FEWSHOT_EXAMPLES, prompt_text)).strip()


def critique_revise_rlaif(complete, draft_text: str) -> str:
    """Arm (iii): one combined critique-then-revise call over an existing
    draft (reuses the Unconstrained arm's draft -- see
    docs/table4_pilot/README table4_pilot.md). Extracts the text after the
    "REVISED:" marker; falls back to the raw response if the model didn't
    follow the marker format (still real model output, just not cleanly
    delimited -- better than silently dropping a real generation)."""
    raw = complete(_RLAIF_PROMPT % (RLAIF_PRINCIPLES, draft_text))
    marker = "REVISED:"
    idx = raw.rfind(marker)
    if idx == -1:
        return raw.strip()
    return raw[idx + len(marker):].strip()
