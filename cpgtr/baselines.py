"""Baseline arms for the contextual-compliance comparison (no graph rewriting).

Role in the pipeline: generates the free-text policy drafts of the three
LLM-only baselines that CP-GTR is compared against:

  (i)   Unconstrained  - draft directly from the prompt;
  (ii)  Few-Shot       - draft with two in-context exemplars of compliant text;
  (iii) RLAIF          - Constitutional-AI-style critique-and-revise of a draft.

The detection-then-regenerate arm (CP-GTR-Detect) lives in ``run_e2e.py``
(`flag_violations` / `cpgtr_detect_repair`), and CP-GTR itself is
`cpgtr.realize.process_document`.

Each function takes a ``complete`` client (see `cpgtr.llm.make_complete`) and
returns plain output text. No JSON schema is used, since these arms produce
free-text drafts rather than structured rules. Outputs are scored later by gold
annotation (`eval_compliance`), not here.
"""
from __future__ import annotations

# max_tokens for critique_revise_rlaif()'s critique-then-revise call. The
# response contains both a critique and a full revision (and reasoning models
# also spend budget on hidden reasoning), so a small budget can cut the revision
# off mid-text.
RLAIF_MAX_TOKENS = 4096

# Two short, generic compliant-policy exemplars for the Few-Shot arm. They are
# not copied from any real company policy and are phrased as general patterns
# (parental consent + withdrawal; retention + no third-party sale) rather than
# quoting library.py's rule templates, so the baseline gets a realistic
# in-context hint without a preview of the rule library.
FEWSHOT_EXAMPLES = """Example 1: If our service determines a user is under the \
age of digital consent in their jurisdiction, we require verifiable consent \
from a parent or legal guardian before collecting any personal data, and this \
consent may be withdrawn at any time by contacting our support team.

Example 2: We retain personal data only as long as necessary to provide our \
service, and delete it within 24 months of account closure. We do not sell or \
share personal data with third parties for their own marketing purposes."""

# General Constitutional-AI-style principles (Bai et al., 2022) for the RLAIF
# arm's critique-and-revise step. They are deliberately broad and do not restate
# the four specific violation codes in docs/table4_pilot/codebook.md: a
# constitutional-AI baseline critiques against general values, not against the
# evaluator's own answer key.
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
%s
Return only the drafted text, no commentary."""

_RLAIF_PROMPT = """You are reviewing a draft privacy-policy section against \
the following principles:

%s
%s
DRAFT:
%s

First, briefly critique the draft against each principle that applies. Then, \
on a new line starting with exactly "REVISED:", write the full revised draft \
text that addresses any problems you found while keeping the same purpose. If \
the draft already satisfies every principle, the revised text may be \
unchanged. Do not include anything after the revised text."""


def _context_line(A, J):
    """Return a prompt line stating the reader's age and jurisdiction.

    The Unconstrained arm is context-blind by design: it produces the draft that
    CP-GTR repairs, and CP-GTR's context awareness comes from the certified
    rules' own context predicates. The Few-Shot and RLAIF arms (and the
    regeneration step of CP-GTR-Detect) have no other source of context, since
    a model asked to draft or revise a privacy-policy section cannot know which
    law applies unless told the reader. Without this line their results would
    not differ across the age/jurisdiction grid.

    Returns "" when both `A` and `J` are None, which leaves the prompt
    context-blind.
    """
    if A is None and J is None:
        return ""
    return f"\nThe intended reader of this document is {A} years old and located in {J}.\n"


def draft_unconstrained(complete, prompt_text: str) -> str:
    """Arm (i): draft directly from the prompt, with no compliance guidance.

    Deliberately context-blind (see `_context_line`)."""
    return complete(_UNCONSTRAINED_PROMPT % prompt_text).strip()


def draft_fewshot(complete, prompt_text: str, A=None, J=None) -> str:
    """Arm (ii): draft with two in-context exemplars of compliant language.

    If `A` (age) and `J` (jurisdiction) are given, the reader's context is
    stated in the prompt, so the arm must be drafted once per (prompt, context)
    pair rather than once per prompt. Omit both for a context-blind draft."""
    return complete(_FEWSHOT_PROMPT % (FEWSHOT_EXAMPLES, prompt_text, _context_line(A, J))).strip()


def critique_revise_rlaif(complete, draft_text: str, A=None, J=None) -> str:
    """Arm (iii): one combined critique-then-revise call over an existing draft.

    The draft is typically the Unconstrained arm's output. The model critiques
    it against `RLAIF_PRINCIPLES` and writes a revision after a "REVISED:"
    marker; the text after the last marker is returned. If the model omits the
    marker, the whole response is returned (it is still real model output,
    just not cleanly delimited). `A` and `J` behave as in `draft_fewshot`."""
    raw = complete(_RLAIF_PROMPT % (RLAIF_PRINCIPLES, _context_line(A, J), draft_text),
                   max_tokens=RLAIF_MAX_TOKENS)
    marker = "REVISED:"
    idx = raw.rfind(marker)
    if idx == -1:
        return raw.strip()
    return raw[idx + len(marker):].strip()
