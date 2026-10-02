# Table 4 pilot: prompt corpus, evaluation grid, and violation codebook

A small pilot toward CP-GTR.tex's Contextual Compliance Rate (CCR, Table 4:
`tab:ccr`). This is deliberately **not** the full LegiSafe-Bench (`N_p = 40` pilot
/ `300` scaled per `CP-GTR.tex:1439-1442`) -- it's small enough to actually
review by hand, so the workflow below can be validated before committing to
that scale.

## What's here

- `prompts.json` -- 5 drafting prompts, chosen to each exercise a different
  one (or combination) of the four real, manuscript-sourced rules in
  `cpgtr/library.py` (r_PC, r_WD, r_SALE, r_RET), not arbitrary topics.
- `codebook.md` -- the violation definitions annotators mark outputs against,
  one entry per rule, with the statutory citation.
- `grid.json` (to be generated) -- the 5 prompts crossed with the three Table 4
  age contexts (9, 15, 21), fixed at jurisdiction EU for this pilot (`r_SALE`
  is the one rule that also applies under US -- a jurisdiction-diverse
  follow-up pilot can extend this once the workflow is validated). 15
  (prompt, context) pairs total.

## Workflow (why this can't be finished in one step)

1. **Generate.** For each of the 15 (prompt, context) pairs, each method
   (Unconstrained LLM, Few-Shot Prompted, RLAIF, CP-GTR) produces its own
   output text. CP-GTR's output is the *repaired* version of the same
   unconstrained draft the Unconstrained-LLM arm uses (see CLAUDE.md/
   run_e2e.py) -- an apples-to-apples comparison of "draft alone" vs. "draft
   after CP-GTR repair." This step needs real LLM calls (OPENAI_API_KEY).
2. **Annotate.** A human reviewer reads each method's *actual output text*
   for each pair and marks which violations from `codebook.md` are present
   (empty list = compliant). Per the manuscript's own protocol this should be
   >=2 trained annotators with Cohen's kappa computed on disagreements --
   for a first pilot pass, at minimum one qualified reviewer (you, or someone
   you delegate to) doing this deliberately, not Claude self-annotating.
   Annotation is inherently post-hoc: violations are a property of what a
   method *actually wrote*, not something that can be predicted before
   generation.
3. **Score.** `cpgtr/eval_compliance.py: compliance_rate()` takes the
   annotated outputs for one method and computes the fraction compliant +
   95% bootstrap CI, per age column.

Steps 1-2 are not implemented in code on purpose -- they require real
generation and real human judgment respectively. Once you have annotated
output for a method, hand it to `compliance_rate()` directly.
