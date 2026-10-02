# Pilot prompt grid and violation codebook

A small pilot set for the contextual compliance rate (CCR). It is small enough to
review by hand, so the generate, annotate, score workflow can be checked before
running the full evaluation (the full evaluation prompts are in
`docs/phase2_prompts_v1.md`).

## Contents

- `prompts.json`: 5 drafting prompts, each chosen to exercise one or a
  combination of the four hand-written worked-example rules in
  `cpgtr/library.py` (r_PC, r_WD, r_SALE, r_RET).
- `grid.json`: the 5 prompts crossed with the three evaluation ages (9, 15, 21)
  at jurisdiction EU, giving 15 (prompt, context) pairs. This is the file read by
  `run_e2e.py` when generating Table 5 outputs.
- `codebook.md`: the violation definitions annotators mark outputs against, one
  entry per rule, with the statutory citation.

## Workflow

1. **Generate.** For each (prompt, context) pair, each method (unconstrained LLM,
   few-shot prompted, critique-and-revise, CP-GTR) produces its own output text.
   CP-GTR's output is the repaired version of the same unconstrained draft that
   the unconstrained arm uses, so the comparison isolates the effect of repair.
   This step needs real LLM calls (see the README's "Running the pipeline").
2. **Annotate.** A qualified human reviewer reads each output and marks which
   violations from `codebook.md` are present (an empty list means compliant).
   Use at least two annotators and report Cohen's kappa on a shared subset.
   Violations are a property of what a method actually wrote, so annotation can
   only happen after generation. `annotate.py` supports this step.
3. **Score.** `cpgtr/eval_compliance.py: compliance_rate()` takes the annotated
   outputs of one method and computes the compliant fraction with a 95%
   bootstrap confidence interval per age column.
