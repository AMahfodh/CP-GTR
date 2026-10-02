# CP-GTR: certified repair of LLM-drafted legal text

Reference implementation for the paper *Certified Repair of LLM-Drafted Legal Text with
Context-Gated Graph Transformation Rules*. The code turns statutory privacy-law text into
typed graph-transformation rules, certifies the rule set (termination and confluence), and
uses it to rewrite a specific privacy notice for a given reader age and jurisdiction. The
evaluation benchmark is called LegiSafe-Bench. It covers COPPA, GDPR Art. 8, the CCPA and
the UK Age-Appropriate Design Code, with the UN Convention on the Rights of the Child as a
normative (non-operative) source.

This repository contains code and documentation only. **The statutory texts, the benchmark
data, cached model outputs and human annotations are not distributed here.** The section
"Obtaining the data" explains how to fetch the public sources and rebuild what can be rebuilt.

## What the pipeline does

1. **Extract.** An LLM converts each statutory provision into a candidate rule, constrained to
   a fixed vocabulary of node and edge types (the type graph). A rule says: if this pattern is
   present, rewrite it to that pattern, unless a forbidden pattern also holds, and only when
   the reader's context (age A, jurisdiction J) satisfies the rule's guard.
2. **Certify.** A candidate is admitted only if the whole rule set stays terminating and
   confluent. Termination comes from a global stratification rank computed once over all
   candidates (cycles are broken by rejecting a minimum feedback vertex set). Confluence comes
   from critical-pair analysis with negative application conditions, checked over every
   context cell (each maximal region of age and jurisdiction on which the active rule set is
   constant), not only a sampled grid.
3. **Repair.** A privacy notice is parsed into the same graph representation, the certified
   rules are applied to a normal form, and the result is realized back into edited sentences.
   A round-trip check re-parses the output. If a rule still fires, the document is flagged for
   review instead of being reported as compliant.
4. **Evaluate.** `run_e2e.py` reproduces the paper's result tables 4 to 8: corpus extraction
   yield, contextual compliance rate for five methods, parser precision/recall/F1, and two
   certification ablations (mechanism validation on seeded unsafe rules, and deployment
   realism on a real extracted library).

[GLOSSARY.md](GLOSSARY.md) maps the paper's notation to the code identifiers that implement it.

## Repository layout

| Path | Contents |
|---|---|
| `cpgtr/` | The pipeline package: extraction, admission and certification, context cells, the double-pushout rewriting engine, parsing, realization, baselines, evaluation, and the LLM client. `cpgtr/test_certify_d2d3.py` holds unit tests. |
| `run_e2e.py` | End-to-end runner that prints Tables 4 to 8 and writes `results.json`. |
| `legisafe_eval.py` | Re-renders the tables from an existing `results.json` with no recomputation. |
| `annotate.py` | Blinded annotation tool that collects the human violation labels behind Table 5. |
| `corpus/download_corpus.py`, `corpus/segment_corpus.py` | Fetch the statutory sources and segment them into provision spans. |
| `schema/` | The canonical type-graph schema (`attach.yaml`) and the approved edge equivalences (`equivalences.yaml`). |
| `analysis/` | Scripts for the compliance study and post-hoc audit. Each has a module docstring with its inputs, outputs and place in the run order. |
| `docs/phase2_prompts_v1.md`, `docs/table4_pilot/` | The evaluation prompts, the pilot prompt grid, and the violation codebook used by annotators. |
| `requirements.txt`, `.env.example` | Optional dependencies and the API-key template. |

## Setup

Python 3.10 or newer is recommended (developed and tested on 3.11). The core pipeline, the
unit tests and the offline demo run use only the standard library. Install the optional
packages only for the steps that need them:

```bash
pip install -r requirements.txt
```

Check the installation (no data or API key needed):

```bash
python -m unittest cpgtr.test_certify_d2d3
```

A quick sanity check of the certification gate on the paper's worked example. `r_BAD` must be
rejected, and the admitted set must not depend on the input order:

```python
from cpgtr.library import r_PC, r_WD, r_BAD, r_SALE, r_RET
from cpgtr.admit import admit

cert, log = admit([r_PC(), r_WD(), r_BAD(), r_SALE(), r_RET()])
print([r.name for r in cert])        # ['r_SALE', 'r_WD', 'r_RET', 'r_PC']
cert2, _ = admit([r_BAD(), r_RET(), r_SALE(), r_WD(), r_PC()])
assert [r.name for r in cert2] == [r.name for r in cert]
```

## Obtaining the data

The statutory source texts are published by their official bodies and are not redistributed
in this repository. Everything below runs from the repository root.

### 1. Fetch the statutory sources into `corpus/raw/`

| Source | Official publisher | File the segmenter expects |
|---|---|---|
| COPPA Rule, 16 C.F.R. Part 312 | eCFR: https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312 | `corpus/raw/coppa_16cfr312.txt` |
| GDPR (Regulation (EU) 2016/679), full text. Only Article 8 is used. | EUR-Lex: https://eur-lex.europa.eu/eli/reg/2016/679/oj | `corpus/raw/gdpr_reg2016_679.txt` |
| CCPA, California Civil Code Title 1.81.5 | California Legislative Information (see `MANIFEST` in `corpus/download_corpus.py`) | `corpus/raw/ccpa_civ_1798.txt` |
| UK Age-Appropriate Design Code | Information Commissioner's Office: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/ | `corpus/raw/uk_aadc2.txt` |
| UN Convention on the Rights of the Child | OHCHR: https://www.ohchr.org/en/instruments-mechanisms/instruments/convention-rights-child | `corpus/raw/OHCHR.txt` |

Run the downloader first. It fetches COPPA, GDPR, CCPA and the UK Code's web page. It sends a
descriptive User-Agent and pauses between requests. Edit the contact address in the `UA`
string at the top of `corpus/download_corpus.py` before running it.

```bash
python corpus/download_corpus.py
```

Two sources need manual steps, because the downloader does not automate them:

- **UN CRC.** Download the Convention text from OHCHR and save it as plain text at
  `corpus/raw/OHCHR.txt`.
- **UK Code.** The segmenter is written for the text extracted from the Code's PDF (it
  removes page banners and re-joins hard-wrapped lines). Download the PDF from the ICO page,
  extract plain text (for example with `pdftotext`), and save it as `corpus/raw/uk_aadc2.txt`.
  Delete or move the `uk_aadc.txt` written by the downloader. Otherwise it is segmented as an
  extra source.

Statutory text changes over time, and the downloader pins an eCFR snapshot date in its
`MANIFEST`. Confirm that each snapshot is the version you intend to study. Nothing here is
legal advice.

### 2. Segment the sources into provision spans

```bash
python corpus/segment_corpus.py
```

This writes `corpus/provisions.json`, the input to Stage-1 extraction, and stops with an error
if the per-source span counts look implausible. Run it only after all five files exist in
`corpus/raw/`. With no input files it writes an empty list. The paper's corpus has 1,493
spans (COPPA 153, GDPR Art. 8 4, CCPA 362, UK AADC 795, UN CRC 179). Other source snapshots
will give similar but not identical counts, and downstream numbers will shift accordingly.

### 3. Other inputs that are not distributed

| Input | How to obtain it |
|---|---|
| `corpus/gold_parses.json`, `corpus/holdout_hosts*.json` (parser gold set and repair holdout hosts used by Tables 6 to 8) | Not included. `python run_e2e.py --dry-run` lists every file the run expects. To build your own holdout hosts from drafted policy text, see `generate_real_holdout_hosts` in `run_e2e.py`. |
| `cache/*.jsonl` (cached extraction output) | Regenerated by running extraction in real mode. The cache makes reruns deterministic and free. |
| Human annotations (`cache/table5_*.json`, `annotation/`) | Not distributed. Collect your own labels with `annotate.py` and the codebook in `docs/table4_pilot/codebook.md`. |
| Frozen rule libraries and codebooks used by the compliance study in `analysis/` | Rebuilt by the scripts described in `analysis/`. Several are verified by SHA-256, so a rebuilt library will fail the hash check unless you update the expected values in `analysis/phase2_generation_lib.py`. |

## Running the pipeline

All commands run from the repository root.

```bash
python run_e2e.py                    # offline demo mode: no API calls
python run_e2e.py --dry-run          # check configuration and required files, run nothing
python legisafe_eval.py              # re-render the tables from results.json
```

Demo mode needs `corpus/provisions.json` and the benchmark fixtures. It prints the tables
that can be measured offline and states plainly which cells cannot be filled. The runner
never prints a placeholder number: a cell appears only when a real measurement backs it,
and anything unmeasurable raises `NotImplementedError` or is reported as "not run". Table 5
needs generated outputs plus human annotations, and Table 8 needs real mode.

### Real mode (calls an LLM)

Real mode uses any of four OpenAI-compatible providers, selected by `CPGTR_PROVIDER`:
`cerebras` (default, open-weight `gpt-oss-120b`), `groq`, `openrouter` or `poe`. The model
differs between providers, so numbers from different providers are not directly comparable.
Every real run records its provider and model in `results.json`.

```bash
cp .env.example .env                 # then put your key in .env; never commit it
CPGTR_MODE=real CPGTR_PROVIDER=cerebras python run_e2e.py
```

Every call is appended to `logs/llm_calls.jsonl` with its token usage. Extraction results are
cached in `cache/real_extraction_cache.jsonl`, so rerunning does not repeat paid calls. The
sampling seed (42) and sample size (200 provisions) are set at the top of `run_e2e.py`. LLM
output is not bit-reproducible even at temperature 0, so expect small differences from the
paper's extraction counts. Certification and repair are deterministic given the same
candidate rules.

## The compliance study and audit scripts in `analysis/`

These scripts generate the evaluation documents for six arms (10 prompts, 3 ages, 3
jurisdictions), score them against human labels, and run the post-hoc certification audit.
Every script states its inputs, outputs and place in the run order in its module docstring.
Because they depend on frozen artifacts and annotation data that are not distributed here,
a full rerun requires you to produce your own annotations. The scripts are published so the
analysis choices (sampling, bootstrap over prompts, Holm correction, agreement statistics)
can be inspected and reused.

Suggested order: `phase1a_run.py` (canonical extraction), `phase1_run.py` and
`topology_inventory.py` (baseline and topology inventory), `phase2_generation_run*.py`
(document generation), `phase2_taskZ_annotation_sheet.py` and
`phase2_build_annotation_workbooks.py` (annotation sheets), `phase2_final_results_part2.py`
and `phase2_final_results_part3.py` (results), `phase2_recompute_tables_4_7_8.py`
(offline recomputation of Tables 4, 7 and 8), then the `verify_*.py` audit scripts.

## Notes on scope

- The offline pipeline is deterministic. Only the LLM calls (extraction, parsing in real
  mode, drafting, baselines) are not bit-reproducible.
- The certification guarantee covers termination and confluence of the admitted rule set. It
  does not establish that a rule faithfully states the law. An independent legal review of
  rule fidelity is a separate step reported in the paper.
- Statutory text is not legal advice, and repaired policies must be confirmed with counsel.

