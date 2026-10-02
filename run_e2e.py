"""run_e2e.py -- end-to-end runner that reproduces the paper's result tables.

CP-GTR turns statutory privacy-law text into typed graph-transformation rules,
certifies the rule set (termination and confluence), and uses it to rewrite a
specific privacy policy for a given reader (age A, jurisdiction J). This script
drives the pipeline end to end and prints five tables. It also writes every
number to ``results.json``, with the run metadata (mode, provider, model,
temperature, sampling seed), so that ``legisafe_eval.py`` can regenerate the
tables later without recomputing anything.

Tables (the function that produces each is in brackets)
  Table 4  corpus sources and extraction yield  [table4]
      Spans per source, counted from ``corpus/provisions.json``. In real mode
      the spans are de-duplicated, a stratified sample (by source and by
      presence of an age threshold, fixed seed) is sent to the Stage-1
      extractor, and the proposed and certified rule counts are attributed back
      to the source of each rule.
  Table 5  contextual compliance rate (CCR)  [table5_ccr]
      For each of five methods and three age columns (child 9, teen 15, adult
      21), the percentage of generated outputs with no violation at that
      (age, jurisdiction), with a bootstrap 95% confidence interval. Violations
      are labelled by human annotators (``annotate.py``); nothing is
      self-graded. The five methods are an unconstrained LLM, a few-shot
      prompted LLM, a constitutional-AI critique-and-revise baseline,
      CP-GTR-Detect and CP-GTR.
  Table 6  parser quality  [table6_parser]
      Precision, recall and F1 of the deterministic RuleParser against the gold
      graph parses in ``corpus/gold_parses.json``.
  Table 7  certification mechanism validation  [table7a]
      Four configurations (full pipeline, without CPA, without stratification,
      without both) run over a fixed pool of hand-written rules that includes
      seeded adversarial fixtures known to be unsafe. It checks that the gate
      rejects the fixtures and counts "unsound" admissions per configuration.
      Needs no LLM and is deterministic.
  Table 8  certification deployment realism  [table8b]
      The same four configurations run on the unmodified rule library produced
      by a real extraction pass, repaired against holdout hosts derived from
      real drafted policy text. Needs ``CPGTR_MODE=real``.

Integrity rule: a cell is printed only if a real measurement backs it. A method
or table that cannot be measured (for example, Table 5 before annotation
exists) raises ``NotImplementedError`` or reports "not run"; no placeholder
number is ever printed. A truncated ablation run (wall-clock budget exceeded)
is refused rather than reported.

CP-GTR-Detect is the fourth Table 5 baseline. It uses the identical certified
rule set, parser and corpus as CP-GTR, but only flags the rule matches
(``flag_violations``) without rewriting the graph. The flagged document is then
handed back to the LLM to regenerate (``cpgtr_detect_repair``), which isolates
the value of repair from the value of detection.

Usage, from the repository root
    python run_e2e.py                       # demo mode: offline, no API calls
    CPGTR_MODE=real CPGTR_PROVIDER=cerebras python run_e2e.py
    python run_e2e.py --dry-run             # check config and files, run nothing
    python run_e2e.py --results-json PATH   # write the results artifact to PATH
    python legisafe_eval.py                 # re-render tables from results.json

Environment variables
    CPGTR_MODE                   "demo" (default) or "real". Demo mode is
                                 offline: Table 4 prints "n/a" for the proposed
                                 and certified columns, Table 8 is not run,
                                 and Table 5 arms without annotations are
                                 reported as not implemented.
    CPGTR_PROVIDER               LLM provider for real mode: "cerebras"
                                 (default), "poe", "groq" or "openrouter". Each
                                 needs its own API key (CEREBRAS_API_KEY,
                                 POE_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY),
                                 read from a ``.env`` file or the environment.
                                 See ``cpgtr/llm.py`` and ``.env.example``.
    CPGTR_EXTRACT_WORKERS        extraction worker threads (default 8).
    CPGTR_EXTRACT_BATCH_SIZE     provisions per extraction call (default 5).

Files read (relative to the repository root)
    corpus/provisions.json                        all tables with a corpus
    corpus/gold_parses.json                       Table 6
    corpus/holdout_hosts.json                     Table 7
    corpus/holdout_hosts_mechanism_probe.json     Table 7
    corpus/holdout_hosts_real.json                Table 8
    docs/table4_pilot/grid.json                   Table 5 generation
    cache/table5_annotations.json                 Table 5 labels
    cache/real_extraction_cache.jsonl             real-mode extraction cache
Files written
    results.json (or --results-json PATH)         all table results
    cache/real_extraction_cache.jsonl             real-mode extraction cache
    cache/table5_generation.json                  generate_table5_grid()
    cache/table5_blind_map.json                   generate_table5_grid()
    corpus/holdout_hosts_real.json                generate_real_holdout_hosts()
    logs/llm_calls.jsonl                          every real LLM call (cpgtr/llm.py)

Two preparation steps are not run by main() because they spend API budget:
``generate_table5_grid('real')`` produces the Table 5 outputs and blinding map
(then annotate with ``annotate.py`` and export), and
``generate_real_holdout_hosts('real')`` builds the Table 8 holdout hosts.
"""
import argparse
import datetime
import os
import json
import random
import time

from cpgtr.parse import RuleParser, LLMParser
from cpgtr.extract import extract, load_cached_candidates
from cpgtr.admit import admit, CONTEXTS
from cpgtr.library import lib, r_PC, r_WD, r_SALE, r_RET, r_NOTIFY
from cpgtr.eval_parser import score_corpus
from cpgtr.eval_compliance import compliance_rate
from cpgtr.eval_repair import run_repair
from cpgtr.llm import make_complete, DEFAULT_MODEL
from cpgtr.rules import one_step, Rule
from cpgtr.baselines import draft_unconstrained, draft_fewshot, critique_revise_rlaif
from cpgtr.realize import process_document

PROVISIONS = os.path.join("corpus", "provisions.json")
GOLD       = os.path.join("corpus", "gold_parses.json")
HOLDOUT    = os.path.join("corpus", "holdout_hosts.json")
# Synthetic stress-test hosts used by Table 7 only. Realistic hosts in HOLDOUT
# do not use the synthetic types and jurisdiction gaps that the seeded
# adversarial rules exercise, so without these hosts a known-bad rule could be
# admitted under a relaxed configuration without ever being exercised.
# table7a() concatenates them with HOLDOUT; table8b() does not, because a
# real-extracted rule never references these synthetic types.
HOLDOUT_PROBE = os.path.join("corpus", "holdout_hosts_mechanism_probe.json")
# Holdout hosts derived from real drafted policy text rather than hand-authored
# graphs; built by generate_real_holdout_hosts() and used by Table 8.
HOLDOUT_REAL = os.path.join("corpus", "holdout_hosts_real.json")
# Consolidated results artifact written by write_results_json(); it is the only
# input legisafe_eval.py reads.
RESULTS_JSON = "results.json"
# Table 5 files: the prompt grid, the generated outputs, the hidden blind-label
# map, and the gold annotations (see generate_table5_grid() and annotate.py).
TABLE5_GRID        = os.path.join("docs", "table4_pilot", "grid.json")
TABLE5_GENERATION  = os.path.join("cache", "table5_generation.json")
TABLE5_BLIND_MAP   = os.path.join("cache", "table5_blind_map.json")
TABLE5_ANNOTATIONS = os.path.join("cache", "table5_annotations.json")
TABLE5_SEED = 7   # blind-label shuffle seed, independent of EXTRACT_SEED
MODE       = os.environ.get("CPGTR_MODE", "demo")
T_MAX      = 10_000   # repair step cap per host; longer runs count as non-terminating

# Extraction budget. Spans are de-duplicated, then a stratified sample of
# EXTRACT_SAMPLE_N spans is sent to the LLM (one stratum per source and
# presence of an age threshold). This is an adjustable pilot-scale setting, like
# T_MAX, not a value fixed by the method; changing it or EXTRACT_SEED changes
# which provisions are extracted and therefore Tables 4 and 8. The cost scales
# with the sample size: provider rate limits and daily token caps can be
# binding at larger values.
EXTRACT_SAMPLE_N = 200
EXTRACT_SEED = 42

# Persistent cache of the sampled extraction's raw rule specifications (see
# extract.py: cache_path and load_cached_candidates). It is a JSONL file that
# extract() appends to after each provision, so an interrupted run resumes at
# the granularity of a single provision, and separate processes share one
# extraction instead of repeating it.
EXTRACTION_CACHE = os.path.join("cache", "real_extraction_cache.jsonl")

# Real extraction runs concurrently in batches. Worker count and batch size can
# be overridden through environment variables without editing code. See
# cpgtr/extract.py and cpgtr/llm.py for the concurrency rationale.
EXTRACT_WORKERS = int(os.environ.get("CPGTR_EXTRACT_WORKERS", "8"))
EXTRACT_BATCH_SIZE = int(os.environ.get("CPGTR_EXTRACT_BATCH_SIZE", "5"))


# --------------------------------------------------------------------------- #
def build_parser(mode):
    """Return the parser for `mode`: LLMParser in real mode, RuleParser in demo.

    Does not touch admission or certification, so a parser can be built even
    when no rule library is available (for example demo mode, or real mode
    without API access)."""
    if mode == "real":
        return LLMParser(make_complete())
    elif mode == "demo":
        return RuleParser()
    raise ValueError(f"unknown MODE: {mode!r}")


_real_complete = None
_real_candidates = None


def get_real_complete():
    """Return the shared LLM client, creating it on first use.

    The provider is selected with CPGTR_PROVIDER (default "cerebras", see
    cpgtr/llm.py). One client serves the whole real run, so token usage
    (cpgtr.llm.UsageTotals) accumulates in one place."""
    global _real_complete
    if _real_complete is None:
        _real_complete = make_complete()
    return _real_complete


def get_real_candidates(force_refresh=False):
    """Run (or load) the real Stage-1 extraction and return candidate rules.

    The provisions are de-duplicated, stratified-sampled (see
    deduplicate_provisions() and stratified_sample()), then extracted
    concurrently and in batches (cpgtr/extract.py). The result is memoized
    for the rest of the process.

    Tables 4 and 8 both need "the real extracted library". extract() resumes
    from EXTRACTION_CACHE (keyed by content hash), so it does not matter which
    table triggers extraction first or whether an earlier run was interrupted;
    callers can ask for the candidates freely without repeating API calls.

    Args:
        force_refresh: ignore EXTRACTION_CACHE and re-extract everything (for
            example after changing EXTRACT_SAMPLE_N or EXTRACT_SEED, or to get
            fresh model output for items that already succeeded).
    """
    global _real_candidates
    if _real_candidates is not None and not force_refresh:
        return _real_candidates

    provisions = json.load(open(PROVISIONS, encoding="utf-8"))
    deduped, dedup_ratio = deduplicate_provisions(provisions)
    sample = stratified_sample(deduped)
    os.makedirs(os.path.dirname(EXTRACTION_CACHE), exist_ok=True)
    if force_refresh and os.path.exists(EXTRACTION_CACHE):
        os.remove(EXTRACTION_CACHE)   # nothing to resume from, so extract() redoes all
    _real_candidates = extract(get_real_complete(), provisions=sample,
                                cache_path=EXTRACTION_CACHE,
                                batch_size=EXTRACT_BATCH_SIZE, max_workers=EXTRACT_WORKERS)
    print(f"[extract] dedup ratio {dedup_ratio:.2f}, sampled n={len(sample)} "
          f"of {len(deduped)} deduped ({len(provisions)} raw), seed={EXTRACT_SEED}, "
          f"got {len(_real_candidates)} candidates -- cached to {EXTRACTION_CACHE}")
    return _real_candidates


def build_cpgtr(mode):
    """Return (parser, cert, log) for CP-GTR under the chosen mode.

    admit() is called without an explicit `contexts` argument on purpose: it
    then certifies against the candidate set's own context cells
    (cpgtr.context), so the guarantee holds for every (A, J). The fixed
    CONTEXTS / CTX grid is used only for reporting and for sampling prompts
    (Table 5), which is a separate concern from certification.
    """
    parser = build_parser(mode)
    if mode == "real":
        cert, log = admit(get_real_candidates())
    elif mode == "demo":
        cert, log = admit(lib.synthesize())
    else:
        raise ValueError(f"unknown MODE: {mode!r}")
    return parser, cert, log


# --------------------------------------------------------------------------- #
# TABLE 4 — LegiSafe-Bench corpus sources and extraction yield (tab:corpus).   #
# --------------------------------------------------------------------------- #
# Raw file name in corpus/raw -> (display label, tier). "operative" sources are
# binding law; the UN CRC is "normative" (an interpretive standard).
SOURCE_TIER = {
    "coppa_16cfr312.txt":   ("COPPA (16 C.F.R. pt. 312)", "operative"),
    "gdpr_reg2016_679.txt": ("GDPR Art. 8", "operative"),
    "ccpa_civ_1798.txt":    ("CCPA", "operative"),
    "uk_aadc2.txt":         ("UK AADC", "operative"),
    "OHCHR.txt":            ("UN CRC", "normative"),
}


def _normalize_for_dedup(text):
    return " ".join(text.lower().split())


def deduplicate_provisions(provisions):
    """Drop repeated spans, since spans within a statutory section are often
    redundant. The key is the span text in lowercase with collapsed
    whitespace, which catches duplicates that differ only in formatting (the
    dominant redundancy in boilerplate-heavy statutory text). It is not a
    fuzzy near-duplicate detector.

    Returns (deduped, ratio) where ratio = len(deduped) / len(provisions).
    """
    seen, deduped = set(), []
    for p in provisions:
        key = _normalize_for_dedup(p["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    ratio = len(deduped) / len(provisions) if provisions else 1.0
    return deduped, ratio


def stratified_sample(provisions, n=EXTRACT_SAMPLE_N, seed=EXTRACT_SEED):
    """Draw a sample of up to n spans, stratified by source and by whether the
    span has an age threshold tau.

    The seed is fixed for reproducibility. Each non-empty stratum contributes
    floor(n / n_strata) spans (at least 1); any remainder is filled round-robin
    from strata that still have spans left. The sample is therefore as close
    to n as the corpus allows and never exceeds a stratum's actual size.
    """
    strata = {}
    for p in provisions:
        key = (p["source"], p["tau"] is not None)
        strata.setdefault(key, []).append(p)

    rng = random.Random(seed)
    for key in strata:
        rng.shuffle(strata[key])

    per_stratum = max(1, n // len(strata)) if strata else 0
    sample, taken = [], {key: 0 for key in strata}
    for key, items in strata.items():
        take = items[:per_stratum]
        sample.extend(take)
        taken[key] = len(take)

    # Round-robin fill from strata with leftover spans until n is reached or
    # every stratum is exhausted.
    keys = list(strata.keys())
    i = 0
    while len(sample) < n and any(taken[k] < len(strata[k]) for k in keys):
        key = keys[i % len(keys)]
        if taken[key] < len(strata[key]):
            sample.append(strata[key][taken[key]])
            taken[key] += 1
        i += 1

    return sample[:n]


def table4(mode):
    """Table 4: LegiSafe-Bench corpus sources and extraction yield.

    Spans is always a real count from corpus/provisions.json. Proposed is the
    number of candidate rules the extractor produced from each source, and
    Certified is how many of those the admission gate (cpgtr.admit) accepted;
    each rule is attributed to its source through Rule.source. Both columns
    need real extraction (CPGTR_MODE=real); in demo mode they print "n/a"
    rather than a made-up number.

    Returns a dict with per-source "rows" and the totals, for results.json.
    """
    provisions = json.load(open(PROVISIONS, encoding="utf-8"))
    spans_by_source = {}
    for p in provisions:
        spans_by_source.setdefault(p["source"], 0)
        spans_by_source[p["source"]] += 1

    print("\n=== Table 4: LegiSafe-Bench corpus sources and extraction yield ===")
    hdr = f"{'Source':<28}{'Tier':<12}{'Spans':>8}{'Proposed':>10}{'Certified':>11}"
    print(hdr); print("-" * len(hdr))

    proposed_by_source = certified_by_source = None
    if mode == "real":
        candidates = get_real_candidates()   # shared with Table 8; see its docstring
        cert, _log = admit(candidates)
        proposed_by_source, certified_by_source = {}, {}
        for r in candidates:
            proposed_by_source[r.source] = proposed_by_source.get(r.source, 0) + 1
        for r in cert:
            certified_by_source[r.source] = certified_by_source.get(r.source, 0) + 1

    total_spans = total_prop = total_cert = 0
    for fname, (label, tier) in SOURCE_TIER.items():
        spans = spans_by_source.get(fname, 0)
        total_spans += spans
        prop = proposed_by_source.get(fname, 0) if proposed_by_source is not None else None
        cert_n = certified_by_source.get(fname, 0) if certified_by_source is not None else None
        if prop is not None:
            total_prop += prop
        if cert_n is not None:
            total_cert += cert_n
        prop_s = str(prop) if prop is not None else "n/a"
        cert_s = str(cert_n) if cert_n is not None else "n/a"
        print(f"{label:<28}{tier:<12}{spans:>8}{prop_s:>10}{cert_s:>11}")
    print("-" * len(hdr))
    total_prop_s = str(total_prop) if proposed_by_source is not None else "n/a"
    total_cert_s = str(total_cert) if certified_by_source is not None else "n/a"
    print(f"{'Total':<28}{'':<12}{total_spans:>8}{total_prop_s:>10}{total_cert_s:>11}")

    rows = []
    for fname, (label, tier) in SOURCE_TIER.items():
        rows.append({
            "source": label, "tier": tier,
            "spans": spans_by_source.get(fname, 0),
            "proposed": (proposed_by_source.get(fname, 0)
                         if proposed_by_source is not None else None),
            "certified": (certified_by_source.get(fname, 0)
                          if certified_by_source is not None else None),
        })
    return {
        "rows": rows,
        "total_spans": total_spans,
        "total_proposed": total_prop if proposed_by_source is not None else None,
        "total_certified": total_cert if certified_by_source is not None else None,
    }


# --------------------------------------------------------------------------- #
# TABLE 5 — contextual compliance rate per method x context (tab:ccr).        #
# --------------------------------------------------------------------------- #
# Table 5 columns: Child A=9, Teen A=15, Adult A=21.
CTX = [("Child (A=9)", 9), ("Teen (A=15)", 15), ("Adult (A=21)", 21)]


def _ccr_from_annotations(method_name):
    """Load one method's gold annotations and return its CCR per age column.

    Shared by the five ccr_*() functions. It reads TABLE5_ANNOTATIONS, which
    is produced by annotate.py --export from human labels on the outputs that
    generate_table5_grid() wrote. If the file or this method's entries are
    missing it raises NotImplementedError instead of fabricating a rate.
    compliance_rate() takes only already-labelled outputs; nothing here
    generates or self-judges anything.
    """
    if not os.path.exists(TABLE5_ANNOTATIONS):
        raise NotImplementedError(
            f"No gold annotations yet for {method_name!r}. Run "
            f"generate_table5_grid('real') to produce real output text (see "
            f"{TABLE5_GENERATION}), then annotate it: `python annotate.py "
            f"--annotator NAME`, then `python annotate.py --export` to write "
            f"{TABLE5_ANNOTATIONS}."
        )
    annotations = json.load(open(TABLE5_ANNOTATIONS, encoding="utf-8"))
    entries = annotations.get(method_name)
    if not entries:
        raise NotImplementedError(
            f"{TABLE5_ANNOTATIONS} exists but has no entries for "
            f"{method_name!r} yet -- annotation for this method is incomplete."
        )
    result = {}
    for label, age in CTX:
        try:
            result[label] = compliance_rate(entries, age)
        except ValueError as e:
            raise NotImplementedError(str(e)) from e
    return result


# One function per Table 5 row. Each loads that method's gold annotations; none
# generates or grades output itself. The `mode` argument is unused and is kept
# so all rows share one signature.
def ccr_cpgtr(mode):
    return _ccr_from_annotations("CP-GTR (Ours)")


def ccr_unconstrained(mode):
    return _ccr_from_annotations("Unconstrained LLM")


def ccr_fewshot(mode):
    return _ccr_from_annotations("Few-Shot Prompted")


def ccr_rlaif(mode):
    return _ccr_from_annotations("Constitutional AI (RLAIF)")


def flag_violations(cert, G, ctx):
    """CP-GTR-Detect's detection step: find matches without applying them.

    Returns [(rule, match), ...], exactly the matches one_step() would fire
    for context `ctx`, but unfired. Rule set and matching are identical to
    CP-GTR's; the only difference is that a match is reported instead of used
    to rewrite the graph, which is what isolates repair from detection.
    """
    return [(r, m) for (r, m, _H) in one_step(cert, G, ctx)]


# max_tokens for the regeneration call in cpgtr_detect_repair(). It is set high
# enough that a full revised document is not cut off (the client default is
# 1024 tokens).
DETECT_REGEN_MAX_TOKENS = 4096

_DETECT_REGEN_PROMPT = """You are revising a drafted document to fix specific
compliance issues. Below is the original draft, followed by a list of
detected violations (in plain language). Rewrite the document so that none of
the listed violations apply. Keep everything else about the document
unchanged. Return only the revised document text, no commentary.

ORIGINAL DRAFT:
%s

The intended reader of this document is %s years old and located in %s.

DETECTED VIOLATIONS:
%s
"""


def cpgtr_detect_repair(complete, cert, parser, text, ctx, doc=None):
    """CP-GTR-Detect: flag violations with the rules, regenerate with the LLM.

    The certified rule set, parser and corpus are identical to CP-GTR's, but
    the rules only flag violations. The flagged document is handed back to the
    LLM to regenerate, the detect-then-regenerate division of labour used by
    constraint-, alignment- and execution-based systems. Returns the
    regenerated text, or `text` unchanged if nothing is flagged.

    `ctx` = (A, J) is used twice: flagging is context-gated (a rule's guard
    depends on the reader), and the regeneration prompt states the reader's
    age and jurisdiction so the rewrite targets the same context.

    `doc` is an optional pre-parsed Document for `text`, as in
    process_document(); a caller that already parsed this text can pass it to
    skip a second parse (an LLM call for LLMParser).
    """
    doc = doc if doc is not None else parser.parse(text)
    flagged = flag_violations(cert, doc.graph, ctx)
    if not flagged:
        return text   # nothing detected; nothing to regenerate
    violation_lines = "\n".join(
        f"- {r.template}" if r.template else f"- violates rule {r.name}"
        for r, _m in flagged
    )
    A, J = ctx
    prompt = _DETECT_REGEN_PROMPT % (text, A, J, violation_lines)
    return complete(prompt, max_tokens=DETECT_REGEN_MAX_TOKENS)


def ccr_cpgtr_detect(mode):
    return _ccr_from_annotations("CP-GTR-Detect (ablation)")


def generate_table5_grid(mode):
    """Generate the Table 5 outputs for every (prompt, context) pair.

    Reads the prompt grid docs/table4_pilot/grid.json and, for each pair,
    produces one output per method. Writes TABLE5_GENERATION (output texts)
    and TABLE5_BLIND_MAP (a seeded shuffle that maps the anonymous labels
    "A".."E" to method names; annotate.py never shows it to the annotator).
    Requires CPGTR_MODE=real.

    CP-GTR and CP-GTR-Detect use the small hand-written rule set r_PC, r_WD,
    r_SALE, r_RET, r_NOTIFY (cpgtr.library), not the larger extracted library:
    the pilot prompts and violation codebook (docs/table4_pilot/) are built
    around these rules, and the extracted rules have no codebook entries an
    annotator could mark against. The adversarial fixtures (r_BAD, r_RET_BAD,
    r_FLIP*) are certification test cases and are not used here.

    CP-GTR's arm parses with RuleParser (deterministic, no LLM calls). This
    keeps its cost near zero and avoids parser noise in the compliance
    comparison; it is a choice for this table, not the real-mode default
    elsewhere in this file.

    Drafting happens once per prompt, not once per pair: the Unconstrained,
    Few-Shot and RLAIF methods do not know the reader's age when drafting
    (only the compliance evaluation is context-dependent), and CP-GTR and
    CP-GTR-Detect both repair the same unconstrained draft that the
    Unconstrained method reports. CP-GTR-Detect's flagging is context-gated, so
    its regeneration runs once per pair.
    """
    if mode != "real":
        raise ValueError("generate_table5_grid requires CPGTR_MODE=real")

    grid = json.load(open(TABLE5_GRID, encoding="utf-8"))
    complete = get_real_complete()
    parser = RuleParser()
    cert, _log = admit([r_PC(), r_WD(), r_SALE(), r_RET(), r_NOTIFY()])

    by_prompt = {}
    for item in grid:
        by_prompt.setdefault(item["prompt_id"], []).append(item)

    generation = {}
    n_calls = 0
    for prompt_id, items in by_prompt.items():
        prompt_text = items[0]["prompt_text"]
        unconstrained = draft_unconstrained(complete, prompt_text)
        fewshot = draft_fewshot(complete, prompt_text)
        rlaif = critique_revise_rlaif(complete, unconstrained)
        n_calls += 3

        for item in items:
            pair_id = item["id"]
            ctx = (item["context"]["A"], item["context"]["J"])
            doc = parser.parse(unconstrained)
            flagged = flag_violations(cert, doc.graph, ctx)
            detect_out = cpgtr_detect_repair(complete, cert, parser, unconstrained, ctx)
            if flagged:
                n_calls += 1
            result = process_document(parser, cert, ctx, unconstrained)
            generation[pair_id] = {
                "prompt_id": prompt_id,
                "context": item["context"],
                "Unconstrained LLM": unconstrained,
                "Few-Shot Prompted": fewshot,
                "Constitutional AI (RLAIF)": rlaif,
                "CP-GTR-Detect (ablation)": detect_out,
                # scored_text is the unrepaired draft when the round-trip check
                # failed (the document is flagged for review), and the repaired
                # output otherwise; see cpgtr.realize.process_document().
                "CP-GTR (Ours)": result["scored_text"],
                # Internal diagnostic, not one of the five method keys mapped to
                # the labels A-E, so the annotator never sees it. It records
                # whether CP-GTR's round-trip check reached a normal form
                # ("enforcement soundness"), which is a different question from
                # compliance and is never used in place of gold labels.
                "_cpgtr_meta": {
                    "rules_fired": result["rules_fired"],
                    "round_trip_compliant": result["compliant"],
                },
            }

    os.makedirs("cache", exist_ok=True)
    with open(TABLE5_GENERATION, "w", encoding="utf-8") as f:
        json.dump(generation, f, indent=2, ensure_ascii=False)

    rng = random.Random(TABLE5_SEED)
    blind_map = {}
    for pair_id in generation:
        names = list(CCR_METHODS.keys())   # CCR_METHODS is defined below at module
        rng.shuffle(names)                  # level; it exists by the time this runs
        blind_map[pair_id] = dict(zip(["A", "B", "C", "D", "E"], names))
    with open(TABLE5_BLIND_MAP, "w", encoding="utf-8") as f:
        json.dump(blind_map, f, indent=2)

    print(f"[table5-gen] {len(grid)} pairs, {len(by_prompt)} prompts, "
          f"~{n_calls} real LLM calls -- wrote {TABLE5_GENERATION} and "
          f"{TABLE5_BLIND_MAP}")
    return generation


# Table 5 rows in display order: method name -> function returning its CCR.
CCR_METHODS = {
    "Unconstrained LLM":        ccr_unconstrained,
    "Few-Shot Prompted":        ccr_fewshot,
    "Constitutional AI (RLAIF)":ccr_rlaif,
    "CP-GTR-Detect (ablation)": ccr_cpgtr_detect,
    "CP-GTR (Ours)":            ccr_cpgtr,
}


def table5_ccr(mode):
    """Table 5: contextual compliance rate, one row per method.

    Each cell is the percentage of that method's outputs at the column's age
    (across the jurisdictions present) that carry no gold-labelled violation,
    with a bootstrap 95% confidence interval from cpgtr.eval_compliance. A
    method whose annotations are missing prints "-- not implemented --" with
    the reason; no rate is invented. Returns {method: cells or reason}.
    """
    print("\n=== Table 5: Contextual Compliance Rate (%) [95% CI] ===")
    hdr = f"{'Method':<28}" + "".join(f"{c:>22}" for c, _ in CTX)
    print(hdr); print("-" * len(hdr))
    results = {}
    for name, fn in CCR_METHODS.items():
        try:
            res = fn(mode)
            cells = "".join(
                f"{res[c][0]:>10.1f} [{res[c][1][0]},{res[c][1][1]}]".rjust(22)
                for c, _ in CTX)
            print(f"{name:<28}{cells}")
            results[name] = {c: {"rate": res[c][0], "ci": list(res[c][1])}
                              for c, _ in CTX}
        except NotImplementedError as e:
            print(f"{name:<28}{'-- not implemented --':>66}")
            print(f"    ({e})")
            results[name] = {"not_implemented": str(e)}
    return results


# --------------------------------------------------------------------------- #
# TABLE 6 — parser quality (tab:parser).                                      #
# --------------------------------------------------------------------------- #
def table6_parser(parser):
    """Table 6: parser precision, recall and F1 against gold graph parses.

    `parser` is scored on every entry of corpus/gold_parses.json with a
    Smatch-style triplet match (cpgtr.eval_parser.score_corpus). main() passes
    the deterministic RuleParser, so the table needs no API calls and is
    reproducible.

    There is no audited false-negative column. A human false-negative audit
    protocol is specified and released with the paper, but no audited rate is
    reported because that review has not been conducted, so the table prints
    nothing for it rather than a placeholder.
    """
    gold = json.load(open(GOLD, encoding="utf-8"))
    res = score_corpus(parser, gold)
    p, r, f = res["overall"]["all"]
    print("\n=== Table 6: SLG triplet extraction ===")
    print(f"{'Component':<24}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 48)
    print(f"{'SLG triplet extraction':<24}{p:8.2f}{r:8.2f}{f:8.2f}")
    print("No audited false-negative rate is reported: that review has not "
          "been conducted.")
    if res["fallback_count"]:
        # A nonzero count means the scores above are partly RuleParser's output
        # rather than `parser`'s on those entries, so they must not be reported
        # as a clean measurement of `parser` alone.
        print(f"WARNING: {res['fallback_count']}/{len(gold)} gold entries fell back to "
              f"RuleParser after exhausting retries (ids: {res['fallback_ids']}) -- "
              f"the P/R/F1 above is not purely `parser`'s own output.")
    return res


# --------------------------------------------------------------------------- #
# TABLE 7 (tab:ablation-a) — certification mechanism validation.              #
# TABLE 8 (tab:ablation-b) — certification deployment realism.                #
#                                                                              #
#   7. Mechanism validation: does the gate reject known-bad synthetic rules   #
#      (r_BAD, r_RET_BAD, r_FLIP1, r_FLIP2, defined in cpgtr/library.py for   #
#      this purpose) while admitting the worked-example rules (r_PC, r_WD,    #
#      r_SALE, r_RET, r_NOTIFY)? No LLM calls, fully deterministic, always    #
#      runnable. It does not use lib.synthesize(), an offline demo stand-in   #
#      that is never used for reported results.                               #
#   8. Deployment realism: the same four configurations run on the candidate  #
#      rules from one real Stage-1 extraction over corpus/provisions.json.    #
#      Requires CPGTR_MODE=real.                                              #
#                                                                              #
# Both tables use the same four configurations (toggling CPA and             #
# stratification) and the same holdout-host repair experiment; see            #
# ablation_run() for what each column means.                                  #
# --------------------------------------------------------------------------- #
def validation_candidates():
    """Mechanism-validation candidate pool for Table 7.

    r_PC/r_WD/r_SALE/r_RET are the paper's worked-example rules (the Box 2
    and Box 3 examples). r_NOTIFY forms a genuine creation-dependency cascade
    with r_RET (it can match only after r_RET fires; see its docstring in
    library.py), which is what makes a median cascade depth above 1 measurable.
    r_BAD/r_RET_BAD/r_FLIP1/r_FLIP2 are the seeded adversarial fixtures: r_BAD
    and r_RET_BAD are non-joinable-overlap fixtures (rejected by CPA, caught
    as "reject:non-joinable-CP"); r_FLIP1/r_FLIP2 are the stratification
    ping-pong pair (rejected as "reject:no-measure"). r_RET_BAD's wider guard
    (EU, UK vs. r_RET's EU-only) is what makes --CPA's admission of it
    OBSERVABLE in the repair experiment on a UK host, not just visible in the
    admission log -- see corpus/holdout_hosts_mechanism_probe.json.
    """
    from cpgtr.library import (r_PC, r_WD, r_SALE, r_RET, r_NOTIFY,
                                r_BAD, r_RET_BAD, r_FLIP1, r_FLIP2)
    return [r_PC(), r_WD(), r_SALE(), r_RET(), r_NOTIFY(),
            r_BAD(), r_RET_BAD(), r_FLIP1(), r_FLIP2()]


def ablation_run(candidates, use_cpa, use_strat, holdout, edgefree="keep",
                  budget_seconds=None):
    """Run one ablation configuration on a fixed candidate pool and measure it.

    Args:
        candidates: candidate rules; admit() sets `.rho` on them, so pass
            fresh instances if the same pool is used for another configuration.
        use_cpa: run the confluence (critical-pair) check.
        use_strat: run the global stratification; if False, a naive size
            measure (accept a rule if |R| <= |L|) replaces it.
        holdout: holdout-host dicts for the repair experiment.
        edgefree: "keep" (default) or "reject"; passed to admit(). "reject"
            rejects candidates whose left-hand side has no edge up front.
        budget_seconds: optional wall-clock budget for the whole call, admit()
            and run_repair() together, which share one absolute deadline. It
            matters when CPA is disabled: such a certificate can admit rules
            with no confluence guarantee, and run_repair()'s T_MAX is a step
            cap, not a time cap, so a growing cascade can run for a very long
            time. If the deadline is hit, the metrics come from the partial
            results and "budget_exceeded" and "progress_trace" are set, so a
            truncated run is never mistaken for a complete one.

    Returns dict:
      term_rate       : "100%" when stratification is enabled (it is what
                        certifies termination; CPA only checks confluence),
                        "---" when use_strat is False. It is set from the
                        configuration, not measured per run.
      nonterm         : number of confirmed non-termination incidents from
                        actually running repair on the holdout hosts (a repair
                        sequence exceeding T_MAX and confirmed by cycle
                        detection), not an admission-time count.
      strat_reject    : candidates rejected by the global stratification stage
                        (certify.stratify); structurally 0 when use_strat is
                        False.
      repair_success  : fraction of holdout hosts repaired within T_MAX.
      median_cascade  : median cascade depth d ("---" when termination is not
                        certified).
      budget_exceeded : True if admit() or run_repair() hit the deadline.
      repair_budget_exceeded_hosts : number of hosts cut short by the deadline.
      progress_trace  : list of (candidate_name, status) in admission order,
                        including "budget-exceeded" entries for candidates
                        admit() never reached.
      repair_per_host : per-host repair outcomes, for audit.
    """
    deadline = (time.time() + budget_seconds) if budget_seconds is not None else None
    cert, log = admit(candidates, enable_cpa=use_cpa, enable_strat=use_strat,
                       edgefree=edgefree, deadline=deadline)

    # Same absolute deadline (not a fresh budget_seconds): admit() and
    # run_repair() share one wall-clock allowance for the whole ablation row.
    rep = run_repair(cert, holdout, T_MAX, deadline=deadline)   # real repair experiment

    # Termination is certified by the stratification stage, not by CPA -- CPA
    # only checks confluence. Full pipeline and --CPA (both use_strat=True)
    # show 100%; --Strat and --CPA,--Strat (both use_strat=False) show '---'.
    term_certified = use_strat
    return {
        "term_rate":      "100%" if term_certified else "---",
        "nonterm":        rep["nonterm_incidents"],
        "strat_reject":   getattr(log, "strat_rejections"),
        "repair_success": rep["success_fraction"],
        "median_cascade": rep["median_cascade"] if term_certified else "---",
        "budget_exceeded": getattr(log, "budget_exceeded", False) or rep["budget_exceeded"] > 0,
        "repair_budget_exceeded_hosts": rep["budget_exceeded"],
        "progress_trace":  list(getattr(log, "notes", [])),
        "repair_per_host": rep["per_host"],
    }


# (label, flags). use_cpa toggles the confluence check; use_strat toggles the
# global stratification that certifies termination.
ABLATION_CONFIGS = [
    ("Full pipeline",     dict(use_cpa=True,  use_strat=True)),
    ("--CPA",             dict(use_cpa=False, use_strat=True)),
    ("--Strat",           dict(use_cpa=True,  use_strat=False)),
    ("--CPA,--Strat",     dict(use_cpa=False, use_strat=False)),
]

ABLATION_COLS = ["term_rate", "nonterm", "strat_reject",
                 "repair_success", "median_cascade"]
ABLATION_LABELS = ["Term.rate", "Non-term", "Strat.rej", "Repair", "Cascade d"]

# Table 7 only: the seeded adversarial fixtures (cpgtr/library.py), as opposed to
# the worked-example rules. See validation_candidates().
FIXTURE_NAMES = {"r_BAD", "r_RET_BAD", "r_FLIP1", "r_FLIP2"}


def _fmt_ablation_cell(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


class BudgetExceeded(RuntimeError):
    """Raised by _print_ablation_table() when an ablation row hit its deadline.

    It is a hard failure rather than a warning; see that function."""


def _print_ablation_table(title, rows, cols=ABLATION_COLS, labels=ABLATION_LABELS):
    """Print an ablation table, refusing if any row is a truncated run.

    Table 7 and Table 8 both pass their rows through this function before the
    numbers are printed or written to results.json. A row whose ablation_run()
    hit its wall-clock deadline reflects a partial certificate and/or a partial
    repair pass, not a complete result. Reporting it would break the integrity
    rule just as a placeholder cell would, so BudgetExceeded is raised before
    any of the table is printed, and the failure surfaces at its own call site.
    """
    truncated = [(name, m) for name, m in rows if m.get("budget_exceeded")]
    if truncated:
        detail = "; ".join(
            f"{name!r} ({sum(1 for _n, s in m.get('progress_trace', []) if s == 'budget-exceeded')} "
            f"candidate(s) unprocessed)"
            for name, m in truncated
        )
        raise BudgetExceeded(
            f"{title}: refusing to report -- {len(truncated)} configuration(s) hit "
            f"the wall-clock deadline before finishing: {detail}. A truncated "
            f"certification/repair run must not be reported as a result. "
            f"Re-run with a larger budget_seconds, or omit this table."
        )

    print(f"\n=== {title} ===")
    hdr = f"{'Configuration':<20}" + "".join(f"{l:>13}" for l in labels)
    print(hdr); print("-" * len(hdr))
    for name, m in rows:
        row = f"{name:<20}" + "".join(
            f"{_fmt_ablation_cell(m[c]):>13}" for c in cols)
        print(row)


def _guard_for_status(status):
    if status == "reject:no-measure":
        return "stratification"
    if status in ("reject:non-joinable-CP", "reject:cpa-nonterm"):
        return "CPA"
    return "other"


def unsound_admissions(candidates, use_cpa, use_strat, full_pipeline_rejections):
    """Count and log seeded fixtures admitted under (use_cpa, use_strat) that
    the full pipeline rejects.

    This makes the contribution of CPA and of stratification directly visible
    in Table 7. repair_success and median_cascade cannot show it: both are
    structurally identical for the full pipeline and --CPA, because both
    terminate every host whenever stratification is active.

    Args:
        candidates: fresh candidate list for this configuration (not reused
            across configurations, because admit() mutates .rho on its inputs).
        full_pipeline_rejections: {fixture_name: (status, guard)}, computed
            once from a full-pipeline run and reused for all four
            configurations, so each call needs only one admission run.

    Returns list of (fixture_name, guard, full_pipeline_status) for every
    seeded fixture admitted in this config that the full pipeline rejects.
    """
    cert, _log = admit(candidates, enable_cpa=use_cpa, enable_strat=use_strat)
    admitted_names = {r.name for r in cert}
    return [(name, guard, status)
            for name, (status, guard) in full_pipeline_rejections.items()
            if name in admitted_names]


def dependency_rate_comparison(candidates):
    """Report the stratification rejection rate under type-level (the default)
    and instance-level creation dependency, side by side, for one candidate set.

    Instance-level dependency (certify.creation_dependency_instance) is an
    option, not a replacement. This function does not change what any run
    certifies; it only reports the comparison. Each count is taken right after
    its stratify() call, before the next call overwrites .rho on the same
    objects. It is not called by main().
    """
    from cpgtr.certify import stratify

    stratify(candidates, instance_level=False)
    type_rejected = sum(1 for r in candidates if r.rho is None)

    stratify(candidates, instance_level=True)
    inst_rejected = sum(1 for r in candidates if r.rho is None)

    n = len(candidates)
    print("\n=== Stratification rejection rate: type-level vs instance-level "
          "creation dependency ===")
    print(f"  type-level (default):    {type_rejected}/{n} rejected "
          f"({type_rejected / n:.1%})" if n else "  (no candidates)")
    print(f"  instance-level (option): {inst_rejected}/{n} rejected "
          f"({inst_rejected / n:.1%})" if n else "")
    return {"n": n, "type_level_rejected": type_rejected,
            "instance_level_rejected": inst_rejected}


def table7a():
    """Table 7: certification mechanism validation. No LLM calls; always runnable.

    The candidate pool (validation_candidates()) mixes the worked-example rules
    with seeded adversarial fixtures. Each of the four configurations in
    ABLATION_CONFIGS admits from it and then repairs the toy holdout hosts plus
    the mechanism-probe hosts. Besides the shared ablation columns, an
    "Unsound" column counts the seeded fixtures admitted under that
    configuration although the full pipeline rejects them, and a per-fixture
    log names which guard (stratification or CPA) would have caught each one.
    That makes CPA's contribution directly observable, which repair_success
    and median_cascade cannot do (see unsound_admissions()).

    Returns {"rows": [(config, metrics)], "unsound_log": [...]}.
    """
    holdout = (json.load(open(HOLDOUT, encoding="utf-8"))
               + json.load(open(HOLDOUT_PROBE, encoding="utf-8")))

    _, full_log = admit(validation_candidates(), enable_cpa=True, enable_strat=True)
    full_rejections = {
        name: (status, _guard_for_status(status))
        for name, status in full_log.notes
        if name in FIXTURE_NAMES and status != "admit"
    }

    rows, unsound_log = [], []
    for name, flags in ABLATION_CONFIGS:
        candidates = validation_candidates()   # fresh instances per configuration
        m = ablation_run(candidates, holdout=holdout, **flags)
        unsound = unsound_admissions(validation_candidates(), flags["use_cpa"],
                                      flags["use_strat"], full_rejections)
        m["unsound"] = len(unsound)
        unsound_log.extend((name, fixture, guard, status) for fixture, guard, status in unsound)
        rows.append((name, m))

    _print_ablation_table(
        "Table 7: Certification mechanism validation "
        "(seeded adversarial fixtures + the manuscript's worked examples)",
        rows, cols=ABLATION_COLS + ["unsound"], labels=ABLATION_LABELS + ["Unsound"])

    print("\nPer-fixture unsound-admission log:")
    if not unsound_log:
        print("  (none -- no seeded fixture was ever admitted under a relaxed "
              "config that the full pipeline rejects)")
    for config_name, fixture, guard, status in unsound_log:
        print(f"  [{config_name:<14}] {fixture:<12} admitted -- full pipeline "
              f"catches it via {guard} ({status})")

    return {"rows": rows, "unsound_log": unsound_log}


# Prompts used to draft the real holdout hosts. They span the themes of the
# extracted rules across all five corpus sources, not just consent, retention
# and sale. Some are deliberately single-issue, some multi-issue (to produce
# cascades), and some already satisfy a negative application condition, which
# mirrors the variety of the hand-authored hosts in corpus/holdout_hosts.json.
REAL_HOST_PROMPTS = [
    "Draft a short privacy notice for a mobile app's sign-up flow: we collect "
    "the user's email to send product updates, and by signing up the user "
    "consents to this.",
    "Draft a section of a privacy policy describing how long we retain a "
    "user's account data after they stop using the service, with no fixed "
    "end date -- we keep it as long as needed for our records.",
    "Draft a section of a privacy policy describing that we share a user's "
    "personal information with third-party advertising partners for their "
    "own marketing purposes.",
    "Draft a consent section for a photo-sharing app aimed at families: "
    "uploading a photo means agreeing to our data collection and to our "
    "advertising partners receiving usage data for marketing.",
    "Draft a section of a privacy policy describing our data collection "
    "consent process together with our data retention policy, which keeps "
    "data indefinitely for service-improvement purposes.",
    "Draft a section of a privacy policy stating that a user may withdraw "
    "their consent at any time, and separately describing that we retain "
    "account data indefinitely after consent is withdrawn.",
    "Draft a consent section for a children's education app stating that we "
    "require a parent or legal guardian's consent before collecting a "
    "child's data.",
    "Draft a comprehensive privacy policy section covering: what personal "
    "data we collect during sign-up and consent, how long we retain it, and "
    "whether we share or sell it to third-party advertising partners.",
    "Draft a section of a privacy policy for a streaming service describing "
    "that we process a user's viewing data by collecting it upon sign-up "
    "consent, and that we sell aggregated interest data to advertising "
    "partners.",
    "Draft a section of a privacy policy describing our account deletion "
    "process: when a user closes their account, we retain their data for "
    "an unspecified period in case it is needed later.",
    "Draft a data-sharing disclosure for a children's mobile game: we "
    "collect data upon consent during account creation and share device "
    "identifiers with advertising partners.",
    "Draft a section of a privacy policy describing international data "
    "transfers: personal data collected during the consent process may be "
    "retained indefinitely on servers in multiple countries.",
]


def generate_real_holdout_hosts(mode):
    """Build corpus/holdout_hosts_real.json: holdout hosts from real policy text.

    Each REAL_HOST_PROMPTS entry is drafted by the LLM (via
    cpgtr.baselines.draft_unconstrained, with no compliance guidance, as a
    real-world drafter would) and parsed once by LLMParser into a graph, which
    becomes the host. The parser is only a one-time construction step: hosts
    are consumed as graphs and are not re-parsed during run_repair() (see
    eval_repair._host_to_graph), so parser noise does not enter the repair
    experiment itself. The hand-authored toy hosts in corpus/holdout_hosts.json
    are kept for Table 7 and regression checks.

    LLMParser is used rather than RuleParser for a structural reason.
    Schema-constrained extraction fixes the allowed node and edge types but not
    the topology that connects them, and many extracted rules need patterns
    that RuleParser's fixed, toy-library-shaped backbone never produces (for
    example Operator-permits->DataProcessing, or DataSubject-subjectOf->
    DataProcessing directly). Hosts built with RuleParser would mostly not
    trigger any real rule, so the repair experiment would measure
    non-coverage rather than repair. The trade-off is that LLMParser is
    model-based and not guaranteed identical from run to run (temperature 0
    makes it mostly reproducible).

    Contexts cycle round-robin through admit.CONTEXTS (the 3-age by
    3-jurisdiction reporting grid), so the set also covers the UK.
    """
    if mode != "real":
        raise ValueError("generate_real_holdout_hosts requires CPGTR_MODE=real")

    complete = get_real_complete()
    parser = LLMParser(complete)
    hosts = []
    fallback_hosts = []
    for i, prompt in enumerate(REAL_HOST_PROMPTS):
        text = draft_unconstrained(complete, prompt)
        doc = parser.parse(text)
        host_id = f"real{i + 1:03d}"
        if getattr(doc, "parse_fallback", False):
            fallback_hosts.append(host_id)
        A, J = CONTEXTS[i % len(CONTEXTS)]
        hosts.append({
            "id": host_id,
            "description": f"Real-drafted (Cerebras/gpt-oss-120b), parsed via "
                            f"LLMParser (same model). Source prompt: {prompt}",
            "context": {"A": A, "J": J},
            "nodes": [{"id": n, "type": t} for n, t in doc.graph.nodes.items()],
            "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                      for e, (s, t, ty) in doc.graph.edges.items()],
            "source_prompt": prompt,
            "source_text": text,
            "parse_fallback": doc.parse_fallback,
        })

    os.makedirs("corpus", exist_ok=True)
    with open(HOLDOUT_REAL, "w", encoding="utf-8") as f:
        json.dump(hosts, f, indent=2, ensure_ascii=False)

    empty = sum(1 for h in hosts if not h["nodes"])
    print(f"[holdout-real] wrote {len(hosts)} hosts to {HOLDOUT_REAL} "
          f"({empty} parsed to an empty graph -- LLMParser/fallback found "
          f"nothing schema-relevant in that draft)")
    if fallback_hosts:
        # These hosts' nodes and edges are RuleParser's output, not
        # LLMParser's, even after retrying.
        print(f"WARNING: {len(fallback_hosts)}/{len(hosts)} hosts fell back to "
              f"RuleParser after exhausting retries: {fallback_hosts}")
    return hosts


def table8b(mode):
    """Table 8: certification deployment realism. Requires CPGTR_MODE=real.

    Runs the four ABLATION_CONFIGS on the unmodified candidate library from the
    real extraction pass (get_real_candidates()) and repairs the real-derived
    holdout hosts in HOLDOUT_REAL (see generate_real_holdout_hosts()), rather
    than the hand-authored hosts in HOLDOUT that Table 7 uses to test the seeded
    fixtures. Returns the rows, or None when not run.
    """
    print("\n=== Table 8: Certification deployment realism (real extraction) ===")
    if mode != "real":
        print("  -- not run: requires CPGTR_MODE=real (see cpgtr/llm.py for "
              "API key setup) --")
        return None
    if not os.path.exists(HOLDOUT_REAL):
        raise FileNotFoundError(
            f"{HOLDOUT_REAL} does not exist -- run generate_real_holdout_hosts"
            f"('real') first. table7a() still uses the original "
            f"{HOLDOUT} for toy-library mechanism validation; this table "
            f"needs the real-policy-derived set specifically."
        )
    candidates = get_real_candidates()   # shared with Table 4; see its docstring
    holdout = json.load(open(HOLDOUT_REAL, encoding="utf-8"))
    rows = []
    for name, flags in ABLATION_CONFIGS:
        m = ablation_run(candidates, holdout=holdout, **flags)
        rows.append((name, m))
    _print_ablation_table("Table 8: Certification deployment realism", rows)
    return rows


# --------------------------------------------------------------------------- #
def _dry_run_report(mode):
    """Check that the environment is ready, without any API call or table run.

    Reports the mode and provider, whether the provider's API key is set (real
    mode), and which input and cache files exist. It catches a missing key or
    file before a real run spends budget.
    """
    from cpgtr.llm import PROVIDERS, load_dotenv
    load_dotenv()   # loads .env into the environment; the key checks below
                     # read the environment, so they need this first
    provider = os.environ.get("CPGTR_PROVIDER", "cerebras")
    print(f"[dry-run] MODE={mode!r} CPGTR_PROVIDER={provider!r}")
    if mode == "real":
        if provider not in PROVIDERS:
            print(f"  [FAIL] unknown provider {provider!r}; choose from "
                  f"{sorted(PROVIDERS)}")
        else:
            cfg = PROVIDERS[provider]
            key_set = bool(os.getenv(cfg["api_key_env"]))
            print(f"  [{'OK' if key_set else 'FAIL'}] {cfg['api_key_env']} "
                  f"{'is set' if key_set else 'is NOT set'} "
                  f"(model would be {cfg['default_model']!r})")
    for label, path in [
        ("provisions corpus", PROVISIONS), ("gold parses", GOLD),
        ("toy holdout hosts", HOLDOUT), ("mechanism probe hosts", HOLDOUT_PROBE),
        ("real-derived holdout hosts", HOLDOUT_REAL),
        ("real extraction cache", EXTRACTION_CACHE),
        ("Table 5 annotations", TABLE5_ANNOTATIONS),
    ]:
        exists = os.path.exists(path)
        print(f"  [{'OK' if exists else 'MISSING'}] {label}: {path}")
    print(f"[dry-run] EXTRACT_SEED={EXTRACT_SEED} EXTRACT_SAMPLE_N={EXTRACT_SAMPLE_N}")
    print("[dry-run] no API calls made, no tables run.")


def write_results_json(mode, table_results, path=RESULTS_JSON):
    """Write one results file with every table's measured numbers and run metadata.

    The metadata records the mode, provider, model, temperature and extraction
    sampling settings so that a result is never ambiguous about how it was
    produced. legisafe_eval.py reads this file and re-renders the tables; it
    never recomputes anything.
    """
    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "provider": None,
        "model": None,
        "temperature": 0.0,   # cpgtr.llm.make_complete's default, used for every
                               # real call; not an independent setting
        "extract_seed": EXTRACT_SEED,
        "extract_sample_n": EXTRACT_SAMPLE_N,
    }
    if mode == "real":
        complete = get_real_complete()
        meta["provider"] = complete.provider
        meta["model"] = complete.model

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tables": table_results}, f, indent=2,
                   ensure_ascii=False, default=str)
    print(f"\n[results-json] wrote {path}")


def main():
    """Run every table, then write the results artifact.

    Each table is independent: if one table's data is unavailable (demo mode,
    or real mode without API access), the others still run and print whatever
    they can measure. The integrity rule (raise rather than fabricate) is
    enforced inside each table function.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="validate config/files, make no API calls, run no tables")
    ap.add_argument("--results-json", metavar="PATH", default=RESULTS_JSON,
                     help=f"where to write the consolidated results artifact "
                          f"(default: {RESULTS_JSON})")
    args = ap.parse_args()

    print(f"MODE = {MODE}  (set CPGTR_MODE=real to call the configured LLM provider)")

    if args.dry_run:
        _dry_run_report(MODE)
        return

    table_results = {
        "table4": table4(MODE),
        "table5": table5_ccr(MODE),
        # Table 6 always scores the deterministic RuleParser, in real mode too.
        # Scoring build_parser(MODE) would use the sampling-based LLMParser in
        # real mode, making the numbers vary between runs.
        "table6": table6_parser(RuleParser()),
        "table7a": table7a(),
        "table8b": table8b(MODE),
    }
    write_results_json(MODE, table_results, path=args.results_json)


if __name__ == "__main__":
    main()
