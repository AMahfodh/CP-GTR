"""Shared evaluation harness for the six-arm compliance experiment.

Supports: the contextual-compliance evaluation of the paper (CCR and
violations-per-document tables, the per-code table, and the annotation
workflow that feeds them). This is a library module: it is imported by the
generation, annotation-sheet, coverage and results scripts and is not run
directly.

It provides:
  - frozen-library loading, gated on the SHA-256 recorded in
    analysis/phase2_library.json, and reconstruction of live rule objects
    from cache/canonical_extraction_cache.jsonl;
  - the template override that turns extracted drafter-style instructions
    into reader-facing notice sentences;
  - text normalisation and the annotation unit (text hash, age, jurisdiction),
    so that identical texts are labelled once per context and the label is
    inherited by every document that shares the unit;
  - blind annotation-sheet construction and label propagation;
  - run_six_arms_for_pair(): the six arms for one (prompt, context) pair.
    The paper's critique-and-revise arm is named "Constitutional AI (RLAIF)"
    and its unreviewed-library arm "CP-GTR-Ungated" in the code and data
    files. CP-GTR variants repair the same unconstrained draft, and a
    document flagged by the round-trip check is scored as its unrepaired
    draft.

dry_run_pilot() exercises the plumbing offline against the cached pilot
generation, using the deterministic RuleParser instead of the model-based
parser, so it validates the harness code paths but is not a measurement.
"""
from __future__ import annotations
import hashlib
import json
import random
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.extract import _load_cache, json_to_rule
from cpgtr.parse import RuleParser, LLMParser
from cpgtr.realize import process_document
from cpgtr.rules import one_step
from cpgtr.graph import Graph
from run_e2e import flag_violations, cpgtr_detect_repair

PHASE2_LIBRARY_PATH = REPO_ROOT / "analysis" / "phase2_library.json"
CANONICAL_CACHE = REPO_ROOT / "cache" / "canonical_extraction_cache.jsonl"

# The six arms, in the order used in the paper's tables. The library is None for
# the three non-CP-GTR arms (they never see a certified rule set).
ARM_LIBRARY = {
    "Unconstrained": None,
    "Few-Shot Prompted": None,
    "Constitutional AI (RLAIF)": None,
    "CP-GTR-Detect": "LIB-FAITHFUL-P",
    "CP-GTR": "LIB-FAITHFUL-P",
    "CP-GTR-Ungated": "LIB-FULL",
}

OVER_REQUIREMENT = "OVER_REQUIREMENT"


# --------------------------------------------------------------------------- #
# Library loading, gated on the frozen SHA-256: refuse to run on any mismatch. #
# --------------------------------------------------------------------------- #
class LibraryHashMismatch(RuntimeError):
    """Raised when a library's recorded SHA-256 in phase2_library.json does
    not match a fresh hash of its own recorded rule list, i.e. the frozen
    file is internally inconsistent (hand-edited, corrupted, or partially
    regenerated). This is a hard failure rather than a warning: a run must
    never proceed on a library that differs from the one it claims to use."""


def _rows_sha256(rows):
    """Serialization used for phase2_library.json's SHA-256 values: sort the
    rule rows by content_hash, then hash json.dumps(sort_keys=True)."""
    serialized = json.dumps(sorted(rows, key=lambda x: x["content_hash"]), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_frozen_library(library_id, library_path=PHASE2_LIBRARY_PATH):
    """Load `library_id` (e.g. "LIB-FAITHFUL-P", "LIB-FULL") from
    phase2_library.json, verify its recorded SHA-256 against a fresh hash of
    its own recorded rule rows, and return (rule_ids, record). Raises
    LibraryHashMismatch on any mismatch -- refuses to run rather than
    proceeding on an unverified library."""
    data = json.loads(Path(library_path).read_text(encoding="utf-8"))
    if library_id not in data:
        raise KeyError(f"{library_id!r} not found in {library_path}")
    record = data[library_id]
    recomputed = _rows_sha256(record["rules"])
    if recomputed != record["sha256"]:
        raise LibraryHashMismatch(
            f"{library_id}: recorded sha256={record['sha256']} but recomputed "
            f"sha256={recomputed} from its own rule list -- refusing to run"
        )
    rule_ids = {row["content_hash"] for row in record["rules"]}
    return rule_ids, record


def build_rules_for_library(library_id, library_path=PHASE2_LIBRARY_PATH,
                             cache_path=CANONICAL_CACHE):
    """Hash-verify `library_id` (raises LibraryHashMismatch on failure), then
    reconstruct live Rule objects for exactly that frozen rule_id set by
    rebuilding every candidate from cache/canonical_extraction_cache.jsonl
    and filtering on content_hash() membership. Filtering on the frozen id
    list, rather than re-running admit(), keeps the loaded library tied to
    the recorded hash even if the admission code changes later."""
    rule_ids, record = load_frozen_library(library_id, library_path)
    cache = _load_cache(str(cache_path))
    rules = []
    for h, entry in cache.items():
        r = json_to_rule(entry["spec"], source=entry["source"], enforce_topology=True)
        if r.content_hash() in rule_ids:
            rules.append(r)
    if len(rules) != len(rule_ids):
        raise LibraryHashMismatch(
            f"{library_id}: {len(rule_ids)} rule_ids recorded but only "
            f"{len(rules)} reconstructed from {cache_path} -- cache drift, refusing to run"
        )
    return rules, record


TEMPLATE_CORRECTIONS_PATH = REPO_ROOT / "analysis" / "realization_templates_corrections.json"


def apply_template_corrections(rules, corrections_path=TEMPLATE_CORRECTIONS_PATH):
    """Override each rule's `template` with its reader-facing notice sentence.

    As extracted, a rule's `template` is a drafter-facing instruction ("Add a
    requirement for verifiable parental consent...") rather than a sentence
    that can appear in a privacy notice. This replaces `.template` in place,
    keyed by `content_hash()`. content_hash() covers the L/K/R structure,
    name and source but not the template text, so the frozen library hashes
    in phase2_library.json stay valid. The raw extraction cache is left
    untouched: this is a load-time override, not an edit to extraction
    provenance. It is deliberately not part of build_rules_for_library(), so
    callers that want the as-extracted templates (e.g. a fidelity review)
    simply do not call it. Returns `rules` for chaining. Raises if a rule has
    no recorded correction, so drift fails loudly instead of being skipped."""
    corrections = json.loads(Path(corrections_path).read_text(encoding="utf-8"))
    missing = [r.name for r in rules if r.content_hash() not in corrections]
    if missing:
        raise KeyError(f"no template correction recorded for: {missing}")
    for r in rules:
        r.template = corrections[r.content_hash()]
    return rules


# --------------------------------------------------------------------------- #
# Text normalisation, hashing, annotation units, label inheritance.           #
# --------------------------------------------------------------------------- #
def normalize_text(text):
    """Unicode NFC, whitespace collapsed, trailing space stripped (in that
    order). Texts equal after this normalisation share an annotation unit."""
    t = unicodedata.normalize("NFC", text)
    t = " ".join(t.split())
    return t.rstrip()


def text_hash(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def annotation_unit(text, A, J):
    """The annotation unit is (text_hash, A, J): identical texts under
    different contexts are labelled per context, because compliance depends
    on the reader's age A and jurisdiction J."""
    return (text_hash(text), A, J)


def build_scored_documents(generation):
    """generation: {pair_id: {"context": {"A","J"}, arm_name: text, ...}}
    (one entry per (prompt, context) pair: the 90-pair grid of the full
    evaluation, 15 pairs in the early pilot). Returns a flat list of
    scored-document dicts, one per (pair, arm): {pair_id, arm, A, J, text,
    unit}."""
    docs = []
    for pair_id, entry in generation.items():
        A, J = entry["context"]["A"], entry["context"]["J"]
        for arm in ARM_LIBRARY:
            text = entry.get(arm)
            if text is None:
                continue
            docs.append({
                "pair_id": pair_id, "arm": arm, "A": A, "J": J,
                "text": text, "unit": annotation_unit(text, A, J),
            })
    return docs


def unique_units(scored_documents):
    """De-duplicated (text_hash, A, J) units across all scored documents --
    what the annotator actually has to label."""
    return sorted(set(d["unit"] for d in scored_documents))


def build_blind_annotation_sheet(scored_documents, seed):
    """One row per UNIQUE unit, not per scored document. Rows carry no arm
    names and no counts, so nothing reveals how many arms produced a unit and
    annotators stay blind to the arm. Order is randomised with a recorded
    seed. Returns (rows, seed); rows have only {row_id, text, A, J},
    deliberately nothing else."""
    units = unique_units(scored_documents)
    # One representative text per unit: all documents sharing a unit have, by
    # construction, the same normalized text, so any of them will do. Use the
    # first encountered for a stable, reproducible sheet.
    text_for_unit = {}
    for d in scored_documents:
        text_for_unit.setdefault(d["unit"], d["text"])

    rng = random.Random(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)

    rows = []
    for i, (th, A, J) in enumerate(shuffled):
        rows.append({
            "row_id": i,
            "text": text_for_unit[(th, A, J)],
            "A": A, "J": J,
            # Violation columns are added later by the workbook builders.
        })
    return rows, seed


def propagate_labels(scored_documents, unit_labels):
    """unit_labels: {(text_hash, A, J): [violation codes]} (or any codebook
    label the annotator recorded per unit). Returns
    {(pair_id, arm): labels} for EVERY scored document whose unit has a
    label -- a document whose unit isn't labelled yet is simply absent from
    the return value, not defaulted to anything."""
    out = {}
    for d in scored_documents:
        if d["unit"] in unit_labels:
            out[(d["pair_id"], d["arm"])] = unit_labels[d["unit"]]
    return out


def inheritance_report(scored_documents):
    """Report scored documents, unique annotation units, and (per arm) how
    many of that arm's documents share a unit with
    at least one OTHER document already counted -- i.e. how many were
    inherited rather than needing a fresh label. The first document (by
    stable input order) to use a given unit is the "source" label; every
    later document on the same unit inherits."""
    seen_units = set()
    per_arm_total = {}
    per_arm_inherited = {}
    for d in scored_documents:
        per_arm_total[d["arm"]] = per_arm_total.get(d["arm"], 0) + 1
        if d["unit"] in seen_units:
            per_arm_inherited[d["arm"]] = per_arm_inherited.get(d["arm"], 0) + 1
        else:
            seen_units.add(d["unit"])
    return {
        "n_scored_documents": len(scored_documents),
        "n_unique_units": len(unique_units(scored_documents)),
        "per_arm_total": per_arm_total,
        "per_arm_inherited": per_arm_inherited,
    }


# --------------------------------------------------------------------------- #
# Six-arm generation and scoring for one (prompt, context) pair. Needs a live   #
# `complete` and the canonical `LLMParser`.                                    #
# --------------------------------------------------------------------------- #
def run_six_arms_for_pair(complete, parser, cpgtr_cert, cpgtr_ungated_cert,
                           unconstrained_text, fewshot_text, rlaif_text, ctx,
                           flag_fn, detect_repair_fn):
    """Build all 6 arms' texts for one (prompt, context) pair, given the
    three base generations (unconstrained/fewshot/rlaif) already produced by
    the caller.

    `fewshot_text` and `rlaif_text` must be generated once per (prompt,
    context) pair, via `draft_fewshot(complete, prompt_text, A, J)` and
    `critique_revise_rlaif(complete, unconstrained_text, A, J)`, because those
    baselines have no other mechanism for context sensitivity (see
    cpgtr.baselines._context_line). That is 90 calls each over the full grid.
    `unconstrained_text` alone is context-blind and is drafted once per
    prompt (10 calls), then reused across the 9 contexts. It is the base
    draft every CP-GTR variant repairs; their context sensitivity comes
    entirely from the certified rules' Phi gates at repair time.
    `detect_repair_fn` (`cpgtr_detect_repair`) receives `ctx` and threads it
    into its own regeneration prompt.

    `flag_fn` and `detect_repair_fn` are injected so this function has no
    direct LLM dependency beyond what is passed in.

    All CP-GTR variants repair the same `unconstrained_text`. Scoring follows
    the flagged-document rule: a document whose repair failed the round-trip
    check is scored as its unrepaired draft. This applies to CP-GTR and
    CP-GTR-Ungated through process_document()'s scored_text field.
    CP-GTR-Detect never mechanically repairs: its output is the draft (or a
    regenerated draft when flagged) together with a separate flag.

    The draft is parsed once into `host_doc` and shared by the Detect
    flagging step and both process_document() calls, via their `doc=`
    parameter. This saves two model calls per pair and guarantees all three
    operate on the same graph. The round-trip re-parses inside
    process_document() (of each arm's own repaired text) and the conditional
    re-parse inside cpgtr_detect_repair() are not shared, since each parses
    different text.
    """
    host_doc = parser.parse(unconstrained_text)
    detect_matches = flag_fn(cpgtr_cert, host_doc.graph, ctx)
    detect_flagged = bool(detect_matches)
    detect_text = detect_repair_fn(complete, cpgtr_cert, parser, unconstrained_text, ctx, doc=host_doc) \
        if detect_flagged else unconstrained_text

    cpgtr_result = process_document(parser, cpgtr_cert, ctx, unconstrained_text, doc=host_doc)
    ungated_result = process_document(parser, cpgtr_ungated_cert, ctx, unconstrained_text, doc=host_doc)

    return {
        "Unconstrained": unconstrained_text,
        "Few-Shot Prompted": fewshot_text,
        "Constitutional AI (RLAIF)": rlaif_text,
        "CP-GTR-Detect": detect_text,
        "CP-GTR": cpgtr_result["scored_text"],
        "CP-GTR-Ungated": ungated_result["scored_text"],
        "_meta": {
            "CP-GTR-Detect_flagged": detect_flagged,
            "CP-GTR-Detect_rules_fired": [r.name for r, _m in detect_matches],
            "CP-GTR_flagged": not cpgtr_result["compliant"],
            "CP-GTR-Ungated_flagged": not ungated_result["compliant"],
            "CP-GTR_rules_fired": cpgtr_result["rules_fired"],
            "CP-GTR-Ungated_rules_fired": ungated_result["rules_fired"],
            # Per-document parse diagnostics for the shared host parse and
            # each arm's own round-trip parse. They let a reader check whether
            # the parser under-parses fresh drafts (a recall risk the small
            # gold set cannot measure), beyond the aggregate flag bits above.
            "host_parse_fallback": host_doc.parse_fallback,
            "host_parse_attempts": host_doc.parse_attempts,
            "host_parse_fallback_reason": host_doc.parse_fallback_reason,
            "host_n_nodes": len(host_doc.graph.nodes),
            "host_n_edges": len(host_doc.graph.edges),
            # Negation-filter drops for the shared host parse and for each
            # arm's own round-trip re-parse.
            "host_negation_drops": host_doc.negation_drops,
            "CP-GTR_round_trip_passed": cpgtr_result["compliant"],
            "CP-GTR_round_trip_parse_fallback": cpgtr_result["round_trip_parse_fallback"],
            "CP-GTR-Ungated_round_trip_passed": ungated_result["compliant"],
            "CP-GTR-Ungated_round_trip_parse_fallback": ungated_result["round_trip_parse_fallback"],
            # Edge types of the round-trip re-parse of the repaired text, to
            # check whether a "grants" edge (ConsentProcess -> DataProcessing)
            # survives the re-parse.
            "CP-GTR_redoc_edge_types": cpgtr_result["redoc_edge_types"],
            "CP-GTR-Ungated_redoc_edge_types": ungated_result["redoc_edge_types"],
            # Per-document provenance-verification detail (see
            # cpgtr.realize.realize()): whether provenance was verified, the
            # attributed and verified sentence for each attempted deletion,
            # and whether a verification failure caused a flag.
            "CP-GTR_provenance_verified": cpgtr_result["provenance_verified"],
            "CP-GTR_provenance_flagged": cpgtr_result["provenance_flagged"],
            "CP-GTR_provenance_verification_log": cpgtr_result["provenance_verification_log"],
            "CP-GTR-Ungated_provenance_verified": ungated_result["provenance_verified"],
            "CP-GTR-Ungated_provenance_flagged": ungated_result["provenance_flagged"],
            "CP-GTR-Ungated_provenance_verification_log": ungated_result["provenance_verification_log"],
            "CP-GTR_round_trip_negation_drops": cpgtr_result["round_trip_negation_drops"],
            "CP-GTR-Ungated_round_trip_negation_drops": ungated_result["round_trip_negation_drops"],
        },
    }


def dry_run_pilot(old_generation_path=None, old_annotations_path=None):
    """Plumbing test on the cached 15-pair pilot: no API calls. This is not
    a measurement of the evaluation. It makes three deliberate substitutions:

    1. Unconstrained/Few-Shot/Critique texts are reused verbatim from
       cache/table5_generation.json. That pilot drafted the Few-Shot and
       critique texts once per prompt rather than once per (prompt, context)
       pair, so this dry run does not exercise the per-context generation
       that run_six_arms_for_pair() expects of the real run.
    2. Hosts and drafts are parsed with RuleParser(), not the model-based
       LLMParser, which needs a live model call. The harness functions take
       `parser` as a parameter, so the real run passes
       LLMParser(complete, enforce_topology=True) without code changes.
    3. CP-GTR-Detect's regeneration step (cpgtr_detect_repair, which needs
       `complete`) is not called. The flag is recorded and the draft is
       returned with a marked placeholder.

    What is exercised for real: library loading and hash verification, live
    rule reconstruction, RuleParser-based flagging and repair against
    LIB-FAITHFUL-P and LIB-FULL, flagged-document scoring, text normalisation
    and hashing, annotation-unit construction, the blind-sheet builder, and
    label propagation checked against the real pilot annotations.
    """
    old_gen_path = old_generation_path or (REPO_ROOT / "cache" / "table5_generation.json")
    old_ann_path = old_annotations_path or (REPO_ROOT / "cache" / "table5_annotations.json")
    old_generation = json.loads(Path(old_gen_path).read_text(encoding="utf-8"))
    old_annotations = json.loads(Path(old_ann_path).read_text(encoding="utf-8"))

    cpgtr_rules, cpgtr_rec = build_rules_for_library("LIB-FAITHFUL-P")
    ungated_rules, ungated_rec = build_rules_for_library("LIB-FULL")
    parser = RuleParser()   # offline stand-in for the model-based LLMParser

    new_generation = {}
    for pair_id, old_entry in old_generation.items():
        ctx = (old_entry["context"]["A"], old_entry["context"]["J"])
        unconstrained = old_entry["Unconstrained LLM"]

        detect_matches = flag_violations(cpgtr_rules, parser.parse(unconstrained).graph, ctx)
        detect_flagged = bool(detect_matches)
        detect_text = (unconstrained + "  [DRY-RUN PLACEHOLDER: real run would "
                        "call cpgtr_detect_repair() here]") if detect_flagged else unconstrained

        cpgtr_result = process_document(parser, cpgtr_rules, ctx, unconstrained)
        ungated_result = process_document(parser, ungated_rules, ctx, unconstrained)

        new_generation[pair_id] = {
            "context": old_entry["context"],
            "Unconstrained": unconstrained,
            "Few-Shot Prompted": old_entry["Few-Shot Prompted"],
            "Constitutional AI (RLAIF)": old_entry["Constitutional AI (RLAIF)"],
            "CP-GTR-Detect": detect_text,
            "CP-GTR": cpgtr_result["scored_text"],
            "CP-GTR-Ungated": ungated_result["scored_text"],
            "_meta": {
                "CP-GTR-Detect_flagged": detect_flagged,
                "CP-GTR_flagged": not cpgtr_result["compliant"],
                "CP-GTR-Ungated_flagged": not ungated_result["compliant"],
            },
        }

    scored_documents = build_scored_documents(new_generation)
    report = inheritance_report(scored_documents)
    sheet, seed = build_blind_annotation_sheet(scored_documents, seed=2026)

    # Verify label propagation against the pilot labels.
    # Pilot format: {method: [{"prompt_id","age","jurisdiction","violations"}]}.
    # The lookup is restricted to the three arms whose text is unchanged
    # between the pilot and this generation (Unconstrained, Few-Shot,
    # critique-and-revise): only for those is "reproduces the pilot labels
    # exactly" meaningful. The CP-GTR arms now produce different text (new
    # libraries, different parser), so the old labels do not apply to them.
    old_method_name = {
        "Unconstrained": "Unconstrained LLM",
        "Few-Shot Prompted": "Few-Shot Prompted",
        "Constitutional AI (RLAIF)": "Constitutional AI (RLAIF)",
    }
    old_lookup = {}
    for method, entries in old_annotations.items():
        for e in entries:
            old_lookup[(method, e["prompt_id"], e["age"], e["jurisdiction"])] = e["violations"]

    # Build unit_labels from the pilot annotations for exactly those three
    # arms, keyed by the annotation unit computed from the (unchanged) text.
    unit_labels = {}
    for d in scored_documents:
        if d["arm"] not in old_method_name:
            continue
        prompt_id = new_generation[d["pair_id"]].get("prompt_id", old_generation[d["pair_id"]]["prompt_id"])
        key = (old_method_name[d["arm"]], prompt_id, d["A"], d["J"])
        if key in old_lookup:
            unit_labels[d["unit"]] = old_lookup[key]

    propagated = propagate_labels(scored_documents, unit_labels)

    n_checked = 0
    n_exact_match = 0
    mismatches = []
    for d in scored_documents:
        if d["arm"] not in old_method_name:
            continue
        prompt_id = old_generation[d["pair_id"]]["prompt_id"]
        old_key = (old_method_name[d["arm"]], prompt_id, d["A"], d["J"])
        if old_key not in old_lookup:
            continue
        n_checked += 1
        expected = old_lookup[old_key]
        got = propagated.get((d["pair_id"], d["arm"]))
        if got == expected:
            n_exact_match += 1
        else:
            mismatches.append({"pair_id": d["pair_id"], "arm": d["arm"],
                                "expected": expected, "got": got})

    return {
        "n_pairs": len(old_generation),
        "inheritance_report": report,
        "blind_sheet_rows": len(sheet),
        "blind_sheet_seed": seed,
        "label_propagation_check": {
            "n_checked": n_checked, "n_exact_match": n_exact_match,
            "mismatches": mismatches,
        },
        "libraries_used": {
            "CP-GTR": {"library": "LIB-FAITHFUL-P", "n_rules": len(cpgtr_rules),
                       "sha256": cpgtr_rec["sha256"]},
            "CP-GTR-Ungated": {"library": "LIB-FULL", "n_rules": len(ungated_rules),
                               "sha256": ungated_rec["sha256"]},
        },
        "new_generation": new_generation,
        "sheet_sample": sheet[:3],
    }


def flag_rate(generation, arm):
    """Fraction of pairs where `arm` was flagged (reported as its own column
    in the CCR table). Only meaningful for CP-GTR-family arms."""
    key = f"{arm}_flagged"
    flags = [entry["_meta"][key] for entry in generation.values() if "_meta" in entry]
    if not flags:
        return None
    return sum(1 for f in flags if f) / len(flags)
