"""Shared configuration loading and provenance capture for the generation runs.

Supports: generation of the 90 (prompt, context) pairs and 540 documents that
the compliance tables (CCR, violations per document, per-code counts) score.

Library module, imported by phase2_generation_run_AD.py,
phase2_generation_run.py, phase2_final_results_part2.py and
phase2_taskY_coverage.py; it is never run directly.

  load_prompts()   parses the ten evaluation prompts and two practice prompts
                   from docs/phase2_prompts_v1.md.
  assert_config()  loads and verifies every artifact a generation run depends
                   on (frozen libraries by SHA-256 and rule count, template
                   corrections by SHA-256, codebook self-hash, parser token
                   limit, parser configuration) and refuses to run on any
                   mismatch. It also builds the live model client, so it
                   needs a configured provider key (see cpgtr/llm.py).

Inputs : docs/phase2_prompts_v1.md, schema/codebook_v2.json,
         analysis/realization_templates_corrections.json,
         analysis/phase2_library.json, cache/canonical_extraction_cache.jsonl.
Outputs: none (returns a configuration dictionary).
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROMPTS_PATH = REPO_ROOT / "docs" / "phase2_prompts_v1.md"
CODEBOOK_PATH = REPO_ROOT / "schema" / "codebook_v2.json"
CORRECTIONS_PATH = REPO_ROOT / "analysis" / "realization_templates_corrections.json"

# Expected values of the frozen configuration. assert_config() raises
# AssertionError if any of these does not hold.
EXPECTED = {
    "LIB-FAITHFUL-P-v2_sha256": "59d44de9a7c2f54a2a2c0bad99deb95462849bc7c86ab4b8f5b89a974d6a6129",
    "LIB-FAITHFUL-P-v2_n_rules": 8,
    "LIB-FULL_n_rules": 31,
    "corrections_sha256": "e4121e77ac0b1fee01f652a5981e96b688dc744b8116eb5d144fcf8c3808cc9d",
    "parse_max_tokens": 8192,
}


def load_prompts():
    """Extract the 10 evaluation prompts (P1-P10) and both practice prompts
    (PRACTICE-1/2) from docs/phase2_prompts_v1.md by parsing the markdown
    directly, so the text cannot drift from the frozen prompt file through
    hand transcription. Returns (eval_prompts, practice_prompts), each
    {id: text}, in file order (P1..P10)."""
    text = PROMPTS_PATH.read_text(encoding="utf-8")
    eval_start = text.index("## Evaluation prompts (10)")
    practice_start = text.index("## Practice prompts")
    notes_start = text.index("## Notes for review")
    eval_section = text[eval_start:practice_start]
    practice_section = text[practice_start:notes_start]

    pattern = re.compile(r"\*\*(P(?:RACTICE-)?\d+)\s+—[^*]*\*\*\s*\n>\s*(.+?)(?=\n\n|\n\*\*|\n###|\Z)",
                          re.DOTALL)

    def extract(section):
        out = {}
        for m in pattern.finditer(section):
            pid = m.group(1)
            lines = [l.lstrip(">").strip() for l in m.group(2).split("\n")]
            out[pid] = " ".join(lines).strip()
        return out

    eval_prompts = extract(eval_section)
    practice_prompts = extract(practice_section)
    assert len(eval_prompts) == 10, f"expected 10 eval prompts, got {len(eval_prompts)}: {list(eval_prompts)}"
    assert len(practice_prompts) == 2, f"expected 2 practice prompts, got {len(practice_prompts)}"
    # Callers rely on P1..P10 order, so build the dict in that order
    # explicitly instead of trusting the regex scan order.
    ordered = {f"P{i}": eval_prompts[f"P{i}"] for i in range(1, 11)}
    return ordered, practice_prompts


def assert_config():
    """Load and hash/count-verify every configured artifact. Raises
    AssertionError with a specific message on any mismatch rather than
    proceeding on a drifted artifact. Returns a dict of everything a
    generation run needs, already verified."""
    from cpgtr.parse import PARSE_MAX_TOKENS, LLMParser
    from cpgtr.llm import make_complete
    from analysis.phase2_harness import build_rules_for_library, apply_template_corrections

    assert PARSE_MAX_TOKENS == EXPECTED["parse_max_tokens"], (
        f"PARSE_MAX_TOKENS={PARSE_MAX_TOKENS}, expected {EXPECTED['parse_max_tokens']}")

    cpgtr_cert, cpgtr_rec = build_rules_for_library("LIB-FAITHFUL-P-v2")
    assert cpgtr_rec["sha256"] == EXPECTED["LIB-FAITHFUL-P-v2_sha256"], (
        f"LIB-FAITHFUL-P-v2 sha256={cpgtr_rec['sha256']}, expected {EXPECTED['LIB-FAITHFUL-P-v2_sha256']}")
    assert len(cpgtr_cert) == EXPECTED["LIB-FAITHFUL-P-v2_n_rules"], (
        f"LIB-FAITHFUL-P-v2 n_rules={len(cpgtr_cert)}, expected {EXPECTED['LIB-FAITHFUL-P-v2_n_rules']}")

    # build_rules_for_library() raises LibraryHashMismatch if LIB-FULL's
    # recorded hash differs from a fresh hash of its own rule list, so a
    # successful call already verifies the library is unmodified.
    ungated_cert, ungated_rec = build_rules_for_library("LIB-FULL")
    assert len(ungated_cert) == EXPECTED["LIB-FULL_n_rules"], (
        f"LIB-FULL n_rules={len(ungated_cert)}, expected {EXPECTED['LIB-FULL_n_rules']}")

    corrections_sha = hashlib.sha256(CORRECTIONS_PATH.read_bytes()).hexdigest()
    assert corrections_sha == EXPECTED["corrections_sha256"], (
        f"realization_templates_corrections.json sha256={corrections_sha}, "
        f"expected {EXPECTED['corrections_sha256']}")
    apply_template_corrections(cpgtr_cert)
    apply_template_corrections(ungated_cert)

    codebook = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
    basis = json.dumps({"codes": codebook["codes"], "over_requirement": codebook["over_requirement"]},
                        sort_keys=True)
    recomputed = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    assert recomputed == codebook["sha256"], (
        f"codebook_v2.json self-consistency check failed: recomputed={recomputed}, "
        f"recorded={codebook['sha256']}")
    assert codebook["version"] == "v2"

    assert PROMPTS_PATH.exists(), f"{PROMPTS_PATH} not found"
    prompts_file_sha256 = hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest()

    complete = make_complete()
    # Production parser configuration: no negative example in the prompt.
    parser = LLMParser(complete, enforce_topology=True, negative_example=False)
    assert parser.negative_example is False

    return {
        "complete": complete, "parser": parser,
        "cpgtr_cert": cpgtr_cert, "cpgtr_rec": cpgtr_rec,
        "ungated_cert": ungated_cert, "ungated_rec": ungated_rec,
        "codebook": codebook, "codebook_path": str(CODEBOOK_PATH),
        "prompts_file_sha256": prompts_file_sha256,
        "corrections_sha256": corrections_sha,
        "config_summary": {
            "CP-GTR/CP-GTR-Detect library": f"LIB-FAITHFUL-P-v2 (n={len(cpgtr_cert)}, sha256={cpgtr_rec['sha256']})",
            "CP-GTR-Ungated library": f"LIB-FULL (n={len(ungated_cert)}, sha256={ungated_rec['sha256']})",
            "parser": "Condition A (negative_example=False)",
            "codebook": f"codebook_v2.json (sha256={codebook['sha256']})",
            "prompts": f"docs/phase2_prompts_v1.md (file sha256={prompts_file_sha256})",
            "templates": f"realization_templates_corrections.json (sha256={corrections_sha})",
            "hosts/parser": f"canonical LLMParser, PARSE_MAX_TOKENS={PARSE_MAX_TOKENS}",
        },
    }
