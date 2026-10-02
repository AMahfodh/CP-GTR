"""Task 7 (F2 fix, D3): retroactively re-score the EXISTING 15 Table 5 pilot
pairs under D3's rule (score the unrepaired draft for flagged documents,
report the flag rate separately) using the ALREADY-COLLECTED annotations --
no new text generation, no API calls, no re-annotation.

Why this is possible without new annotation: CP-GTR-Detect and CP-GTR both
repair the SAME "Unconstrained LLM" draft for a given (prompt, context) pair
(see run_e2e.py: generate_table5_grid()'s docstring) -- so for a FLAGGED
CP-GTR pair, D3's "unrepaired draft" IS, byte-for-byte, the text that was
independently gold-annotated as that pair's "Unconstrained LLM" arm entry.
Substituting that already-collected annotation for the flagged pair's score
is exactly what D3 asks for, with zero new labels needed.

Reads: cache/table5_generation.json (has _cpgtr_meta.round_trip_compliant per
pair), cache/table5_annotations.json (existing gold labels).
Writes: analysis/f2_rescore.json, analysis/f2_report.md.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATION = REPO_ROOT / "cache" / "table5_generation.json"
ANNOTATIONS = REPO_ROOT / "cache" / "table5_annotations.json"
OUT_JSON = Path(__file__).resolve().parent / "f2_rescore.json"
OUT_MD = Path(__file__).resolve().parent / "f2_report.md"

CTX = [("Child (A=9)", 9), ("Teen (A=15)", 15), ("Adult (A=21)", 21)]


def _key(entry):
    return (entry["prompt_id"], entry["age"], entry["jurisdiction"])


def compliance_rate(labeled_outputs, age, n_boot=2000, seed=0, ci=0.95):
    """Identical to cpgtr.eval_compliance.compliance_rate -- reimplemented
    locally only so this script has zero import-time dependency on anything
    that could construct a live LLM client; it is otherwise the exact same
    function/algorithm."""
    import random
    items = [e for e in labeled_outputs if e["age"] == age]
    if not items:
        raise ValueError(f"no gold-labeled outputs for age={age}")

    def _rate(sample):
        compliant = sum(1 for e in sample if not e["violations"])
        return 100.0 * compliant / len(sample)

    point = _rate(items)
    rng = random.Random(seed)
    n = len(items)
    boots = sorted(_rate([items[rng.randrange(n)] for _ in range(n)])
                    for _ in range(n_boot))
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot) - 1
    return point, (boots[lo_idx], boots[hi_idx])


def main():
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))

    cpgtr_before = {_key(e): e for e in annotations["CP-GTR (Ours)"]}
    unconstrained = {_key(e): e for e in annotations["Unconstrained LLM"]}

    flagged_pairs = []
    cpgtr_after = []
    for pair_id, item in generation.items():
        age, jurisdiction = item["context"]["A"], item["context"]["J"]
        key = (item["prompt_id"], age, jurisdiction)
        flagged = not item["_cpgtr_meta"]["round_trip_compliant"]
        if flagged:
            flagged_pairs.append(pair_id)
            # D3: score the unrepaired draft -- reuse the Unconstrained LLM
            # arm's EXISTING annotation for this exact (prompt, context),
            # since that IS the unrepaired draft, byte for byte.
            src = unconstrained[key]
        else:
            src = cpgtr_before[key]
        cpgtr_after.append({
            "prompt_id": src["prompt_id"], "age": src["age"],
            "jurisdiction": src["jurisdiction"], "violations": src["violations"],
        })

    before_rates = {label: compliance_rate(annotations["CP-GTR (Ours)"], age)
                     for label, age in CTX}
    after_rates = {label: compliance_rate(cpgtr_after, age) for label, age in CTX}

    out = {
        "n_pairs": len(generation),
        "n_flagged": len(flagged_pairs),
        "flag_rate": len(flagged_pairs) / len(generation),
        "flagged_pair_ids": sorted(flagged_pairs),
        "cpgtr_before_d3_fix": {label: {"rate": r, "ci": list(ci)} for label, (r, ci) in before_rates.items()},
        "cpgtr_after_d3_fix": {label: {"rate": r, "ci": list(ci)} for label, (r, ci) in after_rates.items()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# Task 7 (F2 fix, D3): before/after CCR for CP-GTR (Ours)", ""]
    lines.append(f"Flag rate: {len(flagged_pairs)}/{len(generation)} = {out['flag_rate']:.1%}")
    lines.append(f"Flagged pairs: {', '.join(sorted(flagged_pairs))}")
    lines.append("")
    lines.append("| Age band | Before (scored repaired output regardless of flag) | After (D3: unrepaired draft when flagged) |")
    lines.append("|---|---|---|")
    for label, _ in CTX:
        b = before_rates[label]
        a = after_rates[label]
        lines.append(f"| {label} | {b[0]:.1f} [{b[1][0]:.1f},{b[1][1]:.1f}] | {a[0]:.1f} [{a[1][0]:.1f},{a[1][1]:.1f}] |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
