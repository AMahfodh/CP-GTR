"""annotate.py -- blinded annotation CLI.

Reads cache/table5_generation.json (real per-arm output text, produced by
run_e2e.py: generate_table5_grid('real')) and cache/table5_blind_map.json (a
hidden, seeded shuffle of method name -> anonymous "Output A".."Output E"
label per pair, generated at the same time). The annotator is shown ONLY the
prompt, context, and anonymized label -- never the real method name --
against docs/table4_pilot/codebook.md's violation codes, filtered to the
codes applicable at that item's (age, jurisdiction). "Without blinding the
comparison is worthless" -- this is not optional.

Usage:
    python annotate.py --annotator NAME              # full 75-item queue
    python annotate.py --annotator NAME --overlap    # 20% double-annotation subset only
    python annotate.py --kappa NAME1 NAME2           # Cohen's kappa on the overlap subset
    python annotate.py --export [--annotator NAME]   # un-blind -> cache/table5_annotations.json

Resumable: labels are saved to cache/table5_labels_<NAME>.json after every
single answer, and already-answered items are skipped on restart. The
--overlap pass writes to a SEPARATE cache/table5_labels_<NAME>_overlap.json
so it always asks fresh, even if the same annotator already covered those
items in a full pass -- the point is an independent second judgment for
inter-annotator reliability, not a resumed continuation of the first.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import sys

# Real LLM-generated output text can contain unicode punctuation (e.g. a
# non-breaking hyphen) outside Windows' default cp1252 console encoding --
# reconfigure before printing any of it rather than crashing mid-annotation.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cpgtr.codebook import CODES, applicable_codes

GENERATION  = os.path.join("cache", "table5_generation.json")
BLIND_MAP   = os.path.join("cache", "table5_blind_map.json")
ANNOTATIONS = os.path.join("cache", "table5_annotations.json")
OVERLAP_SEED = 11
OVERLAP_FRACTION = 0.2
LABELS = ["A", "B", "C", "D", "E"]


def _labels_path(name, overlap):
    suffix = "_overlap" if overlap else ""
    return os.path.join("cache", f"table5_labels_{name}{suffix}.json")


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    os.makedirs("cache", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _overlap_pair_ids(generation):
    """Deterministic ~20% subset of pair_ids, seeded independently of the
    generation/blinding seeds so it isn't guessable from those."""
    pair_ids = sorted(generation.keys())
    if not pair_ids:
        return []
    rng = random.Random(OVERLAP_SEED)
    shuffled = list(pair_ids)
    rng.shuffle(shuffled)
    n = max(1, math.ceil(len(pair_ids) * OVERLAP_FRACTION))
    return sorted(shuffled[:n])


def _queue(generation, overlap):
    pair_ids = _overlap_pair_ids(generation) if overlap else sorted(generation.keys())
    for pair_id in pair_ids:
        for label in LABELS:
            yield pair_id, label


def cmd_annotate(name, overlap):
    generation = _load_json(GENERATION, None)
    blind_map = _load_json(BLIND_MAP, None)
    if generation is None or blind_map is None:
        raise SystemExit(
            f"Missing {GENERATION} / {BLIND_MAP} -- run "
            f"`CPGTR_MODE=real python -c \"from run_e2e import "
            f"generate_table5_grid; generate_table5_grid('real')\"` first."
        )

    path = _labels_path(name, overlap)
    labels = _load_json(path, {})

    queue = list(_queue(generation, overlap))
    todo = [(p, l) for p, l in queue if p not in labels or l not in labels.get(p, {})]

    if not todo:
        print(f"[annotate] Nothing left to do for annotator={name!r} "
              f"(overlap={overlap}) -- {len(queue)}/{len(queue)} already answered "
              f"in {path}.")
        return

    print(f"[annotate] {len(todo)} of {len(queue)} items remaining for "
          f"annotator={name!r} (overlap={overlap}). Answers save after every "
          f"item -- safe to Ctrl-C and resume later.\n")

    for pair_id, label in todo:
        item = generation[pair_id]
        age, jurisdiction = item["context"]["A"], item["context"]["J"]
        text = item[blind_map[pair_id][label]]   # real method name looked up,
                                                   # NEVER printed
        codes = applicable_codes(age, jurisdiction)

        print("=" * 78)
        print(f"Item: {pair_id}  |  Output {label}  |  age={age}, "
              f"jurisdiction={jurisdiction}")
        print("-" * 78)
        print(text)
        print("-" * 78)
        if not codes:
            print("(No codebook violation applies at this age/jurisdiction -- "
                  "recording as compliant automatically.)")
            chosen = []
        else:
            print("Applicable violation codes:")
            for i, code in enumerate(codes, 1):
                print(f"  {i}) {code} ({CODES[code]['citation']}) -- "
                      f"{CODES[code]['description']}")
            raw = input("Enter violated code numbers (space/comma separated), "
                         "or Enter/0 for none: ").strip()
            chosen = _parse_choice(raw, codes)

        labels.setdefault(pair_id, {})[label] = chosen
        _save_json(path, labels)
        print(f"  -> recorded: {chosen or '(compliant)'}\n")

    print(f"[annotate] Done. {len(queue)} items saved to {path}.")


def _parse_choice(raw, codes):
    raw = raw.strip()
    if not raw or raw == "0":
        return []
    chosen = []
    for tok in raw.replace(",", " ").split():
        try:
            i = int(tok)
        except ValueError:
            print(f"  (ignoring unrecognized entry {tok!r})")
            continue
        if 1 <= i <= len(codes):
            chosen.append(codes[i - 1])
        else:
            print(f"  (ignoring out-of-range entry {tok!r})")
    return chosen


def _load_labels_for_kappa(name, prefer_overlap=True):
    if prefer_overlap:
        overlap_path = _labels_path(name, overlap=True)
        if os.path.exists(overlap_path):
            return _load_json(overlap_path, {}), overlap_path
    full_path = _labels_path(name, overlap=False)
    return _load_json(full_path, {}), full_path


def cmd_kappa(name1, name2):
    generation = _load_json(GENERATION, None)
    if generation is None:
        raise SystemExit(f"Missing {GENERATION}.")
    overlap_ids = set(_overlap_pair_ids(generation))

    if name1 == name2:
        # Self-consistency (test-retest), not inter-annotator: compare this
        # annotator's FIRST judgment (their full pass, restricted to the
        # overlap subset) against their INDEPENDENT second judgment (the
        # --overlap pass) -- comparing the overlap file to itself would be a
        # trivial, guaranteed-1.0 no-op, not a real check.
        labels1, path1 = _load_labels_for_kappa(name1, prefer_overlap=False)
        labels2, path2 = _load_labels_for_kappa(name2, prefer_overlap=True)
        if path1 == path2:
            raise SystemExit(
                f"Can't self-check {name1!r}: no separate full-pass file "
                f"found (only {path2} exists). Run the full "
                f"`--annotator {name1}` pass first, or pass a second "
                f"annotator's name for a real inter-annotator comparison."
            )
        print(f"[kappa] SELF-CONSISTENCY check for {name1!r}: first pass "
              f"({path1}, restricted to overlap items) vs independent "
              f"second pass ({path2})")
    else:
        labels1, path1 = _load_labels_for_kappa(name1)
        labels2, path2 = _load_labels_for_kappa(name2)
        print(f"[kappa] comparing {path1} vs {path2} on {len(overlap_ids)} "
              f"overlap items")

    rater1, rater2 = [], []
    missing = 0
    for pair_id in sorted(overlap_ids):
        item = generation[pair_id]
        age, jurisdiction = item["context"]["A"], item["context"]["J"]
        codes = applicable_codes(age, jurisdiction)
        for label in LABELS:
            l1 = labels1.get(pair_id, {}).get(label)
            l2 = labels2.get(pair_id, {}).get(label)
            if l1 is None or l2 is None:
                missing += 1
                continue
            for code in codes:
                rater1.append(code in l1)
                rater2.append(code in l2)

    if missing:
        print(f"[kappa] {missing} (item, label) pairs missing from one rater "
              f"-- skipped.")
    if not rater1:
        raise SystemExit("[kappa] no comparable (item, code) judgments found "
                          "-- both annotators need to complete the --overlap "
                          "pass first.")

    kappa = _cohens_kappa(rater1, rater2)
    n = len(rater1)
    agree = sum(a == b for a, b in zip(rater1, rater2)) / n
    print(f"[kappa] n={n} (item x applicable-code judgments), "
          f"observed agreement={agree:.1%}, Cohen's kappa={kappa:.3f}")
    return kappa


def _cohens_kappa(rater1, rater2):
    """Standard 2-rater Cohen's kappa over paired boolean judgments, pooled
    across every (item, applicable-code) instance in the overlap subset."""
    n = len(rater1)
    po = sum(a == b for a, b in zip(rater1, rater2)) / n
    p1_yes = sum(rater1) / n
    p2_yes = sum(rater2) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def cmd_export(name):
    generation = _load_json(GENERATION, None)
    blind_map = _load_json(BLIND_MAP, None)
    if generation is None or blind_map is None:
        raise SystemExit(f"Missing {GENERATION} / {BLIND_MAP}.")

    if name is None:
        candidates = [
            f[len("table5_labels_"):-len(".json")]
            for f in os.listdir("cache")
            if f.startswith("table5_labels_") and f.endswith(".json")
            and not f.endswith("_overlap.json")
        ] if os.path.isdir("cache") else []
        if len(candidates) != 1:
            raise SystemExit(
                f"--export needs exactly one full-pass labels file to pick "
                f"from automatically, found {candidates or 'none'} -- pass "
                f"--annotator NAME explicitly."
            )
        name = candidates[0]

    labels_path = _labels_path(name, overlap=False)
    labels = _load_json(labels_path, None)
    if labels is None:
        raise SystemExit(f"Missing {labels_path} -- run the full annotation "
                          f"pass for annotator={name!r} first.")

    expected = {p: set(LABELS) for p in generation}
    incomplete = [
        p for p in expected
        if set(labels.get(p, {}).keys()) != expected[p]
    ]
    if incomplete:
        raise SystemExit(
            f"{labels_path} is incomplete -- {len(incomplete)} item(s) still "
            f"missing at least one label, e.g. {incomplete[:3]}. Finish "
            f"`python annotate.py --annotator {name}` before exporting."
        )

    by_method = {}
    for pair_id, item in generation.items():
        age, jurisdiction = item["context"]["A"], item["context"]["J"]
        for label in LABELS:
            method_name = blind_map[pair_id][label]
            violations = labels[pair_id][label]
            by_method.setdefault(method_name, []).append({
                "prompt_id": item["prompt_id"],
                "age": age,
                "jurisdiction": jurisdiction,
                "violations": violations,
            })

    _save_json(ANNOTATIONS, by_method)
    print(f"[export] wrote {ANNOTATIONS}: " +
          ", ".join(f"{k}={len(v)}" for k, v in by_method.items()))


CSV_FIELDS = ["item_id", "pair_id", "output_label", "prompt_id", "age",
              "jurisdiction", "applicable_violation_codes", "output_text",
              "violations_found"]

_COMPLIANT_TOKENS = {"none", "compliant", "0", "n/a", "na"}


def _format_applicable(codes):
    return " | ".join(
        f"{c} ({CODES[c]['citation']}): {CODES[c]['description']}" for c in codes
    )


def cmd_csv_export(path, overlap):
    """Write a survey-style CSV: one row per (pair, blinded output), with an
    empty `violations_found` column for a reviewer to fill in a spreadsheet.
    Blinding is preserved exactly as in the interactive CLI -- output_label
    is "A".."E", never the real method name."""
    generation = _load_json(GENERATION, None)
    blind_map = _load_json(BLIND_MAP, None)
    if generation is None or blind_map is None:
        raise SystemExit(f"Missing {GENERATION} / {BLIND_MAP}.")

    rows = []
    for pair_id, label in _queue(generation, overlap):
        item = generation[pair_id]
        age, jurisdiction = item["context"]["A"], item["context"]["J"]
        codes = applicable_codes(age, jurisdiction)
        rows.append({
            "item_id": f"{pair_id}__{label}",
            "pair_id": pair_id,
            "output_label": label,
            "prompt_id": item["prompt_id"],
            "age": age,
            "jurisdiction": jurisdiction,
            "applicable_violation_codes": _format_applicable(codes),
            "output_text": item[blind_map[pair_id][label]],
            "violations_found": "",
        })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # utf-8-sig: real LLM output can contain non-ASCII punctuation; the BOM
    # makes Excel on Windows render it correctly instead of mangling it.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv-export] wrote {len(rows)} rows to {path}.\n"
          f"  Fill in 'violations_found' per row: comma/semicolon-separated "
          f"violation code names from 'applicable_violation_codes' (copy the "
          f"CODE_NAME part, not the citation/description), or type NONE for "
          f"a compliant output. Leave truly blank ONLY for rows not yet "
          f"reviewed -- blank is treated as 'not answered', not 'compliant'.")


def cmd_csv_import(path, name, overlap):
    """Read a filled-in survey CSV back and merge it into
    cache/table5_labels_<name>(_overlap).json -- the same storage format the
    interactive CLI uses, so --kappa/--export work unchanged regardless of
    which workflow produced the labels. Only rows with a non-blank
    'violations_found' are recorded; existing answers for other items in the
    same label file are preserved (safe to import an in-progress CSV more
    than once as more rows get filled in)."""
    generation = _load_json(GENERATION, None)
    if generation is None:
        raise SystemExit(f"Missing {GENERATION}.")

    out_path = _labels_path(name, overlap)
    labels = _load_json(out_path, {})

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    answered = skipped = 0
    errors = []
    for row in rows:
        pair_id, label = row["pair_id"], row["output_label"]
        if pair_id not in generation or label not in LABELS:
            errors.append(f"{row.get('item_id', '?')}: unrecognized pair_id/output_label")
            continue

        raw = (row.get("violations_found") or "").strip()
        if not raw:
            skipped += 1
            continue

        item = generation[pair_id]
        codes = applicable_codes(item["context"]["A"], item["context"]["J"])
        if raw.lower() in _COMPLIANT_TOKENS:
            chosen = []
        else:
            chosen = [tok.strip() for tok in raw.replace(";", ",").split(",") if tok.strip()]
            bad = [c for c in chosen if c not in codes]
            if bad:
                errors.append(
                    f"{row['item_id']}: {bad} not in this row's applicable "
                    f"codes {codes} (typo, or a code that doesn't apply at "
                    f"age={item['context']['A']}, jurisdiction="
                    f"{item['context']['J']})"
                )
                continue

        labels.setdefault(pair_id, {})[label] = chosen
        answered += 1

    if errors:
        print(f"[csv-import] {len(errors)} row(s) had problems and were NOT "
              f"imported:")
        for e in errors:
            print(f"  - {e}")

    _save_json(out_path, labels)
    total = len(list(_queue(generation, overlap)))
    done = sum(len(v) for v in labels.values())
    print(f"[csv-import] imported {answered} answered row(s), {skipped} "
          f"blank/not-yet-reviewed -- {out_path} now has {done}/{total} "
          f"items for annotator={name!r} (overlap={overlap}).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", help="annotator name (labels saved under this name)")
    ap.add_argument("--overlap", action="store_true",
                     help="restrict to the ~20%% double-annotation subset")
    ap.add_argument("--kappa", nargs=2, metavar=("NAME1", "NAME2"),
                     help="compute Cohen's kappa between two annotators' overlap passes")
    ap.add_argument("--export", action="store_true",
                     help="un-blind labels and write cache/table5_annotations.json")
    ap.add_argument("--csv-export", metavar="PATH",
                     help="write a survey-style CSV (one row per blinded output) to PATH")
    ap.add_argument("--csv-import", metavar="PATH",
                     help="read a filled-in survey CSV from PATH into --annotator's labels")
    args = ap.parse_args()

    if args.kappa:
        cmd_kappa(*args.kappa)
    elif args.export:
        cmd_export(args.annotator)
    elif args.csv_export:
        cmd_csv_export(args.csv_export, args.overlap)
    elif args.csv_import:
        if not args.annotator:
            ap.error("--csv-import requires --annotator NAME")
        cmd_csv_import(args.csv_import, args.annotator, args.overlap)
    elif args.annotator:
        cmd_annotate(args.annotator, args.overlap)
    else:
        ap.error("pass --annotator NAME, --kappa NAME1 NAME2, --export, "
                  "--csv-export PATH, or --csv-import PATH")


if __name__ == "__main__":
    main()
