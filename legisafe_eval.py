"""legisafe_eval.py -- regenerates every result table from results.json.



Usage:
    python run_e2e.py                # measure, write results.json
    python legisafe_eval.py          # render tables from results.json
    python legisafe_eval.py --path other_results.json
"""
from __future__ import annotations
import argparse
import json

from run_e2e import (CTX, ABLATION_COLS, ABLATION_LABELS, _print_ablation_table)


def render_table4(data):
    print("\n=== Table 4: LegiSafe-Bench corpus sources and extraction yield ===")
    hdr = f"{'Source':<28}{'Tier':<12}{'Spans':>8}{'Proposed':>10}{'Certified':>11}"
    print(hdr); print("-" * len(hdr))
    for r in data["rows"]:
        prop_s = str(r["proposed"]) if r["proposed"] is not None else "n/a"
        cert_s = str(r["certified"]) if r["certified"] is not None else "n/a"
        print(f"{r['source']:<28}{r['tier']:<12}{r['spans']:>8}{prop_s:>10}{cert_s:>11}")
    print("-" * len(hdr))
    tp = str(data["total_proposed"]) if data["total_proposed"] is not None else "n/a"
    tc = str(data["total_certified"]) if data["total_certified"] is not None else "n/a"
    print(f"{'Total':<28}{'':<12}{data['total_spans']:>8}{tp:>10}{tc:>11}")


def render_table5(data):
    print("\n=== Table 5: Contextual Compliance Rate (%) [95% CI] ===")
    hdr = f"{'Method':<28}" + "".join(f"{c:>22}" for c, _ in CTX)
    print(hdr); print("-" * len(hdr))
    for name, res in data.items():
        if "not_implemented" in res:
            print(f"{name:<28}{'-- not implemented --':>66}")
            print(f"    ({res['not_implemented']})")
            continue
        cells = "".join(
            f"{res[c]['rate']:>10.1f} [{res[c]['ci'][0]},{res[c]['ci'][1]}]".rjust(22)
            for c, _ in CTX)
        print(f"{name:<28}{cells}")


def render_table6(data):
    p, r, f = data["overall"]["all"]
    print("\n=== Table 6: SLG triplet extraction ===")
    print(f"{'Component':<24}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 48)
    print(f"{'SLG triplet extraction':<24}{p:8.2f}{r:8.2f}{f:8.2f}")
    print("No audited false-negative rate is reported: that review has not "
          "been conducted (CP-GTR_V2.tex sec:metrics item 4).")


def render_ablation(title, rows, cols=ABLATION_COLS, labels=ABLATION_LABELS,
                     unsound_log=None):
    if rows is None:
        print(f"\n=== {title} ===\n  -- not run --")
        return
    _print_ablation_table(title, [(name, m) for name, m in rows], cols=cols, labels=labels)
    if unsound_log is not None:
        print("\nPer-fixture unsound-admission log:")
        if not unsound_log:
            print("  (none -- no seeded fixture was ever admitted under a relaxed "
                  "config that the full pipeline rejects)")
        for config_name, fixture, guard, status in unsound_log:
            print(f"  [{config_name:<14}] {fixture:<12} admitted -- full pipeline "
                  f"catches it via {guard} ({status})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default="results.json",
                     help="results.json to render (default: results.json)")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    print(f"legisafe_eval.py: rendering {args.path}")
    print(f"  generated_at={meta['generated_at']}  mode={meta['mode']}  "
          f"provider={meta['provider']}  model={meta['model']}  "
          f"temperature={meta['temperature']}  extract_seed={meta['extract_seed']}  "
          f"extract_sample_n={meta['extract_sample_n']}")

    tables = data["tables"]
    if tables.get("table4"):
        render_table4(tables["table4"])
    if tables.get("table5"):
        render_table5(tables["table5"])
    if tables.get("table6"):
        render_table6(tables["table6"])
    if "table7a" in tables:
        t7 = tables["table7a"]
        render_ablation(
            "Table 7: Certification mechanism validation "
            "(seeded adversarial fixtures + the manuscript's worked examples)",
            t7["rows"], cols=ABLATION_COLS + ["unsound"],
            labels=ABLATION_LABELS + ["Unsound"], unsound_log=t7["unsound_log"])
    if "table8b" in tables:
        render_ablation("Table 8: Certification deployment realism", tables["table8b"])


if __name__ == "__main__":
    main()
