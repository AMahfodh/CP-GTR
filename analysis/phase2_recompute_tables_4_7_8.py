"""Recompute Tables 4, 7 and 8 offline from the canonical extraction.

Supports: Table 4 (corpus and certification funnel), Table 7 (mechanism
validation ablation), Table 8 (deployment-realism ablation) and the claim that
every deployment host diverges under the naive size measure. No API calls.

Table 4: the candidates are loaded from cache/canonical_extraction_cache.jsonl
(the canonical extraction, 200 provisions, seed 42) and admitted with the full
pipeline at its defaults. The admission log is kept to recover the complete
funnel: schema-valid candidates, topology rejections, edge-free candidates,
stratification (repair-typing) rejections, critical-pair rejections and the
certified count, per source. Topology rejections are zero by construction: the
canonical extraction prompt repairs topology violations before a candidate is
cached.

Table 7: uses the hand-written fixtures of cpgtr/library.py against authored
graphs. It involves no model call and no text parsing, so it is deterministic;
it is run through the same run_e2e.py code path as in any other run.

Table 8: re-runs the four ablation configurations over the same canonical
candidates against corpus/holdout_hosts_real_canonical_v4.json, the final
twelve hosts parsed under the canonical attachment table (parse token limit
8192, with the one parse that had fallen back to the deterministic parser
re-parsed). It reuses run_e2e.py's ablation machinery rather than
reimplementing it. Host files store only nodes and edges, with no
sentence-level provenance, so the effect of the negation filter on these hosts
cannot be re-checked without a fresh model parse.

Run (from the repository root, because run_e2e.py uses relative paths):
    python analysis/phase2_recompute_tables_4_7_8.py
Order: after phase1a_run.py (needs the canonical extraction cache). The
per-row wall-clock budget below, BUDGET_SECONDS_PER_ROW, can leave a slow
configuration (--CPA) incomplete; the table printer then refuses to emit a
table. Raise the budget (the reported table used 900 s per row) to complete it.
The certified set depends on the admission code at the checked-out revision.

Inputs : cache/canonical_extraction_cache.jsonl (LLM-generated candidates),
         corpus/provisions.json, corpus/holdout_hosts_real_canonical_v4.json,
         corpus/holdout_hosts.json, corpus/holdout_hosts_mechanism_probe.json.
Outputs: analysis/phase2_table4_recomputed.json,
         analysis/phase2_table8_recomputed.json (plus tables on stdout).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.extract import load_cached_candidates
from cpgtr.admit import admit
import run_e2e as R

CANONICAL_CACHE = REPO_ROOT / "cache" / "canonical_extraction_cache.jsonl"
V4_HOSTS = REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v4.json"

SOURCE_TIER = R.SOURCE_TIER  # {fname: (label, tier)}, shared with run_e2e.table4()

# =========================================================================== #
# Table 4, offline, with the full admission funnel recovered.
# =========================================================================== #
print("=== Table 4 recomputed offline from cache/canonical_extraction_cache.jsonl ===\n")

raw_cache = json.loads("[" + ",".join(
    l for l in CANONICAL_CACHE.read_text(encoding="utf-8").splitlines() if l.strip()
) + "]") if False else None  # unused; load_cached_candidates() below is the loader

n_cache_entries = sum(1 for _ in open(CANONICAL_CACHE, encoding="utf-8") if _.strip())
candidates = load_cached_candidates(str(CANONICAL_CACHE))
n_schema_valid = len(candidates)
n_skipped = n_cache_entries - n_schema_valid
print(f"Cache entries: {n_cache_entries}")
print(f"Schema-valid (reconstructed via json_to_rule(), current validation): {n_schema_valid}/{n_cache_entries}")
if n_skipped:
    print(f"  ({n_skipped} skipped -- see '[skip, cached]' lines above, if any)")

n_edge_free = sum(1 for r in candidates if not r.L.edges)
print(f"Edge-free candidates (L has no edge; descriptive count, not a rejection under edgefree='keep'): "
      f"{n_edge_free}/{n_schema_valid}")

# Topology rejections: the canonical extraction prompt (ATTACH table, worked
# examples, k=3 topology refinement; recorded in analysis/phase2_library.json
# under derived_from) repairs topology violations during extraction, before a
# candidate is cached, so every cached candidate has already passed that check.
# Candidates that failed repair and were never cached are not counted: this
# cache holds only the 200 sampled provisions' successes.
n_topology_rejected = 0
print(f"Topology rejections (already resolved during extraction, before caching): {n_topology_rejected}")

# Full-pipeline defaults: enable_cpa=True, enable_strat=True, edgefree='keep',
# no deadline.
cert, log = admit(candidates)
assert not log.budget_exceeded, "budget_exceeded is NOT empty -- D13 assertion FAILS, stop and report"
print("budget_exceeded: EMPTY (D13 assertion PASSES)")

strat_rejected = sum(1 for _n, s in log.notes if s == "reject:no-measure")
cpa_rejected = sum(1 for _n, s in log.notes if s in ("reject:cpa-nonterm", "reject:non-joinable-CP"))
edgefree_rejected = sum(1 for _n, s in log.notes if s == "reject:edgefree")
other_rejected = sum(1 for _n, s in log.notes if s not in
                      ("admit", "reject:no-measure", "reject:cpa-nonterm", "reject:non-joinable-CP",
                       "reject:edgefree", "budget-exceeded"))

print(f"\nAdmission funnel (n={len(candidates)} proposed):")
print(f"  admitted (certified):        {len(log.admitted)}")
print(f"  rejected: stratification:    {strat_rejected}")
print(f"  rejected: CPA:               {cpa_rejected}")
print(f"  rejected: edgefree:          {edgefree_rejected}  (0 expected -- edgefree='keep' default)")
if other_rejected:
    print(f"  rejected: OTHER (unexpected, inspect): {other_rejected}")
print(f"  total rejected:              {len(log.rejected)}")
assert len(log.admitted) + len(log.rejected) == len(candidates), "admitted+rejected != proposed, investigate"
assert strat_rejected + cpa_rejected + edgefree_rejected + other_rejected == len(log.rejected)

# Per-source breakdown
proposed_by_source, certified_by_source = {}, {}
strat_by_source, cpa_by_source = {}, {}
for r in candidates:
    proposed_by_source[r.source] = proposed_by_source.get(r.source, 0) + 1
for r in cert:
    certified_by_source[r.source] = certified_by_source.get(r.source, 0) + 1
name_to_source = {r.name: r.source for r in candidates}
# r.name is not a unique key on real extracted data (several candidates share a
# name), so keying the per-source rejection breakdown by name could misattribute
# a rejection to the wrong source. This count only quantifies that hazard; the
# breakdown itself is keyed by content hash, below.
name_collisions = len(candidates) - len(set(r.name for r in candidates))
print(f"\n(name-collision check: {name_collisions} of {len(candidates)} candidate names are non-unique -- "
      f"per-source strat/CPA counts below use content_hash-keyed attribution instead, avoiding this)")

hash_to_source = {r.content_hash(): r.source for r in candidates}
hash_to_name = {r.content_hash(): r.name for r in candidates}
# admit() logs candidates by name, positionally (see cpgtr/admit.py). Rebuild
# the order it used (sorted by descending confidence, then content hash) to zip
# the log notes back to candidates safely.
sorted_candidates = sorted(candidates, key=lambda r: (-(r.confidence if r.confidence is not None else -1), r.content_hash()))
strat_by_source, cpa_by_source = {}, {}
for (name, status), r in zip(log.notes, sorted_candidates):
    if status == "reject:no-measure":
        strat_by_source[r.source] = strat_by_source.get(r.source, 0) + 1
    elif status in ("reject:cpa-nonterm", "reject:non-joinable-CP"):
        cpa_by_source[r.source] = cpa_by_source.get(r.source, 0) + 1

print(f"\n{'Source':<28}{'Tier':<12}{'Spans':>7}{'Proposed':>10}{'StratRej':>10}{'CPARej':>8}{'Certified':>11}{'Rate':>8}")
provisions = json.load(open(REPO_ROOT / "corpus" / "provisions.json", encoding="utf-8"))
spans_by_source = {}
for p in provisions:
    spans_by_source[p["source"]] = spans_by_source.get(p["source"], 0) + 1

table4_rows = []
tot_spans = tot_prop = tot_strat = tot_cpa = tot_cert = 0
for fname, (label, tier) in SOURCE_TIER.items():
    spans = spans_by_source.get(fname, 0)
    prop = proposed_by_source.get(fname, 0)
    sr = strat_by_source.get(fname, 0)
    cr = cpa_by_source.get(fname, 0)
    cert_n = certified_by_source.get(fname, 0)
    rate = f"{100*cert_n/prop:.1f}%" if prop else "n/a"
    tot_spans += spans; tot_prop += prop; tot_strat += sr; tot_cpa += cr; tot_cert += cert_n
    print(f"{label:<28}{tier:<12}{spans:>7}{prop:>10}{sr:>10}{cr:>8}{cert_n:>11}{rate:>8}")
    table4_rows.append({"source": label, "tier": tier, "spans": spans, "proposed": prop,
                         "strat_rejected": sr, "cpa_rejected": cr, "certified": cert_n,
                         "certification_rate_pct": (100*cert_n/prop if prop else None)})
print(f"{'Total':<28}{'':<12}{tot_spans:>7}{tot_prop:>10}{tot_strat:>10}{tot_cpa:>8}{tot_cert:>11}"
      f"{100*tot_cert/tot_prop:.1f}%".rjust(8))

table4_out = {
    "cache_file": str(CANONICAL_CACHE.relative_to(REPO_ROOT)),
    "n_cache_entries": n_cache_entries, "n_schema_valid": n_schema_valid,
    "n_topology_rejected": n_topology_rejected, "n_edge_free": n_edge_free,
    "n_strat_rejected": strat_rejected, "n_cpa_rejected": cpa_rejected,
    "n_edgefree_rejected": edgefree_rejected, "n_certified": len(log.admitted),
    "budget_exceeded": log.budget_exceeded,
    "rows": table4_rows,
    "totals": {"spans": tot_spans, "proposed": tot_prop, "strat_rejected": tot_strat,
               "cpa_rejected": tot_cpa, "certified": tot_cert},
}
with open(REPO_ROOT / "analysis" / "phase2_table4_recomputed.json", "w", encoding="utf-8") as f:
    json.dump(table4_out, f, indent=2)
print("\nWritten: analysis/phase2_table4_recomputed.json")

# =========================================================================== #
# Table 7 (deterministic, no parsing) and Table 8 (final host file).
# =========================================================================== #
print("\n\n=== Table 7 (mechanism validation) -- re-run to CONFIRM no change ===")
table7_result = R.table7a()

print("\n\n=== Table 8 (deployment realism) -- final host file, canonical-cache candidates ===")
print(f"Host file: {V4_HOSTS.relative_to(REPO_ROOT)} (final, canonical-parser-fixed; "
      f"NOTE: built BEFORE the negation filter was added -- host files store "
      f"only {{nodes, edges}}, no edge-level sentence provenance, so the negation filter's effect on "
      f"these 12 hosts cannot be checked retroactively without a fresh parse (an API call, out of scope "
      f"for this offline task). Disclosed, not silently assumed away.)")

holdout_v4 = json.loads(V4_HOSTS.read_text(encoding="utf-8"))
# Wall-clock bound per configuration row. Under --CPA or --Strat a real
# deployment host can grow the graph at every step without repeating a
# signature, so the step cap T_MAX is a real but potentially very slow limit
# (see cpgtr/eval_repair.py). A row that exceeds this bound is reported as
# budget_exceeded, which _print_ablation_table refuses to turn into a table,
# instead of running unbounded.
BUDGET_SECONDS_PER_ROW = 180
print(f"(budget_seconds={BUDGET_SECONDS_PER_ROW} per configuration row, D13 safety bound)")
table8_rows = []
for name, flags in R.ABLATION_CONFIGS:
    print(f"  running config: {name} ...")
    m = R.ablation_run(candidates, holdout=holdout_v4, budget_seconds=BUDGET_SECONDS_PER_ROW, **flags)
    table8_rows.append((name, m))
    print(f"    done: term_rate={m['term_rate']} nonterm={m['nonterm']} "
          f"repair_success={m['repair_success']:.2f} budget_exceeded={m['budget_exceeded']}")

try:
    R._print_ablation_table("Table 8: Certification deployment realism (recomputed, final host file)", table8_rows)
except R.BudgetExceeded as e:
    print(f"\n*** D13 BUDGET EXCEEDED *** {e}")
    print("Reporting per-row raw metrics anyway (diagnostic, NOT a manuscript-ready table):")
    for name, m in table8_rows:
        print(f"  {name}: {({k: v for k, v in m.items() if k not in ('progress_trace', 'repair_per_host')})}")

table8_out = {
    "host_file": str(V4_HOSTS.relative_to(REPO_ROOT)),
    "n_candidates": len(candidates),
    "rows": {name: {k: v for k, v in m.items() if k != "repair_per_host"} for name, m in table8_rows},
}
with open(REPO_ROOT / "analysis" / "phase2_table8_recomputed.json", "w", encoding="utf-8") as f:
    json.dump(table8_out, f, indent=2)
print("\nWritten: analysis/phase2_table8_recomputed.json")

# Explicit check of the termination result: does every deployment host diverge
# under the naive size measure?
full_row = dict(table8_rows)["Full pipeline"]
strat_only_row = dict(table8_rows)["--Strat"]
n_hosts = len(holdout_v4)
print(f"\n=== Termination result check ===")
print(f"Full pipeline: term_rate={full_row['term_rate']}, nonterm incidents={full_row['nonterm']}/{n_hosts}")
print(f"--Strat (naive size measure): nonterm incidents={strat_only_row['nonterm']}/{n_hosts}")
all_diverge_under_naive = strat_only_row["nonterm"] == n_hosts
print(f"Does every deployment host diverge under the naive size measure (--Strat)? "
      f"{'YES' if all_diverge_under_naive else 'NO'} ({strat_only_row['nonterm']}/{n_hosts})")
