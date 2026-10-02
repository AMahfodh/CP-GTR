"""Baseline certification and coverage run on the first (non-canonical) extraction.

Supports: the deployment-realism section's account of coverage before a shared
topology convention existed (hosts matched by only a few rules, a single
edge-free rule accounting for all structural matches), the count of cached
candidates that violate the canonical attachment table, and the reported
comparison against the operator-anchored parser variant. It is the baseline
against which the canonical extraction of phase1a_run.py is compared.

Three configurations are run on the cached candidates, each under both
edge-free policies (keep and reject), six runs in total:
  P1-0   cached candidates and the existing hosts, unchanged.
  P1-B   as P1-0, with the kappa canonicalization applied to rules and hosts.
  P1-BP  as P1-B, with hosts re-parsed from their stored source text by the
         deterministic RuleParser(enable_d7=True), the variant that emits the
         extractor's operator-anchored arrangement. This changes the parser
         type (model-based to deterministic) as well as the feature, so it is
         not a controlled comparison against P1-0 and P1-B.
For each run it records the proposed and topology-rejected candidate counts
(the latter checked retroactively against schema/attach.yaml, since the
extraction predates the attachment table), edge-free, stratification and CPA
rejections, the certified count, per-source counts (Table 4 columns), the
four deployment-ablation configurations, and the coverage measures (host,
rule, structural, hub-parse rate). Because the check uses the current
attachment table, the retroactive count it reports can differ from a value
recorded under an earlier version of that table.

Run (offline, no API calls; long-running because of repeated admission and
repair):
    python analysis/phase1_run.py
Order: independent of the phase-2 scripts; it uses only the first extraction
cache. Results are written incrementally, after each ablation row, and a rerun
resumes from the rows already complete.

Inputs : cache/real_extraction_cache.jsonl (LLM-generated candidates),
         corpus/provisions.json, corpus/holdout_hosts_real.json,
         schema/attach.yaml, schema/equivalences.yaml.
Outputs: analysis/phase1_results.json, analysis/phase1_report.md.
"""
from __future__ import annotations
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from run_e2e import (PROVISIONS, EXTRACTION_CACHE, HOLDOUT_REAL,
                      deduplicate_provisions, stratified_sample,
                      ablation_run, ABLATION_CONFIGS, SOURCE_TIER)
from cpgtr.extract import _provision_hash, MAX_TOKENS_PER_ITEM, _load_cache, json_to_rule
from cpgtr.admit import admit
from cpgtr.eval_repair import run_repair
from cpgtr.parse import RuleParser
from cpgtr.graph import Graph
from cpgtr.topology import canonical_violations, spec_topology_violations
from cpgtr.canon import canonicalize_graph, canonicalize_rule

MODEL = "gpt-oss-120b"
OUT_JSON = Path(__file__).resolve().parent / "phase1_results.json"
OUT_MD = Path(__file__).resolve().parent / "phase1_report.md"

# Wall-clock budget per ablation_run()/admit() call. The same budget applies to
# every configuration, not only the slowest one, so the budget cannot bias
# which configurations complete. A configuration that hits it is reported as
# not completing within the 4-hour budget, with the per-candidate progress
# trace admit() produced up to that point, never silently truncated or
# presented as final.
BUDGET_SECONDS = 4 * 3600


def log(msg):
    print(f"[phase1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base_candidates():
    provisions = json.load(open(PROVISIONS, encoding="utf-8"))
    deduped, _ = deduplicate_provisions(provisions)
    sample = stratified_sample(deduped)
    cache = _load_cache(EXTRACTION_CACHE)
    candidates, missing = [], 0
    for p in sample:
        h = _provision_hash(MODEL, 0.0, MAX_TOKENS_PER_ITEM, p)
        entry = cache.get(h)
        if entry is None:
            missing += 1
            continue
        candidates.append(json_to_rule(entry["spec"], source=entry["source"]))
    assert missing == 0, f"{missing} sampled provisions missing from cache -- not cache-only"
    return candidates, cache, sample


def build_host_graph(h):
    g = Graph()
    for n in h["nodes"]:
        g.add_node(n["id"], n["type"])
    for e in h["edges"]:
        g.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return g


def force_phi_true(rules):
    """Copies of `rules` with phi replaced by "always true", for the
    structural-coverage metric: isolates topology matching from context
    gating."""
    out = []
    for r in rules:
        r2 = copy.copy(r)
        r2.phi = (lambda A, J: True)
        out.append(r2)
    return out


def compute_topology_rejections(specs_or_none, candidates):
    """Retroactive topology-rejection count: how many of `candidates` contain
    an edge not in schema/attach.yaml's ATTACH. It works on Rule objects
    rather than raw specs because kappa-canonicalized candidates no longer
    have a spec, so it applies uniformly before and after kappa."""
    n = 0
    for r in candidates:
        if canonical_violations(r.L) or canonical_violations(r.R):
            n += 1
    return n


def coverage_metrics(cert, hosts_graphs_with_ctx, deadline=None):
    """hosts_graphs_with_ctx: list of (host_id, Graph, (A,J)) tuples.
    Returns host coverage, rule coverage and structural coverage.

    `cert` is normally the critical-pair-checked certified set, so this should
    be fast, but the same wall-clock `deadline` is passed through as a safeguard,
    on the same footing as the ablation rows."""
    n_hosts = len(hosts_graphs_with_ctx)
    host_hit = 0
    rule_hit_names = set()
    struct_hit = 0
    forced_cert = force_phi_true(cert)
    for host_id, g, ctx in hosts_graphs_with_ctx:
        res = run_repair(cert, [{"id": host_id, "context": {"A": ctx[0], "J": ctx[1]},
                                  "nodes": [{"id": n, "type": t} for n, t in g.nodes.items()],
                                  "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                                            for e, (s, t, ty) in g.edges.items()]}],
                         T_max=10000, deadline=deadline)
        per = res["per_host"][0]
        if per["steps"] > 0:
            host_hit += 1
        # run_repair does not record which rule fired, so rule coverage uses
        # a coarser signal: which certified rules have any match on this host
        # (rule coverage = certified rules matching at least one host).
        from cpgtr.rules import one_step
        for r, m, _H in one_step(cert, g, ctx):
            rule_hit_names.add(r.name)

        res_struct = run_repair(forced_cert,
                                 [{"id": host_id, "context": {"A": 0, "J": "EU"},
                                   "nodes": [{"id": n, "type": t} for n, t in g.nodes.items()],
                                   "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                                             for e, (s, t, ty) in g.edges.items()]}],
                                 T_max=10000, deadline=deadline)
        if res_struct["per_host"][0]["steps"] > 0:
            struct_hit += 1

    return {
        "host_coverage": host_hit / n_hosts if n_hosts else None,
        "host_coverage_n": f"{host_hit}/{n_hosts}",
        "rule_coverage": len(rule_hit_names) / len(cert) if cert else None,
        "rule_coverage_n": f"{len(rule_hit_names)}/{len(cert)}",
        "structural_coverage": struct_hit / n_hosts if n_hosts else None,
        "structural_coverage_n": f"{struct_hit}/{n_hosts}",
    }


def per_source_table4(candidates, cert):
    proposed_by_source, certified_by_source = {}, {}
    for r in candidates:
        proposed_by_source[r.source] = proposed_by_source.get(r.source, 0) + 1
    for r in cert:
        certified_by_source[r.source] = certified_by_source.get(r.source, 0) + 1
    rows = []
    for fname, (label, tier) in SOURCE_TIER.items():
        rows.append({
            "source": label, "tier": tier,
            "proposed": proposed_by_source.get(fname, 0),
            "certified": certified_by_source.get(fname, 0),
        })
    return rows


def run_one_config(config_name, candidates, hosts, hub_parse_hits, results):
    log(f"=== {config_name} ===")
    result = {"n_candidates": len(candidates)}
    result["topology_rejections"] = compute_topology_rejections(None, candidates)
    result["edgefree_rejections_available"] = sum(1 for r in candidates if not r.L.edges)
    result["hub_parse_rate"] = f"{hub_parse_hits}/{len(hosts)}"

    ablation_rows = {}
    full_cert = None
    full_cert_budget_exceeded = False
    for i, (name, flags) in enumerate(ABLATION_CONFIGS):
        # Run edgefree-reject first so its result is written to disk before
        # the (typically slower) edgefree-keep budget is spent.
        for ef in ("reject", "keep"):
            key = f"{name} / edgefree-{ef}"
            log(f"  ablation: {key}")
            t_start = time.time()
            m = ablation_run(list(candidates), holdout=[
                {"id": h[0], "context": {"A": h[2][0], "J": h[2][1]},
                 "nodes": [{"id": n, "type": t} for n, t in h[1].nodes.items()],
                 "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                           for e, (s, t, ty) in h[1].edges.items()]}
                for h in hosts
            ], edgefree=ef, budget_seconds=BUDGET_SECONDS, **flags)
            if m["budget_exceeded"]:
                log(f"    ** did not complete within the {BUDGET_SECONDS/3600:.0f}h budget "
                    f"({time.time()-t_start:.0f}s elapsed, "
                    f"{sum(1 for _, s in m['progress_trace'] if s == 'budget-exceeded')} "
                    f"candidate(s) unprocessed) **")
            ablation_rows[key] = m
            # Save after every ablation row, not only per configuration, so a
            # single slow row does not hide the progress of everything after
            # it.
            partial_result = dict(result)
            partial_result["ablation"] = dict(ablation_rows)
            partial_results = dict(results)
            partial_results[config_name] = partial_result
            OUT_JSON.write_text(json.dumps(partial_results, indent=2, ensure_ascii=False, default=str),
                                 encoding="utf-8")
            if name == "Full pipeline" and ef == "keep":
                cert, blog = admit(list(candidates), edgefree=ef,
                                    enable_cpa=flags["use_cpa"], enable_strat=flags["use_strat"],
                                    deadline=time.time() + BUDGET_SECONDS)
                full_cert = cert
                full_cert_budget_exceeded = getattr(blog, "budget_exceeded", False)
                if full_cert_budget_exceeded:
                    log(f"    ** full_cert admit() also did not complete within budget "
                        f"-- coverage metrics below use a PARTIAL cert **")
    result["ablation"] = ablation_rows
    result["certified_count_full_keep"] = len(full_cert) if full_cert is not None else None
    result["certified_count_full_keep_is_partial"] = full_cert_budget_exceeded
    result["table4_per_source_full_keep"] = per_source_table4(candidates, full_cert or [])

    log("  coverage metrics (full pipeline, edgefree-keep cert)...")
    result["coverage"] = coverage_metrics(full_cert or [], hosts,
                                           deadline=time.time() + BUDGET_SECONDS)

    results[config_name] = result
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    log(f"  wrote partial results to {OUT_JSON}")
    return result


def _config_complete(results, name):
    """A configuration counts as done only with all 8 ablation rows (4
    ABLATION_CONFIGS x 2 edgefree settings) and the coverage metrics present.
    Checking only `name in results` would be wrong: with per-row saving it
    becomes true after the first row, and a resumed run would skip the rest
    of that configuration."""
    r = results.get(name)
    if r is None:
        return False
    return len(r.get("ablation", {})) >= 8 and "coverage" in r


def main():
    log("loading base candidates...")
    base_candidates, cache, sample = load_base_candidates()
    log(f"{len(base_candidates)} candidates loaded")

    existing_hosts_raw = json.loads(Path(HOLDOUT_REAL).read_text(encoding="utf-8"))

    results = {}
    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            done = [k for k in results if _config_complete(results, k)]
            partial = [k for k in results if not _config_complete(results, k)]
            log(f"resuming: {done} complete, {partial} partial (will redo)")
        except Exception:
            results = {}

    # ---- P1-0: cached candidates, no kappa, existing hosts ----
    if not _config_complete(results, "P1-0"):
        hosts_p10 = [(h["id"], build_host_graph(h), (h["context"]["A"], h["context"]["J"]))
                     for h in existing_hosts_raw]
        hub_hits = sum(1 for _, g, _ in hosts_p10 if not canonical_violations(g))
        run_one_config("P1-0", base_candidates, hosts_p10, hub_hits, results)

    # ---- P1-B: cached candidates, kappa on rules and hosts ----
    if not _config_complete(results, "P1-B"):
        log("canonicalizing candidates with kappa...")
        kappa_candidates = [canonicalize_rule(r) for r in base_candidates]
        hosts_p1b = []
        for h in existing_hosts_raw:
            g = canonicalize_graph(build_host_graph(h))
            hosts_p1b.append((h["id"], g, (h["context"]["A"], h["context"]["J"])))
        hub_hits = sum(1 for _, g, _ in hosts_p1b if not canonical_violations(g))
        run_one_config("P1-B", kappa_candidates, hosts_p1b, hub_hits, results)

    # ---- P1-BP: cached candidates, kappa on rules and hosts, extended parser ----
    # Hosts are re-parsed from source_text by RuleParser(enable_d7=True); see
    # the module docstring for why this is not a parser-type-controlled
    # comparison.
    if not _config_complete(results, "P1-BP"):
        log("canonicalizing candidates with kappa (same as P1-B)...")
        kappa_candidates = [canonicalize_rule(r) for r in base_candidates]
        d7_parser = RuleParser(enable_d7=True)
        hosts_p1bp = []
        for h in existing_hosts_raw:
            doc = d7_parser.parse(h["source_text"])
            g = canonicalize_graph(doc.graph)
            hosts_p1bp.append((h["id"], g, (h["context"]["A"], h["context"]["J"])))
        hub_hits = sum(1 for _, g, _ in hosts_p1bp if not canonical_violations(g))
        run_one_config("P1-BP", kappa_candidates, hosts_p1bp, hub_hits, results)

    write_report(results)
    log("DONE")


def write_report(results):
    lines = ["# Phase 1 results (P1-0, P1-B, P1-BP x edgefree-keep/reject)", ""]
    for config_name, r in results.items():
        lines.append(f"## {config_name}")
        lines.append(f"- candidates: {r['n_candidates']}, "
                      f"topology_rejections (retroactive): {r['topology_rejections']}, "
                      f"edge-free-L available: {r['edgefree_rejections_available']}")
        lines.append(f"- hub parse rate: {r['hub_parse_rate']}")
        partial_note = " (PARTIAL -- budget exceeded)" if r.get("certified_count_full_keep_is_partial") else ""
        lines.append(f"- certified (Full pipeline, edgefree-keep): {r['certified_count_full_keep']}{partial_note}")
        lines.append(f"- coverage: host={r['coverage']['host_coverage_n']}, "
                      f"rule={r['coverage']['rule_coverage_n']}, "
                      f"structural={r['coverage']['structural_coverage_n']}")
        lines.append("")
        lines.append("| Config | Term | Nonterm | Strat.rej | Repair | Cascade | Budget |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, m in r["ablation"].items():
            budget_col = "EXCEEDED (4h)" if m.get("budget_exceeded") else "ok"
            lines.append(f"| {key} | {m['term_rate']} | {m['nonterm']} | "
                          f"{m['strat_reject']} | {m['repair_success']:.2f} | {m['median_cascade']} | {budget_col} |")
        for key, m in r["ablation"].items():
            if m.get("budget_exceeded"):
                n_unproc = sum(1 for _, s in m["progress_trace"] if s == "budget-exceeded")
                n_total = len(m["progress_trace"])
                lines.append(f"  - **{key}** did not complete within the 4-hour budget: "
                              f"{n_total - n_unproc}/{n_total} candidates processed before the "
                              f"deadline; full per-candidate progress trace is in "
                              f"phase1_results.json under this config's "
                              f"`ablation[\"{key}\"][\"progress_trace\"]`.")
        lines.append("")
        lines.append("Table 4 (per source, Full pipeline / edgefree-keep):")
        for row in r["table4_per_source_full_keep"]:
            lines.append(f"  - {row['source']} ({row['tier']}): "
                          f"proposed={row['proposed']}, certified={row['certified']}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
