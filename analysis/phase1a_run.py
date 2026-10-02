"""Canonical (schema-enforced) rule extraction, certification and coverage run.

Supports: Table 4 (corpus and certification funnel for the canonical extraction
of 200 stratified provisions, seed 42), the deployment-realism ablation and the
host-coverage measurements, and the extraction-section claim that no candidate
violates the attachment table after schema-repair retries. It also defines
coverage_metrics_v3(), imported by phase2_taskC_host_remeasure.py,
phase2_taskF_host_remeasure.py, phase2_task3c_coverage_remeasure.py and
phase2_taskY_coverage.py.

Stages:
  1. Pilot: extract the first 20 sampled spans with the canonical prompt and a
     k=3 schema-repair loop, recording per-item first-try and final outcomes.
     If fewer than 50% of the pilot spans end up canonical, the script writes
     the pilot report and stops.
  2. Full run: extract all 200 spans (P1-A), then repeat with the kappa
     canonicalization applied to candidates and hosts (P1-AB). For each
     configuration it runs the four ablation configurations under both
     edge-free policies (keep and reject), records per-source certified counts
     (Table 4), and computes three coverage measures: host coverage in the
     real context, structural coverage with guards forced true, and structural
     coverage restricted to rules whose left pattern has at least one edge.
Both configurations use the default deterministic parser, not the extended
operator-anchored one.

Run (needs an LLM provider key; see cpgtr/llm.py):
    python analysis/phase1a_run.py
Order: the first extraction step. It must precede every script that loads
cache/canonical_extraction_cache.jsonl (phase2_recompute_tables_4_7_8.py and
the library builders). The run resumes: extracted items are cached and
configurations already complete in phase1a_results.json are skipped.

Inputs : corpus/provisions.json, corpus/holdout_hosts_real.json,
         schema/attach.yaml, schema/equivalences.yaml.
Outputs: cache/canonical_extraction_cache.jsonl (the earlier
         real_extraction_cache.jsonl is never written),
         analysis/phase1a_pilot.json, analysis/phase1a_results.json,
         analysis/phase1a_report.md.
"""
from __future__ import annotations
import copy
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from run_e2e import (PROVISIONS, deduplicate_provisions, stratified_sample,
                      get_real_complete, ablation_run, ABLATION_CONFIGS, SOURCE_TIER,
                      HOLDOUT_REAL)
from cpgtr.extract import (extract, json_to_rule, _single_prompt, _REPAIR_PROMPT,
                            _validate, _SCHEMA_NOTE, _cap, _provision_hash,
                            MAX_TOKENS_PER_ITEM, SINGLE_ITEM_MAX_TOKENS, _load_cache)
from cpgtr.admit import admit
from cpgtr.eval_repair import run_repair
from cpgtr.canon import canonicalize_rule, canonicalize_graph
from cpgtr.topology import canonical_violations, ATTACH
from cpgtr.graph import Graph
from cpgtr.rules import one_step

MODEL = "gpt-oss-120b"
PILOT_N = 20
K_REPAIR = 3
BUDGET_SECONDS = 4 * 3600

CANONICAL_CACHE = str(REPO_ROOT / "cache" / "canonical_extraction_cache.jsonl")
OUT_JSON = Path(__file__).resolve().parent / "phase1a_results.json"
OUT_MD = Path(__file__).resolve().parent / "phase1a_report.md"
PILOT_OUT_JSON = Path(__file__).resolve().parent / "phase1a_pilot.json"


def log(msg):
    print(f"[phase1a {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_sample():
    provisions = json.load(open(PROVISIONS, encoding="utf-8"))
    deduped, _ = deduplicate_provisions(provisions)
    # Same stratified sample (n=200, seed 42) as every other script.
    return stratified_sample(deduped)


# --------------------------------------------------------------------------- #
# Instrumented pilot (sequential, per-item first-try and final tracking)       #
# --------------------------------------------------------------------------- #
def pilot_extract_one(complete, p, k=K_REPAIR):
    """Sequential, instrumented canonical extraction for one provision.

    Exposes the first-try versus after-refinement outcome, which extract()'s
    concurrent path does not track per item. It reuses the prompt builder,
    repair prompt and validator that extract() itself uses (imported, not
    reimplemented), so the pilot measures the production path.

    Each step is wrapped so that a raw-response or JSON-parse failure is
    reported as its own outcome, with the number of attempts used so far,
    instead of propagating to a generic handler that would hide which step
    failed and how far the retry loop got.
    """
    def _complete_and_parse(prompt_text, **kw):
        kw.setdefault("max_tokens", SINGLE_ITEM_MAX_TOKENS)
        raw = complete(prompt_text, **kw)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m is None:
            raise ValueError(f"no JSON object found in response: {raw[:300]!r}")
        return json.loads(m.group(0)), raw

    try:
        d, raw0 = _complete_and_parse(_single_prompt(p))
    except Exception as e:
        return {"id": p.get("id"), "first_try_ok": False, "final_ok": False,
                "attempts_used": 0, "final_error": f"initial call/parse failed: {e}",
                "is_topology_final": False, "spec": None, "failed_at": "initial"}

    err, is_topo = _validate(d, p.get("source"), enforce_topology=True)
    first_try_ok = err is None
    attempts = 0
    last_err = err
    while err is not None and attempts < k:
        attempts += 1
        try:
            d, raw_r = _complete_and_parse(
                _REPAIR_PROMPT % (err, json.dumps(d), _SCHEMA_NOTE,
                                   p["jurisdiction"], p["tau"], _cap(p["text"])))
        except Exception as e:
            return {"id": p.get("id"), "first_try_ok": first_try_ok, "final_ok": False,
                    "attempts_used": attempts, "final_error": f"repair call/parse failed: {e}",
                    "is_topology_final": False, "spec": None, "failed_at": f"repair_{attempts}"}
        err, is_topo = _validate(d, p.get("source"), enforce_topology=True)
        last_err = err
    final_ok = err is None
    return {
        "id": p.get("id"), "first_try_ok": first_try_ok, "final_ok": final_ok,
        "attempts_used": attempts, "final_error": last_err,
        "is_topology_final": (is_topo if not final_ok else False),
        "spec": (d if final_ok else None),
        "failed_at": (None if final_ok else "exhausted_k"),
    }


def run_task3_pilot(complete, sample):
    pilot = sample[:PILOT_N]
    log(f"Pilot: extracting {len(pilot)} spans (k={K_REPAIR}, canonical prompt)...")
    Path(CANONICAL_CACHE).parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip any pilot item whose hash is already in the canonical cache
    # (e.g. from an interrupted run) so API budget is not spent twice.
    already = _load_cache(CANONICAL_CACHE)

    results = []
    for i, p in enumerate(pilot):
        h = _provision_hash(MODEL, 0.0, MAX_TOKENS_PER_ITEM, p)
        if h in already:
            r = {"id": p.get("id"), "first_try_ok": None, "final_ok": True,
                 "attempts_used": None, "final_error": None,
                 "is_topology_final": False, "spec": already[h]["spec"],
                 "failed_at": None, "resumed_from_cache": True}
            results.append(r)
            log(f"  [{i+1}/{len(pilot)}] {r['id']}: resumed from cache (already extracted)")
            continue
        try:
            r = pilot_extract_one(complete, p)
        except Exception as e:
            # Genuinely unexpected: a bug in this script, not an extraction or
            # repair failure (pilot_extract_one handles those and always
            # returns a dict). Recorded distinctly so it is never confused
            # with a real rejection after k repair attempts.
            r = {"id": p.get("id"), "first_try_ok": False, "final_ok": False,
                 "attempts_used": None, "final_error": f"UNEXPECTED SCRIPT ERROR: {e}",
                 "is_topology_final": False, "spec": None, "failed_at": "script_bug"}
        results.append(r)
        err_note = f" err={r['final_error'][:80]!r}" if not r["final_ok"] else ""
        log(f"  [{i+1}/{len(pilot)}] {r['id']}: first_try_ok={r['first_try_ok']} "
            f"final_ok={r['final_ok']} attempts={r['attempts_used']}{err_note}")

        # Persist immediately rather than at the end of the loop, so an
        # interrupted run loses no completed work.
        if r["final_ok"] and not r.get("resumed_from_cache"):
            with open(CANONICAL_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"hash": h, "spec": r["spec"],
                                     "source": p.get("source")}, ensure_ascii=False) + "\n")

    n = len(results)
    n_first = sum(1 for r in results if r["first_try_ok"])
    n_final = sum(1 for r in results if r["final_ok"])
    n_topo_reject = sum(1 for r in results if not r["final_ok"] and r["is_topology_final"])
    n_edge_free = sum(1 for r in results
                       if r["final_ok"] and not r["spec"].get("L", {}).get("edges"))
    n_initial_parse_failure = sum(1 for r in results if r.get("failed_at") == "initial")

    examples = [r for r in results if r["final_ok"]][:3]
    report = {
        "n": n,
        "first_try_canonical_rate": n_first / n,
        "final_canonical_rate": n_final / n,
        "topology_rejections_final": n_topo_reject,
        "other_rejections_final": (n - n_final) - n_topo_reject,
        "initial_json_parse_failures": n_initial_parse_failure,
        "edge_free_rate_of_final": (n_edge_free / n_final) if n_final else None,
        "stop_rule_triggered": (n_final / n) < 0.5,
        "examples_verbatim": examples,
        "all_results": results,
    }
    return report


# --------------------------------------------------------------------------- #
# Full P1-A / P1-AB run                                                       #
# --------------------------------------------------------------------------- #
def build_host_graph(h):
    g = Graph()
    for n in h["nodes"]:
        g.add_node(n["id"], n["type"])
    for e in h["edges"]:
        g.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return g


def force_phi_true(rules):
    out = []
    for r in rules:
        r2 = copy.copy(r)
        r2.phi = (lambda A, J: True)
        out.append(r2)
    return out


def compute_topology_rejections(candidates):
    return sum(1 for r in candidates
               if canonical_violations(r.L) or canonical_violations(r.R))


def coverage_metrics_v3(cert, hosts_graphs_with_ctx, deadline=None):
    """Coverage of the hosts by a certified library, reported three ways:
      (i)   host coverage       -- matches in the host's real context
      (ii)  structural coverage -- guards forced true, all certified rules
      (iii) structural coverage -- guards forced true, only rules whose left
            pattern has at least one edge
    Measure (iii) is needed because an edge-free rule matches any host with a
    node of the right type and carries no structure, so (ii) alone can be
    inflated by a single such rule. Also returns per-rule match counts
    (structural and real-context) across all hosts, for every certified rule.
    """
    n_hosts = len(hosts_graphs_with_ctx)
    host_hit = 0
    struct_hit = 0
    struct_hit_nonef = 0
    rule_hit_names = set()
    forced_all = force_phi_true(cert)
    non_ef = [r for r in cert if r.L.edges]
    forced_nonef = force_phi_true(non_ef)

    per_rule_struct = {r.name: 0 for r in cert}
    per_rule_ctx = {r.name: 0 for r in cert}

    for host_id, g, ctx in hosts_graphs_with_ctx:
        if deadline is not None and time.time() >= deadline:
            break
        res = run_repair(cert, [{"id": host_id, "context": {"A": ctx[0], "J": ctx[1]},
                                  "nodes": [{"id": n, "type": t} for n, t in g.nodes.items()],
                                  "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                                            for e, (s, t, ty) in g.edges.items()]}],
                         T_max=10000, deadline=deadline)
        if res["per_host"][0]["steps"] > 0:
            host_hit += 1
        for r, m, H in one_step(cert, g, ctx):
            rule_hit_names.add(r.name)
            per_rule_ctx[r.name] += 1

        res_struct = run_repair(forced_all,
                                 [{"id": host_id, "context": {"A": 0, "J": "EU"},
                                   "nodes": [{"id": n, "type": t} for n, t in g.nodes.items()],
                                   "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                                             for e, (s, t, ty) in g.edges.items()]}],
                                 T_max=10000, deadline=deadline)
        if res_struct["per_host"][0]["steps"] > 0:
            struct_hit += 1
        for r, m, H in one_step(forced_all, g, (0, "EU")):
            per_rule_struct[r.name] += 1

        if non_ef:
            res_struct_nonef = run_repair(forced_nonef,
                                     [{"id": host_id, "context": {"A": 0, "J": "EU"},
                                       "nodes": [{"id": n, "type": t} for n, t in g.nodes.items()],
                                       "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                                                 for e, (s, t, ty) in g.edges.items()]}],
                                     T_max=10000, deadline=deadline)
            if res_struct_nonef["per_host"][0]["steps"] > 0:
                struct_hit_nonef += 1

    return {
        "host_coverage_n": f"{host_hit}/{n_hosts}",
        "host_coverage": host_hit / n_hosts if n_hosts else None,
        "structural_coverage_all_n": f"{struct_hit}/{n_hosts}",
        "structural_coverage_all": struct_hit / n_hosts if n_hosts else None,
        "structural_coverage_nonedgefree_n": f"{struct_hit_nonef}/{n_hosts}",
        "structural_coverage_nonedgefree": struct_hit_nonef / n_hosts if n_hosts else None,
        "rule_coverage_n": f"{len(rule_hit_names)}/{len(cert)}",
        "rule_coverage": len(rule_hit_names) / len(cert) if cert else None,
        "per_rule_match_counts": {
            r.name: {"structural": per_rule_struct[r.name], "real_context": per_rule_ctx[r.name],
                     "l_edges": len(r.L.edges)}
            for r in cert
        },
    }


def per_source_table4(candidates, cert):
    proposed_by_source, certified_by_source = {}, {}
    for r in candidates:
        proposed_by_source[r.source] = proposed_by_source.get(r.source, 0) + 1
    for r in cert:
        certified_by_source[r.source] = certified_by_source.get(r.source, 0) + 1
    rows = []
    for fname, (label, tier) in SOURCE_TIER.items():
        rows.append({"source": label, "tier": tier,
                     "proposed": proposed_by_source.get(fname, 0),
                     "certified": certified_by_source.get(fname, 0)})
    return rows


def run_one_config(config_name, candidates, hosts, hub_parse_hits, results, topo_rejections_extraction):
    log(f"=== {config_name} ===")
    result = {"n_candidates": len(candidates),
              "topology_rejections_at_extraction": topo_rejections_extraction,
              "topology_rejections_retroactive": compute_topology_rejections(candidates)}
    result["edgefree_available"] = sum(1 for r in candidates if not r.L.edges)
    result["hub_parse_rate"] = f"{hub_parse_hits}/{len(hosts)}"

    ablation_rows = {}
    full_cert = None
    full_cert_budget_exceeded = False
    for i, (name, flags) in enumerate(ABLATION_CONFIGS):
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
                    f"({time.time()-t_start:.0f}s elapsed) **")
            ablation_rows[key] = m
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
    result["ablation"] = ablation_rows
    result["certified_count_full_keep"] = len(full_cert) if full_cert is not None else None
    result["certified_count_full_keep_is_partial"] = full_cert_budget_exceeded
    result["table4_per_source_full_keep"] = per_source_table4(candidates, full_cert or [])

    log("  coverage metrics (3-way, full pipeline / edgefree-keep cert)...")
    result["coverage"] = coverage_metrics_v3(full_cert or [], hosts,
                                              deadline=time.time() + BUDGET_SECONDS)

    results[config_name] = result
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    log(f"  wrote results to {OUT_JSON}")
    return result


def _config_complete(results, name):
    r = results.get(name)
    if r is None:
        return False
    return len(r.get("ablation", {})) >= 8 and "coverage" in r


def run_task4(complete, sample):
    stats = {}
    log("Extracting the full n=200 sample canonically (P1-A base) ...")
    candidates_p1a = extract(complete, provisions=sample, cache_path=CANONICAL_CACHE,
                              repair_attempts=K_REPAIR, enforce_topology=True, stats=stats)
    log(f"P1-A extraction done: {len(candidates_p1a)} candidates, "
        f"topology_rejections={stats.get('topology_rejections', 0)}, "
        f"other_rejections={stats.get('other_rejections', 0)}")

    existing_hosts_raw = json.loads(Path(HOLDOUT_REAL).read_text(encoding="utf-8"))

    results = {}
    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            log(f"resuming: {[k for k in results if _config_complete(results, k)]} complete")
        except Exception:
            results = {}

    if not _config_complete(results, "P1-A"):
        hosts_p1a = [(h["id"], build_host_graph(h), (h["context"]["A"], h["context"]["J"]))
                     for h in existing_hosts_raw]
        hub_hits = sum(1 for _, g, _ in hosts_p1a if not canonical_violations(g))
        run_one_config("P1-A", candidates_p1a, hosts_p1a, hub_hits, results,
                        stats.get("topology_rejections", 0))

    if not _config_complete(results, "P1-AB"):
        log("canonicalizing P1-A candidates with kappa (P1-AB)...")
        kappa_candidates = [canonicalize_rule(r) for r in candidates_p1a]
        hosts_p1ab = []
        for h in existing_hosts_raw:
            g = canonicalize_graph(build_host_graph(h))
            hosts_p1ab.append((h["id"], g, (h["context"]["A"], h["context"]["J"])))
        hub_hits = sum(1 for _, g, _ in hosts_p1ab if not canonical_violations(g))
        run_one_config("P1-AB", kappa_candidates, hosts_p1ab, hub_hits, results,
                        stats.get("topology_rejections", 0))

    write_report(results)
    log("DONE (full extraction)")
    return results


def write_report(results):
    lines = ["# Phase 1-A results (P1-A, P1-AB x edgefree-keep/reject)", ""]
    for config_name, r in results.items():
        lines.append(f"## {config_name}")
        lines.append(f"- candidates: {r['n_candidates']}, "
                      f"topology_rejections (at extraction, k=3): {r['topology_rejections_at_extraction']}, "
                      f"topology_rejections (retroactive check): {r['topology_rejections_retroactive']}, "
                      f"edge-free available: {r['edgefree_available']}")
        lines.append(f"- hub parse rate: {r['hub_parse_rate']}")
        partial_note = " (PARTIAL -- budget exceeded)" if r.get("certified_count_full_keep_is_partial") else ""
        lines.append(f"- certified (Full pipeline, edgefree-keep): {r['certified_count_full_keep']}{partial_note}")
        cov = r["coverage"]
        lines.append(f"- coverage (i) host: {cov['host_coverage_n']}, "
                      f"(ii) structural (all rules): {cov['structural_coverage_all_n']}, "
                      f"(iii) structural (non-edge-free rules only): {cov['structural_coverage_nonedgefree_n']}")
        lines.append(f"- rule coverage: {cov['rule_coverage_n']}")
        lines.append("")
        lines.append("| Config | Term | Nonterm | Strat.rej | Repair | Cascade | Budget |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, m in r["ablation"].items():
            budget_col = "EXCEEDED (4h)" if m.get("budget_exceeded") else "ok"
            lines.append(f"| {key} | {m['term_rate']} | {m['nonterm']} | "
                          f"{m['strat_reject']} | {m['repair_success']:.2f} | {m['median_cascade']} | {budget_col} |")
        lines.append("")
        lines.append("Per-rule match counts (structural / real-context) out of 12 hosts:")
        for name, d in sorted(cov["per_rule_match_counts"].items(),
                               key=lambda kv: -kv[1]["structural"]):
            lines.append(f"  - {name} (|L edges|={d['l_edges']}): "
                          f"structural={d['structural']}/12, real_context={d['real_context']}/12")
        lines.append("")
        lines.append("Table 4 (per source, Full pipeline / edgefree-keep):")
        for row in r["table4_per_source_full_keep"]:
            lines.append(f"  - {row['source']} ({row['tier']}): "
                          f"proposed={row['proposed']}, certified={row['certified']}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sample = build_sample()
    complete = get_real_complete()

    log("=== PILOT (20 spans) ===")
    pilot_report = run_task3_pilot(complete, sample)
    PILOT_OUT_JSON.write_text(json.dumps(pilot_report, indent=2, ensure_ascii=False, default=str),
                               encoding="utf-8")
    log(f"pilot: first_try_canonical_rate={pilot_report['first_try_canonical_rate']:.2f} "
        f"final_canonical_rate={pilot_report['final_canonical_rate']:.2f} "
        f"topology_rejections_final={pilot_report['topology_rejections_final']} "
        f"edge_free_rate_of_final={pilot_report['edge_free_rate_of_final']}")

    if pilot_report["stop_rule_triggered"]:
        log("STOP RULE TRIGGERED: final_canonical_rate < 50%. Stopping here per the "
            "pre-set stop rule -- NOT running the full extraction.")
        sys.exit(0)

    log("Pilot passed the 50% stop rule -- continuing to the full extraction (P1-A/P1-AB).")
    run_task4(complete, sample)
