"""Topology inventory of extracted rules versus parsed hosts.

Supports: the deployment-realism finding that, before a shared topology
convention existed, the extractor and the parser used disjoint arrangements of
the same node types (the extractor's most frequent pattern was
Operator --permits--> DataProcessing, which no host contains, while the parser
recorded consent as ConsentProcess --grants--> DataProcessing). It reports the
left-pattern signatures of the 200 cached candidates, the arrangement
conflicts per node-type pair, and the signatures of the twelve real hosts,
side by side, without proposing any mapping between them.

Read-only and offline: no model calls, and no existing file is modified. The
type vocabulary is read from the source text of cpgtr/extract.py and
cpgtr/parse.py through the AST, so nothing is imported or executed there. The
one non-trivial computation is admitting the cached candidates (a pure graph
algorithm) to know which are certified, because signature frequencies are
reported both overall and for the certified subset.

Run:
    python analysis/topology_inventory.py
Order: independent diagnostic of the pre-canonical extraction; it needs only
the cached extraction and host files, and can run at any time.

Inputs : cache/real_extraction_cache.jsonl (LLM-generated candidates from the
         first, non-canonical extraction of 200 provisions, seed 42),
         corpus/holdout_hosts_real.json (twelve hosts parsed from drafted
         text), cpgtr/extract.py, cpgtr/parse.py.
Outputs: analysis/topology_inventory.json,
         analysis/topology_inventory_summary.md.
"""
from __future__ import annotations
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXTRACTION_CACHE = REPO_ROOT / "cache" / "real_extraction_cache.jsonl"
HOLDOUT_REAL = REPO_ROOT / "corpus" / "holdout_hosts_real.json"
EXTRACT_PY = REPO_ROOT / "cpgtr" / "extract.py"
PARSE_PY = REPO_ROOT / "cpgtr" / "parse.py"
OUT_JSON = Path(__file__).resolve().parent / "topology_inventory.json"
OUT_MD = Path(__file__).resolve().parent / "topology_inventory_summary.md"


def _find_set_literals(source_text, varname):
    """Every `varname = {...}` assignment anywhere in the module, at module
    level or nested inside a method. cpgtr/parse.py's LLMParser.parse()
    defines TG_NODES and TG_EDGES as local variables, not module constants,
    so a plain import of them would not work. Returns a list of sets (there
    can be more than one assignment site); does not import or execute the
    module."""
    tree = ast.parse(source_text)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == varname:
                    try:
                        found.append(set(ast.literal_eval(node.value)))
                    except Exception:
                        pass
    return found


def load_tg_vocab():
    extract_src = EXTRACT_PY.read_text(encoding="utf-8")
    parse_src = PARSE_PY.read_text(encoding="utf-8")
    extract_nodes = _find_set_literals(extract_src, "TG_NODES")
    extract_edges = _find_set_literals(extract_src, "TG_EDGES")
    parse_nodes = _find_set_literals(parse_src, "TG_NODES")
    parse_edges = _find_set_literals(parse_src, "TG_EDGES")
    assert len(extract_nodes) == 1, f"expected exactly 1 TG_NODES in extract.py, found {len(extract_nodes)}"
    assert len(extract_edges) == 1, f"expected exactly 1 TG_EDGES in extract.py, found {len(extract_edges)}"
    return {
        "extract_py": {
            "TG_NODES": sorted(extract_nodes[0]),
            "TG_EDGES": sorted(extract_edges[0]),
        },
        "parse_py": {
            # parse.py may define TG_NODES/TG_EDGES more than once (e.g. once
            # per parser class); report every occurrence found.
            "TG_NODES_occurrences": [sorted(s) for s in parse_nodes],
            "TG_EDGES_occurrences": [sorted(s) for s in parse_edges],
        },
        "identical": {
            "nodes": all(s == extract_nodes[0] for s in parse_nodes) if parse_nodes else None,
            "edges": all(s == extract_edges[0] for s in parse_edges) if parse_edges else None,
        },
    }


def load_cache_entries():
    entries = []
    with open(EXTRACTION_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entries.append(rec)
    return entries


def spec_to_graph_dicts(spec):
    """Return (nodes: {id: type}, edges: [(id, src, tgt, type)]) for spec['L'],
    built directly from the cached spec JSON, reading the stored L field
    without invoking any cpgtr.extract parsing logic."""
    L = spec["L"]
    nodes = {n["id"]: n["type"] for n in L.get("nodes", [])}
    edges = [(e["id"], e["src"], e["tgt"], e["type"]) for e in L.get("edges", [])]
    return nodes, edges


def signature_of(nodes, edges):
    """Multiset of (source type, edge type, target type) triples."""
    triples = []
    for _id, src, tgt, ety in edges:
        src_t = nodes.get(src, "?UNKNOWN?")
        tgt_t = nodes.get(tgt, "?UNKNOWN?")
        triples.append((src_t, ety, tgt_t))
    return tuple(sorted(triples))


def connected_components(nodes, edges):
    """Count connected components of L, treating edges as undirected (the
    structural connectivity of the pattern, not its direction). Isolated nodes
    count as singleton components."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _id, src, tgt, _ety in edges:
        if src in parent and tgt in parent:
            union(src, tgt)

    roots = {find(n) for n in nodes}
    return len(roots)


def triple_str(t):
    return f"{t[0]}--{t[1]}-->{t[2]}"


def build_candidate_records(entries):
    records = []
    for rec in entries:
        spec = rec["spec"]
        h = rec["hash"]
        name = spec.get("name", "?")
        nodes, edges = spec_to_graph_dicts(spec)
        sig = signature_of(nodes, edges)
        n_cc = connected_components(nodes, edges)
        records.append({
            "candidate_id": h[:16],
            "hash": h,
            "name": name,
            "source": rec.get("source") or spec.get("source"),
            "confidence": spec.get("confidence"),
            "signature": [triple_str(t) for t in sig],
            "n_edges_in_L": len(edges),
            "n_nodes_in_L": len(nodes),
            "n_connected_components": n_cc,
        })
    return records


def compute_certified_hashes(entries):
    """Run the model-free admission gate (cpgtr.certify/cpa/admit) on the
    cached candidates to find which are certified. No API calls: it is a graph
    algorithm over already-extracted specs.

    Admitted Rule objects are matched back to their cache entries by object
    identity (id()), not by rule name. Names are not unique among real
    candidates (several candidates are all named "Provision0", an artifact of
    model naming), so matching by name would tag every candidate that shares a
    certified rule's name as certified and over-count."""
    from cpgtr.extract import json_to_rule
    from cpgtr.admit import admit

    candidates = []
    for rec in entries:
        r = json_to_rule(rec["spec"], source=rec.get("source") or rec["spec"].get("source"))
        candidates.append(r)
    cert, _log = admit(candidates)
    cert_ids = {id(r) for r in cert}
    certified_hashes = {entries[i]["hash"] for i, r in enumerate(candidates) if id(r) in cert_ids}
    certified_names = sorted({r.name for r in cert})
    return certified_hashes, certified_names, len(candidates)


def signature_frequencies(records, subset_hashes=None):
    counter = Counter()
    for rec in records:
        if subset_hashes is not None and rec["hash"] not in subset_hashes:
            continue
        key = " | ".join(rec["signature"]) if rec["signature"] else "(empty L)"
        counter[key] += 1
    return dict(sorted(counter.items(), key=lambda kv: -kv[1]))


def arrangement_conflicts(records):
    """For every unordered pair of node types that appear as the two
    endpoints of some direct L edge, all distinct (src_type, edge_type,
    tgt_type) arrangements observed connecting that pair, with counts and up
    to 2 example candidate ids. Only pairs with more than one distinct
    arrangement are included."""
    pair_arrangements = defaultdict(lambda: defaultdict(lambda: {"count": 0, "examples": []}))
    for rec in records:
        seen_this_candidate = set()
        for t_str in rec["signature"]:
            src_t, rest = t_str.split("--", 1)
            ety, tgt_t = rest.split("-->", 1)
            pair = tuple(sorted((src_t, tgt_t)))
            arrangement = (src_t, ety, tgt_t)
            key = (pair, arrangement)
            if key in seen_this_candidate:
                continue
            seen_this_candidate.add(key)
            entry = pair_arrangements[pair][arrangement]
            entry["count"] += 1
            if len(entry["examples"]) < 2:
                entry["examples"].append(rec["candidate_id"])

    conflicts = {}
    for pair, arrangements in pair_arrangements.items():
        if len(arrangements) > 1:
            pair_key = f"{pair[0]} <-> {pair[1]}"
            conflicts[pair_key] = [
                {
                    "arrangement": triple_str(arr),
                    "count": data["count"],
                    "example_candidate_ids": data["examples"],
                }
                for arr, data in sorted(arrangements.items(), key=lambda kv: -kv[1]["count"])
            ]
    return conflicts


def build_host_records():
    hosts = json.loads(HOLDOUT_REAL.read_text(encoding="utf-8"))
    records = []
    for h in hosts:
        nodes = {n["id"]: n["type"] for n in h.get("nodes", [])}
        edges = [(e["id"], e["src"], e["tgt"], e["type"]) for e in h.get("edges", [])]
        sig = signature_of(nodes, edges)
        records.append({
            "host_id": h.get("id"),
            "context": h.get("context"),
            "signature": [triple_str(t) for t in sig],
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_connected_components": connected_components(nodes, edges),
        })
    return records


def main():
    print("[topology-inventory] loading TG vocab from source text (no import)...")
    vocab = load_tg_vocab()

    print("[topology-inventory] loading cached candidates...")
    entries = load_cache_entries()
    print(f"[topology-inventory] {len(entries)} cached candidate entries")

    print("[topology-inventory] building candidate L-signatures...")
    records = build_candidate_records(entries)

    print("[topology-inventory] running admit() to identify the certified subset "
          "(pure graph algorithm, no LLM calls -- this is the slow step)...")
    certified_hashes, certified_names, n_candidates_built = compute_certified_hashes(entries)
    print(f"[topology-inventory] {n_candidates_built} candidates built, "
          f"{len(certified_hashes)} certified")

    overall_freq = signature_frequencies(records)
    certified_freq = signature_frequencies(records, subset_hashes=certified_hashes)
    conflicts = arrangement_conflicts(records)
    host_records = build_host_records()
    host_conflicts_context = arrangement_conflicts(
        [{"candidate_id": h["host_id"], "signature": h["signature"], "hash": h["host_id"]}
         for h in host_records]
    )

    out = {
        "tg_vocabulary": vocab,
        "n_candidates": len(records),
        "n_certified": len(certified_hashes),
        "candidates": records,
        "certified_names": sorted(certified_names),
        "signature_frequencies_overall": overall_freq,
        "signature_frequencies_certified_only": certified_freq,
        "arrangement_conflicts_rule_side": conflicts,
        "holdout_hosts_real": host_records,
        "arrangement_conflicts_host_side": host_conflicts_context,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[topology-inventory] wrote {OUT_JSON}")

    write_summary(out)
    print(f"[topology-inventory] wrote {OUT_MD}")


def write_summary(out):
    lines = []
    lines.append("# Topology inventory summary")
    lines.append("")
    lines.append("Read-only inventory. No LLM/API calls; "
                  "no existing file modified.")
    lines.append("")
    lines.append("## TG vocabulary")
    v = out["tg_vocabulary"]
    lines.append(f"- `cpgtr/extract.py` TG_NODES ({len(v['extract_py']['TG_NODES'])}): "
                  f"{', '.join(v['extract_py']['TG_NODES'])}")
    lines.append(f"- `cpgtr/extract.py` TG_EDGES ({len(v['extract_py']['TG_EDGES'])}): "
                  f"{', '.join(v['extract_py']['TG_EDGES'])}")
    for i, occ in enumerate(v["parse_py"]["TG_NODES_occurrences"]):
        lines.append(f"- `cpgtr/parse.py` TG_NODES occurrence #{i+1} ({len(occ)}): "
                      f"{', '.join(sorted(occ))}")
    for i, occ in enumerate(v["parse_py"]["TG_EDGES_occurrences"]):
        lines.append(f"- `cpgtr/parse.py` TG_EDGES occurrence #{i+1} ({len(occ)}): "
                      f"{', '.join(sorted(occ))}")
    lines.append(f"- Identical to extract.py's? nodes={v['identical']['nodes']}, "
                  f"edges={v['identical']['edges']}")
    lines.append("")

    lines.append(f"## Candidates: {out['n_candidates']} total, {out['n_certified']} certified")
    lines.append("")
    lines.append(f"Distinct L-signatures overall: {len(out['signature_frequencies_overall'])}")
    lines.append(f"Distinct L-signatures among certified: {len(out['signature_frequencies_certified_only'])}")
    lines.append("")

    lines.append("## Top arrangement conflicts (rule side)")
    lines.append("")
    lines.append("Node-type pairs with more than one distinct direct-edge arrangement "
                  "observed across the 200 candidates, sorted by total occurrence count:")
    lines.append("")
    conflicts = out["arrangement_conflicts_rule_side"]
    ranked = sorted(conflicts.items(),
                     key=lambda kv: -sum(a["count"] for a in kv[1]))
    for pair, arrangements in ranked[:20]:
        total = sum(a["count"] for a in arrangements)
        lines.append(f"- **{pair}** ({total} occurrences, {len(arrangements)} arrangements):")
        for a in arrangements:
            lines.append(f"  - `{a['arrangement']}` x{a['count']} "
                          f"(e.g. {', '.join(a['example_candidate_ids'])})")
    lines.append("")

    lines.append("## Host-side signatures (12 real holdout hosts, reported side by side -- no mapping proposed)")
    lines.append("")
    for h in out["holdout_hosts_real"]:
        lines.append(f"- `{h['host_id']}` (A={h['context']['A']}, J={h['context']['J']}, "
                      f"{h['n_connected_components']} component(s)): "
                      f"{'; '.join(h['signature']) if h['signature'] else '(empty)'}")
    lines.append("")

    lines.append("## Observation relevant to the 3/12 host coverage finding")
    lines.append("")
    lines.append("This section only juxtaposes rule-side and host-side arrangements already "
                  "listed above; it proposes no equivalence mapping (equivalences are approved "
                  "individually in schema/equivalences.yaml).")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
