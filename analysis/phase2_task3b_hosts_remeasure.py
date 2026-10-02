"""Task 3b (pasted task list, 2026-09-23): re-parse the 12 holdout hosts'
source_text canonically (LLMParser(enforce_topology=True)) now that
PARSE_MAX_TOKENS=4096 (commit 6f6fdf0) fixes the truncation active when
corpus/holdout_hosts_real_canonical.json and analysis/phase2_gate.md's Task 1
table were produced. Reports node/edge counts per host vs. the earlier
(truncated) run, and whether real002/real010 are still sparser. Real API
calls (CPGTR_MODE=real). Writes corpus/holdout_hosts_real_canonical_v2.json
-- does NOT overwrite the existing (truncated-cap) file.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cpgtr.parse import LLMParser
from cpgtr.llm import make_complete

hosts = json.loads((REPO_ROOT / "corpus" / "holdout_hosts_real.json").read_text(encoding="utf-8"))
old_canon = json.loads((REPO_ROOT / "corpus" / "holdout_hosts_real_canonical.json").read_text(encoding="utf-8"))
old_by_id = {e["id"]: e for e in old_canon}

complete = make_complete()
parser = LLMParser(complete, enforce_topology=True)

new_canon = []
print(f"{'host':10s} {'old N':>6s} {'new N':>6s} {'old E':>6s} {'new E':>6s}")
for h in hosts:
    doc = parser.parse(h["source_text"])
    nodes = [{"id": nid, "type": t} for nid, t in doc.graph.nodes.items()]
    edges = [{"id": eid, "src": s, "tgt": t, "type": ty} for eid, (s, t, ty) in doc.graph.edges.items()]
    entry = dict(h)
    entry["nodes"] = nodes
    entry["edges"] = edges
    new_canon.append(entry)

    old = old_by_id[h["id"]]
    print(f"{h['id']:10s} {len(old['nodes']):6d} {len(nodes):6d} {len(old['edges']):6d} {len(edges):6d}")

(REPO_ROOT / "corpus" / "holdout_hosts_real_canonical_v2.json").write_text(
    json.dumps(new_canon, indent=2), encoding="utf-8")
print("\nWritten: corpus/holdout_hosts_real_canonical_v2.json")
