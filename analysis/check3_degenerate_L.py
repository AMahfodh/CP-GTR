"""Task 6 diagnosis, check 3 (user's decisive check): for the first 20
candidates in real admit()-sort order, log n_edges_in_L, critical pairs
enumerated against the currently-admitted set, and elapsed seconds for the
CPA step -- run separately for the raw (P1-0-style) and kappa-canonicalized
(P1-B-style) candidate lists, so the two are directly comparable.

Read-only / standalone: does not touch the (already-stopped) phase1_run.py
process or its outputs. Reuses admit()'s exact sort and stratify() call so
the "first 20" prefix and each candidate's rho match what a real admit()
call would do; only the main admission loop is truncated to 20 iterations.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from run_e2e import PROVISIONS, EXTRACTION_CACHE, deduplicate_provisions, stratified_sample
from cpgtr.extract import _provision_hash, MAX_TOKENS_PER_ITEM, _load_cache, json_to_rule
from cpgtr.canon import canonicalize_rule
from cpgtr.certify import stratify
from cpgtr.cpa import critical_pairs, strongly_joinable
from cpgtr.context import representative_contexts

MODEL = "gpt-oss-120b"


def load_raw_candidates():
    provisions = json.load(open(PROVISIONS, encoding="utf-8"))
    deduped, _ = deduplicate_provisions(provisions)
    sample = stratified_sample(deduped)
    cache = _load_cache(EXTRACTION_CACHE)
    candidates = []
    for p in sample:
        h = _provision_hash(MODEL, 0.0, MAX_TOKENS_PER_ITEM, p)
        entry = cache.get(h)
        candidates.append(json_to_rule(entry["spec"], source=entry["source"]))
    return candidates


def run_check(label, candidates, n=20):
    print(f"\n=== {label} ===", flush=True)
    candidates = sorted(candidates,
                         key=lambda r: (-(r.confidence if r.confidence is not None else -1),
                                        r.content_hash()))
    contexts = representative_contexts(candidates)
    print(f"  {len(candidates)} candidates, {len(contexts)} context cells", flush=True)

    t0 = time.time()
    stratify(candidates, None)
    print(f"  stratify() over all {len(candidates)}: {time.time()-t0:.2f}s", flush=True)

    cert = []
    for i, r in enumerate(candidates[:n]):
        n_edges_L = len(r.L.edges)
        t1 = time.time()
        if r.rho is None:
            print(f"  [{i:2d}] {r.name[:30]:30s} |L edges|={n_edges_L}  rho=None (strat-reject, skipped)", flush=True)
            continue
        cp_count = 0
        admit_ok = True
        nonterm = False
        try:
            for A, J in contexts:
                if not r.phi(A, J):
                    continue
                active = [x for x in cert if x.phi(A, J)] + [r]
                for p in active:
                    cps = critical_pairs(r, p)
                    cp_count += len(cps)
                    for (S, H1, H2, pers) in cps:
                        if not strongly_joinable(H1, H2, pers, active):
                            admit_ok = False
                            break
                    if not admit_ok:
                        break
                if not admit_ok:
                    break
        except RuntimeError:
            admit_ok, nonterm = False, True
        elapsed = time.time() - t1
        if admit_ok:
            cert.append(r)
        status = "cpa-nonterm(5000 pops)" if nonterm else ("admit" if admit_ok else "reject-CP")
        print(f"  [{i:2d}] {r.name[:30]:30s} |L edges|={n_edges_L}  "
              f"critical_pairs={cp_count:4d}  elapsed={elapsed:8.3f}s  {status}", flush=True)


if __name__ == "__main__":
    raw = load_raw_candidates()
    kappa = [canonicalize_rule(r) for r in raw]
    run_check("P1-0-style (raw candidates)", raw, n=20)
    run_check("P1-B-style (kappa-canonicalized candidates)", kappa, n=20)
    print("\nDONE", flush=True)
