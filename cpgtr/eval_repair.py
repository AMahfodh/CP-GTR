"""Real repair experiment.

Drives each holdout host through the certified rule set's deterministic
derivation (same single-path policy as cpgtr.rules.normalize) and classifies
the outcome:

  success       -- reached a normal form (compliant) within T_max steps.
  nonterm       -- CONFIRMED non-terminating: a graph state isomorphic to an
                   earlier state in this host's own derivation recurred, so the
                   derivation is a genuine cycle, not just slow. This matches
                   CP-GTR:1496-1499's definition ("... confirmed
                   non-terminating by cycle detection on the derivation").
  inconclusive  -- exhausted T_max steps without a confirmed cycle. The
                   manuscript's incident definition requires BOTH exceeding the
                   cutoff AND cycle confirmation; a host that hits the cutoff
                   without a detected cycle does not meet that bar and is kept
                   in a separate bucket rather than silently folded into either
                   "success" or "nonterm_incidents".


"""
from __future__ import annotations
from .graph import Graph, iso
from .rules import one_step


def _host_to_graph(entry):
    """A holdout host is a hand-authored, already-non-compliant SLG (see
    corpus/holdout_hosts.json) -- this experiment tests the REPAIR mechanism,
    not the parser (that is Table 5's job), so hosts are graphs directly, not
    natural-language text to be parsed."""
    G = Graph()
    for n in entry["nodes"]:
        G.add_node(n["id"], n["type"])
    for e in entry["edges"]:
        G.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return G


def _signature(G):
    """Cheap, id-independent bucket key. NOT a substitute for isomorphism --
    distinct non-isomorphic graphs can collide -- only a hash bucket so we run
    the expensive exact iso() check only against same-signature candidates
    instead of every previously visited state."""
    nsig = tuple(sorted(G.nodes.values()))
    esig = tuple(sorted(
        (G.nodes.get(s), ty, G.nodes.get(t)) for (s, t, ty) in G.edges.values()
    ))
    return (nsig, esig)


def _run_one_host(rules, G0, ctx, t_max):
    """Deterministic single-path derivation; returns (outcome, steps) where
    outcome in {"success", "nonterm", "inconclusive"}."""
    seen = {}
    cur = G0
    seen.setdefault(_signature(cur), []).append(cur)

    for step in range(1, t_max + 1):
        st = one_step(rules, cur, ctx)
        if not st:
            return "success", step - 1
        cur = st[0][2]
        sig = _signature(cur)
        bucket = seen.setdefault(sig, [])
        if any(iso(cur, prior) for prior in bucket):
            return "nonterm", step
        bucket.append(cur)

    return "inconclusive", t_max


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def run_repair(cert, holdout, T_max):
    """
    Drive non-compliant holdout hosts toward a compliant normal form.

    Args:
        cert:    certified rule set (as returned by admit())
        holdout: list of holdout-host dicts, each with "context": {"A", "J"},
                 "nodes": [{"id","type"}], "edges": [{"id","src","tgt","type"}]
        T_max:   step budget per host

    Returns dict with:
        "success_fraction": float  -- fraction reaching a normal form within T_max
        "nonterm_incidents": int   -- confirmed (cycle-detected) non-termination count
        "inconclusive": int        -- exhausted T_max without a confirmed cycle
        "median_cascade": float|None -- median steps-to-compliant among successes
        "per_host": list[dict]     -- per-host outcome, for audit
    """
    n_success = n_nonterm = n_inconclusive = 0
    cascades = []
    per_host = []

    for entry in holdout:
        G0 = _host_to_graph(entry)
        ctx = (entry["context"]["A"], entry["context"]["J"])
        outcome, steps = _run_one_host(cert, G0, ctx, T_max)
        per_host.append({"id": entry.get("id"), "outcome": outcome, "steps": steps})
        if outcome == "success":
            n_success += 1
            cascades.append(steps)
        elif outcome == "nonterm":
            n_nonterm += 1
        else:
            n_inconclusive += 1

    total = len(holdout)
    return {
        "success_fraction": (n_success / total) if total else 0.0,
        "nonterm_incidents": n_nonterm,
        "inconclusive": n_inconclusive,
        "median_cascade": _median(cascades),
        "per_host": per_host,
    }
