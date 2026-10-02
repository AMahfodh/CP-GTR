"""Repair-to-normal-form experiment with cycle detection.

Role in the pipeline: runs each held-out non-compliant host graph (see
``corpus/holdout_hosts.json``) through the certified rule set's deterministic
derivation (the same single-path policy as `cpgtr.rules.normalize`) and
classifies the outcome:

  success         - reached a normal form (compliant) within ``T_max`` steps.
  nonterm         - confirmed non-terminating: a graph state isomorphic to an
                    earlier state of the same derivation recurred, so the
                    derivation is a genuine cycle rather than merely slow.
  inconclusive    - exhausted ``T_max`` steps without a confirmed cycle. A host
                    that only hits the cutoff does not meet the bar for a
                    non-termination incident (cutoff exceeded and cycle
                    confirmed), so it is kept in its own bucket rather than
                    folded into success or non-termination.
  budget-exceeded - cut short by the optional wall-clock deadline.

Hosts are hand-authored graphs, not text: this experiment tests the repair
mechanism, not the parser. Reported metrics are the success fraction, the
number of confirmed non-termination incidents, and the median cascade depth
(steps to reach a normal form among successful hosts).
"""
from __future__ import annotations
import time
from .graph import Graph, iso
from .rules import one_step


def _host_to_graph(entry):
    """Build a `Graph` from a holdout-host entry (nodes and edges given directly,
    since hosts are graphs rather than natural-language text)."""
    G = Graph()
    for n in entry["nodes"]:
        G.add_node(n["id"], n["type"])
    for e in entry["edges"]:
        G.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return G


def _signature(G):
    """Cheap, id-independent bucket key for a graph (sorted node types and
    typed edges).

    Not a substitute for isomorphism: distinct non-isomorphic graphs can share a
    signature. It only limits the exact `iso()` check to previously visited
    states in the same bucket."""
    nsig = tuple(sorted(G.nodes.values()))
    esig = tuple(sorted(
        (G.nodes.get(s), ty, G.nodes.get(t)) for (s, t, ty) in G.edges.values()
    ))
    return (nsig, esig)


def _run_one_host(rules, G0, ctx, t_max, deadline=None):
    """Run one host's deterministic single-path derivation.

    Returns ``(outcome, steps)`` with outcome in {"success", "nonterm",
    "inconclusive", "budget-exceeded"}. A cycle is declared when a new state is
    isomorphic to an earlier state of this derivation.

    `deadline` is an optional ``time.time()`` cutoff. Cycle detection only
    fires on a structurally repeated state; a purely additive cascade (each
    step creates fresh structure) never repeats, and the cost per step grows
    with the graph, so ``t_max`` bounds the step count but not wall-clock
    time. The deadline is checked once before each step, so a step's result is
    never reported as final when only partially computed."""
    seen = {}
    cur = G0
    seen.setdefault(_signature(cur), []).append(cur)

    for step in range(1, t_max + 1):
        if deadline is not None and time.time() >= deadline:
            return "budget-exceeded", step - 1
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
    """Median of a list of numbers, or None for an empty list."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def run_repair(cert, holdout, T_max, deadline=None):
    """Drive non-compliant holdout hosts toward a compliant normal form.

    Args:
        cert:    certified rule set (as returned by `admit()`)
        holdout: list of holdout-host dicts, each with "context": {"A", "J"},
                 "nodes": [{"id","type"}], "edges": [{"id","src","tgt","type"}]
        T_max:   step budget per host
        deadline: optional ``time.time()``-style wall-clock cutoff, applied to
                  each host in turn. Once it has passed, every host not yet
                  started is recorded as "budget-exceeded" with steps=0
                  rather than skipped silently (see `_run_one_host`).

    Returns dict with:
        "success_fraction": float  -- fraction reaching a normal form within T_max
        "nonterm_incidents": int   -- confirmed (cycle-detected) non-termination count
        "inconclusive": int        -- exhausted T_max without a confirmed cycle
        "budget_exceeded": int     -- cut short by the wall-clock deadline, not T_max
        "median_cascade": float|None -- median steps-to-compliant among successes
        "per_host": list[dict]     -- per-host outcome, for audit
    """
    n_success = n_nonterm = n_inconclusive = n_budget = 0
    cascades = []
    per_host = []

    for entry in holdout:
        if deadline is not None and time.time() >= deadline:
            per_host.append({"id": entry.get("id"), "outcome": "budget-exceeded", "steps": 0})
            n_budget += 1
            continue
        G0 = _host_to_graph(entry)
        ctx = (entry["context"]["A"], entry["context"]["J"])
        outcome, steps = _run_one_host(cert, G0, ctx, T_max, deadline=deadline)
        per_host.append({"id": entry.get("id"), "outcome": outcome, "steps": steps})
        if outcome == "success":
            n_success += 1
            cascades.append(steps)
        elif outcome == "nonterm":
            n_nonterm += 1
        elif outcome == "budget-exceeded":
            n_budget += 1
        else:
            n_inconclusive += 1

    total = len(holdout)
    return {
        "success_fraction": (n_success / total) if total else 0.0,
        "nonterm_incidents": n_nonterm,
        "inconclusive": n_inconclusive,
        "budget_exceeded": n_budget,
        "median_cascade": _median(cascades),
        "per_host": per_host,
    }
