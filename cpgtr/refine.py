"""Stage 3: counterexample-guided refinement of rejected candidate rules.

Role in the pipeline: wraps `certify.admit()` (a pure graph algorithm that stays
LLM-agnostic) with a refinement loop. When a candidate is rejected, the formal
counterexample from the admission log (`detail`) is serialized and returned to
the extractor LLM with an instruction to tighten the rule's applicability:

  * `reject:non-joinable-CP` - the offending critical pair (shared overlap S and
    the two results H1, H2 that cannot be joined);
  * `reject:no-measure` - the creation-dependency cycle that prevents a
    stratification, given as the names of the peer rules in the cycle.

The feedback is a formal object (a critical pair or a dependency cycle) rather
than a natural-language error string, so the model is corrected against the
exact structural conflict. Refinement is bounded to `REFINE_ROUNDS` rounds.

Scope: refinement is applied to both rejection categories above. Rules
rejected by the stratification check are by far the most common rejection on
real extracted output, so rescuing even a few of them matters. The refinement
paths are logged separately in the returned ``refine_log``.

Inputs: raw extraction specs (spec dict, source) and an LLM client.
Outputs: the certified rule list, the final admission log, and the refinement
log (see `admit_with_refinement`).
"""
from __future__ import annotations
import json
import re

from .extract import json_to_rule, MAX_TOKENS_PER_ITEM
from .certify import admit as certify_admit

REFINE_ROUNDS = 3   # maximum refinement rounds per candidate


def _graph_to_json(g):
    """Serialize a `Graph` to the {"nodes", "edges"} JSON form of the prompts."""
    return {
        "nodes": [{"id": n, "type": t} for n, t in g.nodes.items()],
        "edges": [{"id": e, "src": s, "tgt": t, "type": ty}
                  for e, (s, t, ty) in g.edges.items()],
    }


_CPA_REFINE_PROMPT = """Your previously proposed rule was REJECTED during
certification because it conflicts with an already-certified rule, %(peer)s,
on a shared graph pattern: the two rules cannot always be reconciled to a
common outcome (they are not "strongly joinable"), which would make the
certified system's repair behaviour depend on the order rules happen to
fire -- not allowed.

YOUR PREVIOUS RULE:
%(rule)s

THE CONFLICT (a formal critical pair). S is the shared overlap both rules
match; H1 is the result of applying YOUR rule to S; H2 is the result of
applying %(peer)s to S. The conflict is that H1 and H2 cannot be brought back
to a common graph:
S = %(S)s
H1 (your rule's result) = %(H1)s
H2 (%(peer)s's result) = %(H2)s

Revise YOUR rule so this conflict no longer arises -- typically by adding or
strengthening a negative application condition (NAC) so your rule does not
fire on a pattern %(peer)s has already touched, or by narrowing your rule's
context predicate (age_op/age/jurisdiction) so it and %(peer)s are never both
active for the same reader. Keep the same underlying legal intent -- do not
simply make the rule inert. Return ONLY the corrected JSON object, same
schema as before, no commentary.
"""

_STRAT_REFINE_PROMPT = """Your previously proposed rule was REJECTED during
certification because it forms an unbreakable dependency cycle with: %(peer_names)s.
Your rule creates a structure that one of those rules' left-hand sides needs,
and (directly or through the others) one of them creates a structure YOUR
rule's left-hand side needs back -- so no consistent priority ordering
(stratification) exists between them, and admitting all of them could loop
forever.

YOUR PREVIOUS RULE:
%(rule)s

Revise YOUR rule so it no longer creates something those rules' left-hand
sides can match -- typically by not adding the conflicting element at all, or
by narrowing what your rule's right-hand side introduces. Keep the same
underlying legal intent. Return ONLY the corrected JSON object, same schema
as before, no commentary.
"""


def _validate(spec, source):
    """Return None if `spec` builds a valid Rule, else the error message."""
    try:
        json_to_rule(spec, source=source)
        return None
    except Exception as e:
        return str(e)


def refine_candidate(complete, spec, detail, status):
    """Make one refinement attempt for a single rejected candidate.

    Args:
        complete: str->str LLM client (called with a per-call ``max_tokens``).
        spec: the rejected candidate's extraction spec.
        detail: counterexample attached to the admission log entry (critical
            pair for `reject:non-joinable-CP`, cycle peers for
            `reject:no-measure`).
        status: the rejection status from the admission log.

    Returns the revised spec dict, or None if no formal counterexample is
    available for this status or the LLM call or its parse failed."""
    if status == "reject:non-joinable-CP" and detail:
        prompt = _CPA_REFINE_PROMPT % {
            "peer": detail["peer"],
            "rule": json.dumps(spec, indent=2),
            "S": json.dumps(_graph_to_json(detail["S"]), indent=2),
            "H1": json.dumps(_graph_to_json(detail["H1"]), indent=2),
            "H2": json.dumps(_graph_to_json(detail["H2"]), indent=2),
        }
        # The prompt embeds three extra graphs (S, H1, H2) on top of the rule,
        # and reasoning models spend part of the budget on hidden reasoning, so
        # the default single-item budget can truncate the reply mid-JSON. A
        # truncated reply would be misread as "the model cannot refine this".
        max_tokens = MAX_TOKENS_PER_ITEM * 3
    elif status == "reject:no-measure" and detail and detail.get("cycle_peers"):
        prompt = _STRAT_REFINE_PROMPT % {
            "rule": json.dumps(spec, indent=2),
            "peer_names": ", ".join(detail["cycle_peers"]),
        }
        max_tokens = MAX_TOKENS_PER_ITEM * 2
    else:
        return None   # e.g. reject:cpa-nonterm: no formal counterexample to return

    try:
        raw = complete(prompt, max_tokens=max_tokens)
        return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception:
        return None


def admit_with_refinement(specs, complete, dmax=None, use_cpa=True, use_strat=True,
                           contexts=None, k=REFINE_ROUNDS, verbose=False):
    """Admission (`certify.admit`) followed by up to `k` refinement rounds.

    Each round admits the current candidates, then asks the LLM to revise every
    rejected candidate using its formal counterexample; revised candidates that
    validate join the next round together with the already-admitted ones.

    Args:
        specs: list of ``(spec_dict, source)`` pairs: the parsed extraction
            output before Rule construction. Refinement needs the original
            spec (not just the built Rule) to show the model its previous
            output and to rebuild a Rule from the reply.
        complete: str->str LLM client for refinement calls.
        dmax, use_cpa, use_strat, contexts, verbose: passed through to
            `certify.admit` (ablation switches and context cells).
        k: maximum refinement rounds.

    Returns ``(cert, log, refine_log)``:
        cert, log: same shape as `certify.admit`'s return, for the final round
            (after refinement).
        refine_log: one dict per refinement attempt (not only successes):
            ``{"name", "round", "status_before", "outcome"}`` with outcome in
            {"admitted", "still_rejected", "invalid_response"}. The refinement
            success rate is the number of "admitted" outcomes divided by the
            number of attempts.

    Known limitation: an attempt whose response is unparseable or fails schema
    validation ("invalid_response") is dropped immediately, not retried; only a
    response that validates but is still rejected by admission
    ("still_rejected") is retried in later rounds. This can only under-count
    how many candidates refinement could rescue.
    """
    current = list(specs)
    refine_log = []
    pending = []      # refine_log entries awaiting resolution at the next admit()
    round_num = 0
    cert, log = [], []

    while True:
        candidates, by_name = [], {}
        for spec, source in current:
            try:
                r = json_to_rule(spec, source=source)
            except Exception:
                continue   # these specs have already passed _validate
            candidates.append(r)
            by_name[r.name] = (spec, source)

        cert, log = certify_admit(candidates, dmax=dmax, use_cpa=use_cpa,
                                   use_strat=use_strat, verbose=verbose, contexts=contexts)
        admitted_names = {r.name for r in cert}

        for entry in pending:
            entry["outcome"] = "admitted" if entry["refined_name"] in admitted_names else "still_rejected"
        pending = []

        if round_num >= k:
            return cert, log, refine_log

        rejected = [(name, status, detail) for name, status, _mod, detail in log
                    if status != "admit"]
        if not rejected:
            return cert, log, refine_log

        round_num += 1
        next_round = [(spec, source) for name, (spec, source) in by_name.items()
                      if name in admitted_names]

        for name, status, detail in rejected:
            spec, source = by_name[name]
            new_spec = refine_candidate(complete, spec, detail, status)
            if new_spec is None:
                refine_log.append({"name": name, "round": round_num,
                                   "status_before": status, "outcome": "invalid_response"})
                continue
            new_spec.setdefault("name", name)   # keep name stable unless the model changed it
            err = _validate(new_spec, source)
            if err is not None:
                refine_log.append({"name": name, "round": round_num,
                                   "status_before": status, "outcome": "invalid_response"})
                continue
            next_round.append((new_spec, source))
            entry = {"name": name, "round": round_num, "status_before": status,
                     "outcome": "pending", "refined_name": new_spec["name"]}
            refine_log.append(entry)
            pending.append(entry)

        current = next_round
