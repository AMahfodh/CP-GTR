"""Stage 3: Refinement.

Wraps certify.admit() (which stays LLM-agnostic, a pure graph algorithm) with
a refinement loop: a rejected candidate's counterexample -- the offending
critical pair for a CPA rejection, or the creation-dependency cycle for a
stratification rejection, both attached to certify.admit()'s log as `detail`
(see certify.py's admit() docstring) -- is serialized as a FORMAL object and
returned to the extractor LLM with an instruction to tighten the rule's
applicability. The manuscript is explicit about why this matters: "the
feedback signal is a formal object, a critical pair, and not a
natural-language error string, so the model is corrected against the exact
structural conflict rather than a paraphrase of it." Bounded to k rounds per
candidate (k=3, matching the manuscript).


"""
from __future__ import annotations
import json
import re

from .extract import json_to_rule, MAX_TOKENS_PER_ITEM
from .certify import admit as certify_admit

REFINE_ROUNDS = 3   # CP-GTR_V2.tex Stage 3: "we use k=3"


def _graph_to_json(g):
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
    try:
        json_to_rule(spec, source=source)
        return None
    except Exception as e:
        return str(e)


def refine_candidate(complete, spec, detail, status):
    """One refinement attempt for a single rejected candidate. Returns the
    revised spec dict, or None if no formal counterexample is available for
    this status, or the LLM call/response parse failed outright."""
    if status == "reject:non-joinable-CP" and detail:
        prompt = _CPA_REFINE_PROMPT % {
            "peer": detail["peer"],
            "rule": json.dumps(spec, indent=2),
            "S": json.dumps(_graph_to_json(detail["S"]), indent=2),
            "H1": json.dumps(_graph_to_json(detail["H1"]), indent=2),
            "H2": json.dumps(_graph_to_json(detail["H2"]), indent=2),
        }
        # The CPA prompt embeds three extra full graphs (S, H1, H2) on top of
        # the rule itself -- a real run against Cerebras's gpt-oss-120b showed
        # the default single-item budget (MAX_TOKENS_PER_ITEM) truncating the
        # response mid-JSON (same failure class as extract.py's batch
        # truncation bug), which silently masqueraded as "the model can't
        # refine this" (invalid_response) when it was actually "the response
        # never finished."
        max_tokens = MAX_TOKENS_PER_ITEM * 3
    elif status == "reject:no-measure" and detail and detail.get("cycle_peers"):
        prompt = _STRAT_REFINE_PROMPT % {
            "rule": json.dumps(spec, indent=2),
            "peer_names": ", ".join(detail["cycle_peers"]),
        }
        max_tokens = MAX_TOKENS_PER_ITEM * 2
    else:
        return None   # e.g. reject:cpa-nonterm -- no formal counterexample to hand back

    try:
        raw = complete(prompt, max_tokens=max_tokens)
        return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception:
        return None


def admit_with_refinement(specs, complete, dmax=None, use_cpa=True, use_strat=True,
                           contexts=None, k=REFINE_ROUNDS, verbose=False):
    """Algorithm 3, Stage 2 (certify.admit) + Stage 3 (refinement) together.

    Args:
        specs: list of (spec_dict, source) pairs -- the raw parsed
            extraction output BEFORE Rule construction (see extract.py's
            internal cache entries: {"spec":..., "source":...}). Refinement
            needs the original spec (not just the built Rule) to show the
            model "your previous output" and to rebuild a revised Rule from
            its reply.
        complete: str->str LLM client for refinement calls (reuses the same
            client/session as Stage-1 extraction; refinement calls are
            logged identically via cpgtr.llm).
        k: refinement rounds (CP-GTR_V2.tex: k=3).

    Returns (cert, log, refine_log):
        cert, log: same shape as certify.admit()'s return, for the FINAL
            round (i.e. after refinement, not before).
        refine_log: list of dicts, one per refinement ATTEMPT (not just
            successes): {"name", "round", "status_before", "outcome"} with
            outcome in {"admitted", "still_rejected", "invalid_response"}.
            The refinement success rate CP-GTR_V2.tex calls a headline
            result ("the refinement success rate is itself reported as a
            result") is admitted-outcomes / total attempts from this log.

    Known simplification: a refinement attempt that returns an unparseable
    or still-schema-invalid response ("invalid_response") is dropped
    immediately rather than retried up to k times -- only a response that
    parses and validates but is still rejected by admission
    ("still_rejected") gets the full k-round retry budget. This is
    conservative (it can only under-count how many candidates refinement
    could have rescued, never over-count), not a correctness bug, but is a
    literal reading gap relative to "bounded to k rounds" applying uniformly
    to every failure mode.
    """
    current = list(specs)
    refine_log = []
    pending = []      # refine_log entries awaiting resolution against the NEXT admit() call
    round_num = 0
    cert, log = [], []

    while True:
        candidates, by_name = [], {}
        for spec, source in current:
            try:
                r = json_to_rule(spec, source=source)
            except Exception:
                continue   # specs here already passed _validate at least once
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
