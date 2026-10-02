"""Realization: apply fired repair rules to the source text, then verify.

Role in the pipeline: after a policy has been parsed into a graph and driven to
a normal form by the certified rules, this module edits the original sentences
to match, and checks the result.

  * `realize` edits the source sentences according to the derivation trace and
    each rule's modality (additive, subtractive or substitutive), using the
    rule's natural-language template. Before deleting or replacing a sentence it
    verifies that the sentence really supports the edge being removed.
  * `round_trip` re-parses the edited text and checks that it is a normal form
    (no certified rule still matches).
  * `process_document` runs the whole lane (parse, normalize, realize, check)
    and returns the output together with a compliance flag: an output that does
    not re-parse to a normal form is flagged for review rather than treated as
    repaired.
"""
from __future__ import annotations
from .graph import iso
from .rules import one_step

# Number of verification-only parse calls made by _sentence_verifies_edge (a
# real LLM call for LLMParser, free for RuleParser). Read it to report the extra
# call count for a run; reset to 0 before a measured run if the process is
# reused.
verification_call_count = 0


def normalize_trace(rules, G, ctx, limit=5000):
    """Drive graph `G` to a normal form under `rules` in context `ctx`.

    Returns ``(normal_form_graph, trace)`` where `trace` lists the
    ``(rule, match)`` pairs actually fired, in order. Always applies the first
    available step, so the result is deterministic. Raises RuntimeError if more
    than `limit` steps are taken (suspected non-termination)."""
    cur, trace, budget = G, [], limit
    while True:
        budget -= 1
        if budget < 0:
            raise RuntimeError("non-termination incident")
        st = one_step(rules, cur, ctx)
        if not st:
            return cur, trace
        r, m, H = st[0]
        trace.append((r, m))
        cur = H


def _edge_type_signature(graph, edge_id):
    """Return (source node type, edge type, target node type) for `edge_id`.

    This identifies an edge structurally, independent of the arbitrary node and
    edge ids a fresh parse assigns."""
    src, tgt, etype = graph.edges[edge_id]
    return (graph.nodes[src], etype, graph.nodes[tgt])


def _sentence_verifies_edge(parser, sentence, target_signature):
    """Re-parse `sentence` on its own and test whether it supports an edge.

    True if the resulting graph contains an edge with the same
    (source type, edge type, target type) signature as `target_signature`, the
    edge a subtractive or substitutive repair is about to act on."""
    global verification_call_count
    verification_call_count += 1
    solo = parser.parse(sentence)
    return any(_edge_type_signature(solo.graph, eid) == target_signature
               for eid in solo.graph.edges)


def realize(doc, trace, parser):
    """Apply the text edits implied by the fired rules.

    Returns ``(text, log)``. By rule modality:

      additive     -> append the rule's template sentence
      subtractive  -> delete the source sentence(s) of the removed edge(s),
                      and, if the rule has a template, append it (it states
                      the now-absent fact)
      substitutive -> replace the source sentence of the removed edge with the
                      template

    If several fired rules produce the identical template sentence (exact string
    match, no paraphrase merging), it is appended once, in first-occurrence
    order, so a notice is never repeated.

    Provenance verification: `doc.prov` can attribute an edge to the wrong
    sentence (for example a sale edge attributed to a sentence that denies
    selling). So, for every subtractive or substitutive edge removal, the
    attributed sentence is re-parsed alone and must yield an edge with the same
    (source type, edge type, target type) signature. If it does not, the other
    sentences are tried in order and the first that verifies is used. If none
    verifies, the edge is not removed; it stays in `doc.graph`, so the
    round-trip re-parse in `process_document` detects it and flags the
    document. A rule's template is appended only if at least one of its edges
    was verifiably removed, so a fix is never claimed that did not happen.
    Additive repairs do not use provenance and make no extra parse calls.

    `log` has one record per attempted subtractive/substitutive edge removal:
    ``{rule, edge_id, attributed_idx, attributed_sentence, attributed_verified,
    verified_idx, verified_sentence, flagged}``. Each attempt costs one extra
    `parser.parse()` call when the attributed sentence verifies, more if other
    sentences must be tried.
    """
    sents = list(doc.sentences)
    remove, append, seen = set(), [], set()
    log = []

    def add_once(text):
        if text not in seen:
            seen.add(text)
            append.append(text)

    for rule, m in trace:
        mod = rule.modality()
        if mod == "additive":
            if rule.template:
                add_once(rule.template)
            continue

        any_verified = False
        for le in rule.del_edges():
            src_edge = m["edges"].get(le)
            attributed_idx = doc.prov.get(src_edge)
            if attributed_idx is None:
                continue
            target_sig = _edge_type_signature(doc.graph, src_edge)
            attributed_sentence = sents[attributed_idx] if attributed_idx < len(sents) else None

            verified_idx, verified_sentence, attributed_verified = None, None, False
            if attributed_sentence is not None and \
               _sentence_verifies_edge(parser, attributed_sentence, target_sig):
                verified_idx, verified_sentence, attributed_verified = attributed_idx, attributed_sentence, True
            else:
                for i, s in enumerate(sents):
                    if i == attributed_idx:
                        continue
                    if _sentence_verifies_edge(parser, s, target_sig):
                        verified_idx, verified_sentence = i, s
                        break

            flagged = verified_idx is None
            log.append({
                "rule": rule.name, "edge_id": src_edge,
                "attributed_idx": attributed_idx, "attributed_sentence": attributed_sentence,
                "attributed_verified": attributed_verified,
                "verified_idx": verified_idx, "verified_sentence": verified_sentence,
                "flagged": flagged,
            })

            if verified_idx is not None:
                any_verified = True
                if mod == "substitutive" and rule.template:
                    sents[verified_idx] = rule.template
                else:                       # subtractive
                    remove.add(verified_idx)

        if mod == "subtractive" and rule.template and any_verified:
            add_once(rule.template)
    kept = [s for i, s in enumerate(sents) if i not in remove]
    return " ".join(kept + append), log


def round_trip(parser, rules, ctx, output_text):
    """Round-trip check: re-parse `output_text` and test it for a normal form.

    Returns ``(is_normal_form, redoc)``. True means no certified rule still
    matches the re-parsed graph ("compliant"); False means the output must be
    flagged for review."""
    redoc = parser.parse(output_text)
    residual = one_step(rules, redoc.graph, ctx)
    return len(residual) == 0, redoc


def process_document(parser, rules, ctx, text, doc=None):
    """Full deployment lane: parse, normalize, realize, re-check.

    Args:
        parser: a `RuleParser` or `LLMParser`.
        rules: certified rule list.
        ctx: the (age, jurisdiction) context.
        text: the source policy text.
        doc: optional pre-parsed `Document` for `text`, to avoid a redundant
            (LLM) parse when the caller has already parsed the same input. The
            round-trip re-parse is never skipped: it parses the new output.

    Returns a dict with, among others:
        output: the attempted-repair text, always provided for inspection.
        scored_text: the text to use for compliance measurement: `output` if
            the round-trip check passed, otherwise the unrepaired `text`. A
            flagged document is thus never scored as if it had been repaired.
        compliant: the round-trip result; False means flagged for review.
        violations_before / rules_fired: rule names matching the input and
            the rules applied.
        provenance_*: per-removal verification detail (see `realize`).
        input_parse_fallback / round_trip_parse_fallback: whether either parse
            was a `RuleParser` fallback substituted by `LLMParser`.
        redoc_*: edge types and sizes of the round-trip re-parse.
        input_negation_drops / round_trip_negation_drops: edges removed by the
            parser's negation filter in each parse.
    """
    doc = doc if doc is not None else parser.parse(text)
    before_viol = [r.name for (r, _, _) in one_step(rules, doc.graph, ctx)]
    Gstar, trace = normalize_trace(rules, doc.graph, ctx)
    output, verification_log = realize(doc, trace, parser)
    ok, redoc = round_trip(parser, rules, ctx, output)
    return {
        "context": ctx,
        "input": text,
        "output": output,
        "scored_text": output if ok else text,
        "violations_before": before_viol,
        "rules_fired": [r.name for r, _ in trace],
        "compliant": ok,               # False => FLAGGED for review, not emitted
        # Per-removal provenance verification (see realize()), plus summaries:
        # verified is True iff every attempted removal verified (vacuously
        # True if there were none); flagged is True if any removal was skipped
        # because no sentence verified.
        "provenance_verification_log": verification_log,
        "provenance_verified": all(not e["flagged"] for e in verification_log),
        "provenance_flagged": any(e["flagged"] for e in verification_log),
        # Whether either parse used the RuleParser fallback; callers should
        # aggregate these across a run.
        "input_parse_fallback": getattr(doc, "parse_fallback", False),
        "round_trip_parse_fallback": getattr(redoc, "parse_fallback", False),
        # Summary of the round-trip re-parse's graph, so a caller can check
        # whether a real edge was lost on re-parse.
        "redoc_edge_types": sorted(e[2] for e in redoc.graph.edges.values()),
        "redoc_n_nodes": len(redoc.graph.nodes),
        "redoc_n_edges": len(redoc.graph.edges),
        # Negation-filter drops for this call's input parse and round-trip
        # re-parse (Document.negation_drops).
        "input_negation_drops": getattr(doc, "negation_drops", []),
        "round_trip_negation_drops": getattr(redoc, "negation_drops", []),
    }