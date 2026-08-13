"""SLG* -> compliant text, plus the 'compliant or flagged' round-trip check.

realize() edits the source sentences according to the derivation trace and the
rule templates; round_trip() re-parses the output and re-runs the normal-form
check, so an output is emitted only if it re-parses to a normal form.
"""
from __future__ import annotations
from .graph import iso
from .rules import one_step


def normalize_trace(rules, G, ctx, limit=5000):
    """Deterministic normal form + list of (rule, match) actually fired."""
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


def realize(doc, trace):
    """Apply text edits implied by the fired rules. Returns new text.

    additive     -> append the rule's template sentence
    subtractive  -> delete the source sentence(s) of the removed edge(s),
                    optionally append a template stating the now-absent fact
    substitutive -> replace the source sentence of the removed edge with template
    """
    sents = list(doc.sentences)
    remove, append = set(), []
    for rule, m in trace:
        mod = rule.modality()
        if mod == "additive":
            if rule.template:
                append.append(rule.template)
        else:
            for le in rule.del_edges():
                src_edge = m["edges"].get(le)
                idx = doc.prov.get(src_edge)
                if idx is not None:
                    if mod == "substitutive" and rule.template:
                        sents[idx] = rule.template
                    else:                       # subtractive
                        remove.add(idx)
            if mod == "subtractive" and rule.template:
                append.append(rule.template)
    kept = [s for i, s in enumerate(sents) if i not in remove]
    return " ".join(kept + append)


def round_trip(parser, rules, ctx, output_text):
    """'compliant or flagged': re-parse output; True iff it is a normal form."""
    redoc = parser.parse(output_text)
    residual = one_step(rules, redoc.graph, ctx)
    return len(residual) == 0, redoc


def process_document(parser, rules, ctx, text):
    """Full deployment lane: parse -> gate+normalize -> realize -> re-check."""
    doc = parser.parse(text)
    before_viol = [r.name for (r, _, _) in one_step(rules, doc.graph, ctx)]
    Gstar, trace = normalize_trace(rules, doc.graph, ctx)
    output = realize(doc, trace)
    ok, redoc = round_trip(parser, rules, ctx, output)
    return {
        "context": ctx,
        "input": text,
        "output": output,
        "violations_before": before_viol,
        "rules_fired": [r.name for r, _ in trace],
        "compliant": ok,               # False => FLAGGED for review, not emitted
    }