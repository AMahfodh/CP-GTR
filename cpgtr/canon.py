"""kappa: deterministic canonicalization of graphs and rules.

kappa is a small set of DPO rules, one per approved entry in
schema/equivalences.yaml, run with the ordinary rewriting engine
(`cpgtr.rules.normalize`) rather than a bespoke graph-editing routine. Each
approved entry rewrites one non-canonical edge arrangement into its canonical
form, either by reversing the edge's direction or by relabeling its type (the
only two shapes supported; see `_kappa_rule_from_entry`). Adding a supported
entry to the file and rebuilding picks it up automatically.

`canonicalize_graph` normalizes host graphs, and `canonicalize_rule`
normalizes a candidate rule's L and R and recomputes K.
"""
from __future__ import annotations
import re
from pathlib import Path

from .graph import Graph, iso
from .rules import Rule, normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
EQUIVALENCES_PATH = REPO_ROOT / "schema" / "equivalences.yaml"


def load_approved_equivalences(path=EQUIVALENCES_PATH):
    """Parse the `approved:` list of {from, to} entries in equivalences.yaml.

    Parsing stops at the first `# ===` divider comment, so entries below it
    (reviewed but not approved) are never picked up. Like
    `cpgtr.topology.load_attach`, this is a small hand-rolled parser that
    avoids a YAML dependency. Returns a list of {"from": triple, "to": triple}.
    """
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"^approved:\s*$(.*?)^# =+", text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"could not find an `approved:` section ending in a "
                          f"'# ===' divider in {path}")
    section = m.group(1)
    entries = []
    for chunk in re.split(r"\n(?=\s*-\s*from:)", section):
        if not chunk.strip():
            continue
        fm = re.search(r"from:\s*\[([^\]]*)\]", chunk)
        tm = re.search(r"to:\s*\[([^\]]*)\]", chunk)
        if not fm or not tm:
            continue
        frm = tuple(p.strip() for p in fm.group(1).split(","))
        to = tuple(p.strip() for p in tm.group(1).split(","))
        if len(frm) != 3 or len(to) != 3:
            raise ValueError(f"malformed equivalence entry in {path}: {chunk!r}")
        entries.append({"from": frm, "to": to})
    return entries


def _kappa_rule_from_entry(entry, index):
    """Build the DPO rule for one approved equivalences.yaml entry.

    Two shapes are supported; anything else (a node-type substitution, or a
    combination of the two) raises ValueError.

    1. Direction flip (same edge type, endpoints reversed): L matches
       src_t --ety--> tgt_t and R has the same two preserved nodes with the
       edge reversed, so `to` == [tgt_t, ety, src_t].
    2. Edge-type relabel (same direction and endpoints, different edge type):
       R has the same two preserved nodes with an edge of a different type in
       the same direction, so `to` == [src_t, ety2, tgt_t] with ety2 != ety.
       The relabeled edge is genuinely a different edge, so it is modelled as
       delete-old/create-new, and the content-based re-keying in
       `canonicalize_rule` (which matches on source, target and type) treats
       it that way too.
    """
    src_t, ety, tgt_t = entry["from"]
    src_t2, ety2, tgt_t2 = entry["to"]

    is_direction_flip = (ety == ety2 and src_t2 == tgt_t and tgt_t2 == src_t)
    is_relabel = (ety != ety2 and src_t2 == src_t and tgt_t2 == tgt_t)
    if not (is_direction_flip or is_relabel):
        raise ValueError(
            f"equivalences.yaml entry {entry} is neither a pure same-edge-"
            f"type direction flip (`to` == [tgt_t, ety, src_t]) nor a same-"
            f"direction edge-type relabel (`to` == [src_t, ety2, tgt_t]) -- "
            f"this builder only supports those two shapes"
        )

    L = Graph()
    L.add_node("a", src_t)
    L.add_node("b", tgt_t)
    L.add_edge("e_old", "a", "b", ety)

    R = Graph()
    R.add_node("a", src_t)
    R.add_node("b", tgt_t)
    if is_direction_flip:
        R.add_edge("e_new", "b", "a", ety)          # reversed: R's edge is (b -> a)
    else:
        R.add_edge("e_new", "a", "b", ety2)  # relabeled: new type

    return Rule(
        name=f"kappa{index}_{src_t}_{ety}_{tgt_t}",
        L=L, K_nodes={"a", "b"}, K_edges=set(), R=R, nacs=[],
        phi=lambda A, J: True, template="", source=None, confidence=1.0,
    )


def build_kappa_rules(path=EQUIVALENCES_PATH):
    """Build one kappa rule per approved entry of the equivalences file."""
    return [_kappa_rule_from_entry(e, i)
            for i, e in enumerate(load_approved_equivalences(path))]


KAPPA_RULES = build_kappa_rules()


def canonicalize_graph(G):
    """Rewrite `G` with the kappa rules to a normal form.

    kappa is purely structural, so no (A, J) context is passed and its rules
    (phi always true) are unconditionally active.
    """
    return normalize(KAPPA_RULES, G, ctx=None)


def is_idempotent(G):
    """True if canonicalizing an already-canonical `G` leaves it unchanged.

    Compared up to isomorphism rather than id equality, because `apply_rule`
    mints fresh edge ids on every firing.
    """
    once = canonicalize_graph(G)
    twice = canonicalize_graph(once)
    return iso(once, twice) is not None


def _rekey_matching_edges(new_L, new_R):
    """Rename edges of `new_R` to the ids of their content-equal edges in `new_L`.

    kappa mints a fresh edge id each time it fires, and `canonicalize_rule`
    canonicalizes L and R separately. A preserved edge (same endpoints and
    type, rewritten identically on both sides) therefore comes back under two
    unrelated ids. Node ids are untouched by kappa and stay aligned, so only
    edge ids need reconciling.

    Without this, the id-based recomputation of K would read each such edge as
    an unrelated deletion in L plus an unrelated creation in R. A purely
    additive rule with no NAC would then appear to delete something, wrongly
    pass condition (R1), and could refire on its own output indefinitely.

    Returns a new Graph equal to `new_R` with matched edges renamed to their
    `new_L` counterparts' ids; unmatched edges (genuine creations) keep their
    ids. Returns `new_R` itself if nothing needs renaming.
    """
    from .graph import Graph

    def content(g, e):
        s, t, ty = g.edges[e]
        return (s, t, ty)

    by_content = {}
    for e in new_L.edges:
        by_content.setdefault(content(new_L, e), []).append(e)

    used_l = set()
    rekey = {}
    for e in new_R.edges:
        candidates = [le for le in by_content.get(content(new_R, e), []) if le not in used_l]
        if candidates:
            le = candidates[0]
            used_l.add(le)
            if le != e:
                rekey[e] = le

    if not rekey:
        return new_R
    renamed = Graph()
    renamed.nodes = dict(new_R.nodes)
    for e, (s, t, ty) in new_R.edges.items():
        renamed.add_edge(rekey.get(e, e), s, t, ty)
    return renamed


def canonicalize_rule(rule):
    """Return a copy of `rule` with kappa applied to its L and R.

    kappa is applied to L and R independently. R's edges are first re-keyed to
    L's ids where they match by content (see `_rekey_matching_edges`), so an
    edge rewritten identically on both sides is recognised as preserved. K is
    then recomputed from the canonicalized L and R by id intersection with
    type consistency, as `extract.json_to_rule` does, because kappa can change
    which edge ids exist. NACs are left untouched, and the rank `rho` is reset
    to None.
    """
    new_L = canonicalize_graph(rule.L)
    new_R = _rekey_matching_edges(new_L, canonicalize_graph(rule.R))

    K_nodes = {n for n in (set(new_L.nodes) & set(new_R.nodes))
               if new_L.nodes[n] == new_R.nodes[n]}
    K_edges = {e for e in (set(new_L.edges) & set(new_R.edges))
               if new_L.edges[e] == new_R.edges[e]}

    return Rule(
        name=rule.name, L=new_L, K_nodes=K_nodes, K_edges=K_edges, R=new_R,
        nacs=rule.nacs, phi=rule.phi, rho=None, template=rule.template,
        source=rule.source, confidence=rule.confidence,
    )
