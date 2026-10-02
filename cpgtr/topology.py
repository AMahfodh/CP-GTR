"""Canonical-topology validator for rule and host graphs.

The single source of truth is schema/attach.yaml, which lists the admissible
(source type, edge type, target type) triples (`ATTACH`). A graph is canonical
if and only if every edge matches one of these triples exactly; direction
matters. Used by:
  - cpgtr/extract.py, to reject a candidate whose L, K or R contains a
    non-canonical edge (reported separately as a topology rejection);
  - cpgtr/parse.py, to measure whether a parsed host's backbone is canonical;
  - the analysis scripts, to measure topology rejections on cached candidates.

schema/attach.yaml is written in a narrow flow-style subset (one `[a, b, c]`
list entry per line under `attach:`) that the small parser here reads exactly,
so the core pipeline needs no third-party YAML dependency.
"""
from __future__ import annotations
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTACH_PATH = REPO_ROOT / "schema" / "attach.yaml"

_LIST_LINE = re.compile(r"^\s*-\s*\[([^\]]*)\]\s*(?:#.*)?$")


def load_attach(path=ATTACH_PATH):
    """Parse the `attach:` list of schema/attach.yaml.

    Returns a frozenset of (source_type, edge_type, target_type) tuples.
    Raises ValueError on a malformed entry or if no entries are found.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    in_attach = False
    triples = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("attach:"):
            in_attach = True
            continue
        if not in_attach:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        m = _LIST_LINE.match(line)
        if m is None:
            # A non-list, non-comment, non-blank line ends the `attach:` block
            # (the next top-level key, if any).
            if not line.startswith(" ") and not line.startswith("-"):
                break
            continue
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) != 3:
            raise ValueError(f"malformed attach entry in {path}: {line!r}")
        triples.append(tuple(parts))
    if not triples:
        raise ValueError(f"no attach entries found in {path}")
    return frozenset(triples)


ATTACH = load_attach()


def canonical_violations(graph):
    """List (edge_id, src_type, edge_type, tgt_type) for each edge not in ATTACH.

    An empty list means the graph is canonical.
    """
    violations = []
    for eid, (src, tgt, ety) in graph.edges.items():
        src_t = graph.nodes.get(src)
        tgt_t = graph.nodes.get(tgt)
        if (src_t, ety, tgt_t) not in ATTACH:
            violations.append((eid, src_t, ety, tgt_t))
    return violations


def is_canonical(graph):
    """True if every edge of `graph` is in ATTACH."""
    return not canonical_violations(graph)


class TopologyRejection(ValueError):
    """A candidate's L, K or R contains an edge not in ATTACH.

    Subclasses ValueError so generic parse-error handlers still catch it,
    while a caller that counts topology rejections separately can catch this
    class specifically instead of parsing message text.
    """


# One sentence per RuleParser branch (default configuration), each written to
# trigger the keywords that branch checks for (see RuleParser.parse in
# cpgtr/parse.py). derive_parser_emittable_set() runs the parser on these to
# derive its emittable triple set empirically.
_PARSER_PROBE_SENTENCES = [
    "We collect personal information from users who sign up.",  # backbone
    "We require parental consent before processing data from children.",
    "Users may withdraw their consent to processing at any time.",
    "We may sell your data to third parties for advertising purposes.",
    "Collected data is retained indefinitely with no expiration.",
    "Collected data is retained for 12 months after account closure.",
]


def derive_parser_emittable_set(parser=None):
    """Return the (source_type, edge_type, target_type) triples a parser emits.

    Runs `parser` (default: `RuleParser()` in its default configuration) on a
    probe battery covering every keyword branch and collects the edge triples
    it produces. ATTACH is intended to equal this set exactly;
    `assert_attach_matches_parser` checks that directly.
    """
    from .parse import RuleParser
    if parser is None:
        parser = RuleParser()  # enable_d7=False by default
    emitted = set()
    for sentence in _PARSER_PROBE_SENTENCES:
        doc = parser.parse(sentence)
        for (s, t, ty) in doc.graph.edges.values():
            emitted.add((doc.graph.nodes.get(s), ty, doc.graph.nodes.get(t)))
    return frozenset(emitted)


def assert_attach_matches_parser(parser=None):
    """Check that ATTACH equals the set of triples the parser can emit.

    Raises AssertionError naming the mismatched triples if the two sets
    differ; otherwise returns the verified set so a caller can log it.
    """
    emittable = derive_parser_emittable_set(parser)
    in_attach_not_emittable = ATTACH - emittable
    in_emittable_not_attach = emittable - ATTACH
    if in_attach_not_emittable or in_emittable_not_attach:
        raise AssertionError(
            "ATTACH != parser-emittable set.\n"
            f"  In ATTACH but the parser cannot produce it: {sorted(in_attach_not_emittable)}\n"
            f"  Parser can produce it but ATTACH is missing it: {sorted(in_emittable_not_attach)}"
        )
    return emittable


def spec_topology_violations(spec):
    """Non-canonical edges of a raw extraction spec.

    A candidate is topology-rejected if its L, K or R contains an edge not in
    ATTACH. The spec has no separate K edge list (K is the id-intersection of
    L and R, see `extract.json_to_rule`), so checking L and R covers K.
    Returns {"L": [...], "R": [...]}, each a list of `canonical_violations`
    (empty lists if canonical).
    """
    from .graph import Graph

    def _mk(part):
        g = Graph()
        for n in part.get("nodes", []):
            g.add_node(n["id"], n["type"])
        for e in part.get("edges", []):
            if e["src"] in g.nodes and e["tgt"] in g.nodes:
                g.add_edge(e["id"], e["src"], e["tgt"], e["type"])
        return g

    L = _mk(spec["L"])
    R = _mk(spec["R"])
    return {"L": canonical_violations(L), "R": canonical_violations(R)}
