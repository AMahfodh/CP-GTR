"""Natural-language policy text -> Semantic Legal Graph (SLG) parser.

Role in the pipeline: the realization stage (`realize.process_document`) parses
a privacy-policy passage into a typed graph, rewrites it with the certified
rules, and re-parses the edited text for the round-trip check. The parser also
produces the graphs scored in the parser evaluation (`eval_parser`).

A `Document` bundles the sentences, a typed `Graph` over the fixed node/edge
vocabulary, and provenance (edge id -> index of the sentence that produced it),
so repairs can be applied back to the right sentence. Two parsers share the
same output type:

  * `RuleParser` - deterministic keyword/regex parser; no dependencies; always
    available. Intended for offline runs and tests, not for reported parser
    accuracy.
  * `LLMParser`  - schema-constrained LLM parser; takes a ``complete(prompt)
    -> str`` client (see `llm.make_complete`). Optionally states the canonical
    edge topology in the prompt and repairs topology violations. Empty or
    unparseable responses are retried and, if all attempts fail, fall back to
    `RuleParser`; the fallback is recorded on the returned `Document` and in
    ``logs/parse_fallbacks.jsonl`` so it is never silent.

Negation filter: the LLM parser does not reliably distinguish "we sell X" from
"we do not sell X". After parsing, a deterministic filter drops an edge when
its source sentence contains, in one clause, both a predicate word for that
edge type and a negation marker (`EDGE_TYPE_PREDICATES`, `NEGATION_MARKERS`).
Drops are counted and logged to ``logs/negation_filter_drops.jsonl``. The
filter is a crude, auditable heuristic, not a semantic negation analysis.
"""
from __future__ import annotations
import json
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from .graph import Graph, new_id

# max_tokens passed to complete() for LLMParser's initial and repair calls.
# Reasoning-tuned models (e.g. gpt-oss-120b) spend part of the same max_tokens
# budget on a hidden reasoning trace, so a small budget can leave the visible
# JSON truncated or empty; long passages need a generous ceiling. Retrying
# does not help when the budget itself is too small, so this is set high.
PARSE_MAX_TOKENS = 8192

# Number of attempts for LLMParser's initial call (empty, None or unparseable
# responses are retried). The topology-repair loop has its own counter. If
# every attempt fails, the RuleParser fallback is used and the event is
# recorded on the returned Document and in the fallback log.
MAX_PARSE_ATTEMPTS = 3

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSE_FALLBACK_LOG = REPO_ROOT / "logs" / "parse_fallbacks.jsonl"
_fallback_log_lock = threading.Lock()


def _log_parse_fallback(text, reason, attempts, enforce_topology, log_path=PARSE_FALLBACK_LOG):
    """Append one JSONL record each time LLMParser.parse() falls back to
    RuleParser, so a fallback is always visible in the run's artifacts.

    Thread-safe (append under a lock), since LLMParser may be driven
    concurrently."""
    log_path.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "attempts": attempts,
        "enforce_topology": enforce_topology,
        "text": text,
    }
    with _fallback_log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------ negation filter
# LLMParser can read a negated sentence ("we do NOT sell X") as asserting the
# relationship it denies. Rather than change the model's extraction, a
# deterministic post-processing filter drops edges whose source sentence
# negates them (see the module docstring).

NEGATION_MARKERS = ("do not", "does not", "will not", "never", "no longer",
                     "don't", "doesn't", "won't", "cannot", "can't",
                     "shall not", "will never")

# Small, schema-scoped predicate vocabulary: for each edge type, the verbs a
# sentence asserting that relationship would plausibly use. subjectOf is
# purely structural (a data subject being party to a consent process) and has
# no verb to negate, so it is excluded.
EDGE_TYPE_PREDICATES = {
    "grants": ("collect", "collects", "collecting", "collected",
               "use", "uses", "using", "used",
               "process", "processes", "processing", "processed", "consent"),
    "requires": ("require", "requires", "requiring", "required", "must", "need", "needs"),
    "permits": ("permit", "permits", "permitting", "allow", "allows", "allowing",
                "may", "withdraw", "withdrawn"),
    "sellsTo": ("sell", "sells", "selling", "sold", "sale",
                "share", "shares", "sharing", "shared",
                "rent", "rents", "renting", "rented",
                "disclose", "discloses", "disclosing", "disclosed"),
    "hasDuration": ("retain", "retains", "retaining", "retained",
                     "keep", "keeps", "keeping", "kept",
                     "store", "stores", "storing", "stored"),
}

NEGATION_FILTER_LOG = REPO_ROOT / "logs" / "negation_filter_drops.jsonl"
_negation_log_lock = threading.Lock()
# Process-wide count of edges dropped by the negation filter. Reset to 0
# before a measured run if the process is reused.
negation_filter_drop_count = 0


def _split_clauses(sentence):
    """Split a sentence into clauses at semicolons, commas, "and" and "but".

    This keeps a negation in one clause of a long sentence from suppressing an
    unrelated edge whose predicate is in another clause. Intentionally crude
    and auditable, not a real clause parser."""
    return [p.strip() for p in re.split(r";|,|\band\b|\bbut\b", sentence, flags=re.IGNORECASE)
            if p.strip()]


def _edge_negated_in_sentence(sentence, edge_type):
    """True if some clause of `sentence` contains both one of `edge_type`'s
    predicate words and a negation marker, i.e. the sentence denies this
    relationship rather than asserting it."""
    predicates = EDGE_TYPE_PREDICATES.get(edge_type)
    if not predicates or not sentence:
        return False
    for clause in _split_clauses(sentence):
        low = clause.lower()
        if any(p in low for p in predicates) and any(n in low for n in NEGATION_MARKERS):
            return True
    return False


def _log_negation_drop(sentence, edge_type, src_type, tgt_type, log_path=NEGATION_FILTER_LOG):
    """Count a negation-filter drop and append it to the JSONL drop log."""
    global negation_filter_drop_count
    negation_filter_drop_count += 1
    log_path.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "edge_type": edge_type, "src_type": src_type, "tgt_type": tgt_type,
        "sentence": sentence,
    }
    with _negation_log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class Document:
    """A parsed passage: its sentences, SLG, and edge provenance.

    Attributes:
        sentences: the passage split into sentences.
        graph: typed `Graph` over the fixed node/edge vocabulary.
        prov: edge id -> index of the sentence the edge was derived from.
        parse_fallback: True if this is RuleParser output substituted after
            LLMParser exhausted its attempts (False otherwise).
        parse_fallback_reason: description of the last failure, if any.
        parse_attempts: number of initial LLM attempts made (1 by default).
        negation_drops: edges removed by the negation filter while building
            this document (including topology-repair rebuilds).
    """
    sentences: list
    graph: Graph
    prov: dict = field(default_factory=dict)     # edge_id -> sentence index
    parse_fallback: bool = False
    parse_fallback_reason: str | None = None
    parse_attempts: int = 1
    negation_drops: list = field(default_factory=list)


def split_sentences(text):
    """Split text into sentences at ., ! or ? followed by whitespace."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------------------------------------------------------------- rule parser
class RuleParser:
    """Deterministic keyword/regex parser.

    Detects data-processing/consent, parental consent, withdrawal, third-party
    sale and retention (indefinite or bounded duration) from keywords. Good
    enough to run the pipeline end to end and to test realization; it is not
    the parser to report accuracy for.

    Args:
        enable_d7: if True, also emit ``Operator --permits--> DataProcessing``
            when a sentence says an operator/service processes, collects or
            uses personal data. Off by default. The keyword trigger fires on
            ordinary "we collect ..." phrasing where the hand-labelled gold
            parses expect no Operator node, so enabling it lowers parser
            precision on that gold set.
    """

    def __init__(self, enable_d7: bool = False):
        self.enable_d7 = enable_d7

    def parse(self, text) -> Document:
        """Parse `text` into a `Document` using keyword rules."""
        sents = split_sentences(text)
        G = Graph()
        prov = {}
        # a consent process backbone appears whenever data/consent is mentioned
        ds = cp = dp = None

        def ensure_backbone(idx):
            nonlocal ds, cp, dp
            if cp is None:
                ds = G.add_node(new_id("ds"), "DataSubject")
                cp = G.add_node(new_id("cp"), "ConsentProcess")
                dp = G.add_node(new_id("dp"), "DataProcessing")
                e1 = G.add_edge(new_id("e"), ds, cp, "subjectOf")
                e2 = G.add_edge(new_id("e"), cp, dp, "grants")
                prov[e1] = idx
                prov[e2] = idx

        for i, s in enumerate(sents):
            low = s.lower()
            if any(k in low for k in ("collect", "personal information",
                                      "consent", "sign up", "process")):
                ensure_backbone(i)
            if "parental" in low or "holder of parental responsibility" in low:
                ensure_backbone(i)
                pc = G.add_node(new_id("pc"), "ParentalConsent")
                e = G.add_edge(new_id("e"), cp, pc, "requires")
                prov[e] = i
            if "withdraw" in low:
                ensure_backbone(i)
                w = G.add_node(new_id("w"), "Withdrawal")
                e = G.add_edge(new_id("e"), cp, w, "permits")
                prov[e] = i
            if "sell" in low and ("third part" in low or "advertising" in low
                                  or "partners" in low):
                op = G.add_node(new_id("op"), "Operator")
                tp = G.add_node(new_id("tp"), "ThirdParty")
                e = G.add_edge(new_id("e"), op, tp, "sellsTo")
                prov[e] = i
            # Optional: Operator--permits-->DataProcessing when the text states
            # that an operator/service processes, collects or uses personal
            # data. Rule extraction frequently produces this arrangement, so
            # this gives the host graph a counterpart for it. The gold parses
            # reserve Operator for third-party-sale contexts, so this trigger
            # over-fires relative to gold (see the class docstring).
            if self.enable_d7 and any(
                t in low for t in ("we ", "our service", "our app",
                                    "the operator", "the company",
                                    "this service", "this app")
            ) and any(
                v in low for v in ("process", "collect", "use your",
                                    "uses your", "using your")
            ):
                ensure_backbone(i)
                op2 = G.add_node(new_id("op"), "Operator")
                e = G.add_edge(new_id("e"), op2, dp, "permits")
                prov[e] = i
            if "retain" in low or "retention" in low or "stored" in low:
                rc = G.add_node(new_id("rc"), "RetentionClause")
                if any(k in low for k in ("indefinit", "as long as", "no expir")):
                    di = G.add_node(new_id("di"), "DurationIndefinite")
                    e = G.add_edge(new_id("e"), rc, di, "hasDuration")
                    prov[e] = i
                elif re.search(r"\b\d+\s*(month|year|day)", low):
                    db = G.add_node(new_id("db"), "DurationBounded")
                    e = G.add_edge(new_id("e"), rc, db, "hasDuration")
                    prov[e] = i
        return Document(sents, G, prov)


# ---------------------------------------------------------------- LLM parser
_SCHEMA_PROMPT = """You are a legal-text parser. Extract a typed Semantic Legal
Graph from the passage. Use ONLY these node types:
  DataSubject, ConsentProcess, DataProcessing, ParentalConsent, Withdrawal,
  Operator, ThirdParty, RetentionClause, DurationIndefinite, DurationBounded
and ONLY these edge types:
  subjectOf, grants, requires, permits, sellsTo, hasDuration
Return STRICT JSON: {"nodes":[{"id","type"}],
"edges":[{"id","src","tgt","type","sentence"}]} where "sentence" is the 0-based
index of the source sentence. No prose, JSON only.

SENTENCES:
%s
"""

def _canonical_schema_block_for_parser():
    """Render the canonical (source, edge, target) triples as prompt text.

    Generated from `cpgtr.topology.ATTACH` (the table the validator uses) so
    the prompt cannot drift from the check. Used only when LLMParser is
    constructed with ``enforce_topology=True``."""
    from .topology import ATTACH
    lines = ["Every edge you emit must be one of these EXACT (source type, "
             "edge type, target type) triples -- no other combination is "
             "admissible, even if the node/edge types are individually "
             "allowed:"]
    for (s, e, t) in sorted(ATTACH):
        lines.append(f"  {s} --{e}--> {t}")
    return "\n".join(lines)


_ATTACH_WORKED_EXAMPLES_FOR_PARSER = """Worked examples, one per admissible triple
(each shows a minimal correct nodes/edges fragment for that triple alone):

DataSubject--subjectOf-->ConsentProcess:
  {"nodes":[{"id":"ds1","type":"DataSubject"},{"id":"cp1","type":"ConsentProcess"}],
   "edges":[{"id":"e1","src":"ds1","tgt":"cp1","type":"subjectOf","sentence":0}]}

ConsentProcess--grants-->DataProcessing:
  {"nodes":[{"id":"cp1","type":"ConsentProcess"},{"id":"dp1","type":"DataProcessing"}],
   "edges":[{"id":"e1","src":"cp1","tgt":"dp1","type":"grants","sentence":0}]}

ConsentProcess--permits-->Withdrawal:
  {"nodes":[{"id":"cp1","type":"ConsentProcess"},{"id":"w1","type":"Withdrawal"}],
   "edges":[{"id":"e1","src":"cp1","tgt":"w1","type":"permits","sentence":0}]}

ConsentProcess--requires-->ParentalConsent:
  {"nodes":[{"id":"cp1","type":"ConsentProcess"},{"id":"pc1","type":"ParentalConsent"}],
   "edges":[{"id":"e1","src":"cp1","tgt":"pc1","type":"requires","sentence":0}]}

RetentionClause--hasDuration-->DurationBounded:
  {"nodes":[{"id":"rc1","type":"RetentionClause"},{"id":"db1","type":"DurationBounded"}],
   "edges":[{"id":"e1","src":"rc1","tgt":"db1","type":"hasDuration","sentence":0}]}

RetentionClause--hasDuration-->DurationIndefinite:
  {"nodes":[{"id":"rc1","type":"RetentionClause"},{"id":"di1","type":"DurationIndefinite"}],
   "edges":[{"id":"e1","src":"rc1","tgt":"di1","type":"hasDuration","sentence":0}]}

Operator--sellsTo-->ThirdParty:
  {"nodes":[{"id":"op1","type":"Operator"},{"id":"tp1","type":"ThirdParty"}],
   "edges":[{"id":"e1","src":"op1","tgt":"tp1","type":"sellsTo","sentence":0}]}
"""

# Optional prompt addition, enabled by LLMParser(..., negative_example=True)
# (only together with enforce_topology): tells the model to return an empty
# graph when the passage supports no admissible triple, with one negative
# worked example. Off by default so the two prompt variants can be compared
# from the same code.
_NEGATIVE_EXAMPLE_FOR_PARSER = """If the passage does NOT clearly state or
imply any of the admissible triples above, do not force one into existence --
emit an EMPTY graph instead. Only emit a node or edge the text actually
supports; do not infer the canonical backbone just because the passage is
generally about a privacy policy, cookies, or data in the abstract.

Negative worked example (a passage with no admissible triple at all):
  PASSAGE: "This notice explains our cookie categories and how to manage
  browser preferences."
  {"nodes":[], "edges":[]}
"""

_PARSE_REPAIR_PROMPT = """Your previous JSON output for this passage was
rejected for a structural reason:

ERROR: %s

YOUR PREVIOUS OUTPUT:
%s

Fix ONLY this specific error and return the corrected, COMPLETE JSON object
(same schema as before: {"nodes":[...], "edges":[...]}, JSON only, no
commentary). Every edge must still be one of the admissible triples.

SENTENCES:
%s
"""


def _topology_violation_message_for_parser(violations, attach):
    """Build the repair-prompt feedback for a parsed graph with non-canonical
    edges (parser counterpart of `extract._topology_rejection_message`).

    Names the offending (source, edge, target) triples and lists every
    admissible triple."""
    lines = ["Non-canonical edge(s) -- every edge must be one of the "
             "admissible (source, edge, target) triples listed below."]
    for (eid, src_t, ety, tgt_t) in violations:
        lines.append(f"  offending: {src_t} --{ety}--> {tgt_t} (edge id {eid!r})")
    lines.append("Admissible triples (use ONE of these instead):")
    for (s, e, t) in sorted(attach):
        lines.append(f"  {s} --{e}--> {t}")
    return "\n".join(lines)


class LLMParser:
    """Schema-constrained LLM parser.

    The model is prompted with the numbered sentences and the fixed node/edge
    vocabulary and returns JSON nodes and edges, each edge tagged with its
    source-sentence index. Off-schema nodes and edges are dropped, and edges
    denied by their own sentence are removed by the negation filter (see the
    module docstring).

    Args:
        complete: any callable ``complete(prompt, max_tokens=...) -> str``
            (e.g. from `llm.make_complete`).
        fallback: parser used when every attempt fails (default `RuleParser`).
        enforce_topology: if True, the prompt states the canonical edge
            topology (`cpgtr.topology.ATTACH`) with one worked example per
            admissible triple, and any resulting non-canonical edge is fed
            back to the model for repair, up to `topology_repair_attempts`
            times. Edges still non-canonical after that are dropped from the
            graph (rather than falling back to `RuleParser`, which covers far
            fewer constructs). Default False.
        topology_repair_attempts: repair rounds when `enforce_topology` is set.
        negative_example: if True (and `enforce_topology`), add an
            instruction and a negative example telling the model to emit an
            empty graph when the text supports no admissible triple.

    Failure handling: an empty, None or unparseable response is retried up to
    `MAX_PARSE_ATTEMPTS` times. If every attempt fails, `fallback` is used, the
    returned `Document` has ``parse_fallback=True`` with the reason and attempt
    count, and the event is appended to ``logs/parse_fallbacks.jsonl``.
    """

    def __init__(self, complete, fallback=None, enforce_topology=False,
                 topology_repair_attempts=3, negative_example=False):
        self.complete = complete
        self.fallback = fallback or RuleParser()
        self.enforce_topology = enforce_topology
        self.topology_repair_attempts = topology_repair_attempts
        self.negative_example = negative_example

    def _build_graph(self, data, sents=()):
        """Build (graph, prov, drops) from the model's parsed JSON `data`.

        Off-schema node types, off-schema edge types and edges with unknown
        endpoints are dropped. If `sents` (the sentence list) is given, an edge
        whose own source sentence denies the relationship (see
        `_edge_negated_in_sentence`) is also dropped, logged and counted;
        `drops` lists those edges. With the default ``sents=()`` the negation
        filter is disabled."""
        TG_NODES = {"DataSubject", "ConsentProcess", "DataProcessing",
                    "ParentalConsent", "Withdrawal", "Operator", "ThirdParty",
                    "RetentionClause", "DurationIndefinite", "DurationBounded"}
        TG_EDGES = {"subjectOf", "grants", "requires", "permits",
                    "sellsTo", "hasDuration"}
        G, prov, idmap = Graph(), {}, {}
        drops = []
        for n in data.get("nodes", []):
            if n["type"] in TG_NODES:              # reject off-schema nodes
                idmap[n["id"]] = G.add_node(new_id("n"), n["type"])
        for e in data.get("edges", []):
            if e["type"] in TG_EDGES and e["src"] in idmap and e["tgt"] in idmap:
                sent_idx = int(e.get("sentence", 0))
                sentence = sents[sent_idx] if sents and 0 <= sent_idx < len(sents) else None
                if sentence is not None and _edge_negated_in_sentence(sentence, e["type"]):
                    src_type, tgt_type = G.nodes[idmap[e["src"]]], G.nodes[idmap[e["tgt"]]]
                    _log_negation_drop(sentence, e["type"], src_type, tgt_type)
                    drops.append({"edge_type": e["type"], "src_type": src_type,
                                  "tgt_type": tgt_type, "sentence": sentence})
                    continue
                eid = G.add_edge(new_id("e"), idmap[e["src"]], idmap[e["tgt"]], e["type"])
                prov[eid] = sent_idx
        return G, prov, drops

    def _attempt_json(self, prompt):
        """Make one complete() call and parse its JSON object.

        Returns ``(data, None)`` on success, or ``(None, reason)`` with a short
        description of the failure."""
        try:
            raw = self.complete(prompt, max_tokens=PARSE_MAX_TOKENS)
        except Exception as e:
            return None, f"complete() raised {type(e).__name__}: {e}"
        if not raw:
            # A reasoning model's hidden trace can consume the entire
            # max_tokens budget, leaving the message content None without the
            # SDK raising an exception.
            return None, "empty/None response from complete() (max_tokens likely fully consumed by hidden reasoning trace)"
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None, "no JSON object found in response"
        try:
            return json.loads(m.group(0)), None
        except Exception as e:
            return None, f"invalid JSON: {type(e).__name__}: {e}"

    def parse(self, text) -> Document:
        """Parse `text` into a `Document` (see the class docstring)."""
        sents = split_sentences(text)
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
        if self.enforce_topology:
            prompt = (_SCHEMA_PROMPT % numbered) + "\n" + \
                     _canonical_schema_block_for_parser() + "\n\n" + \
                     _ATTACH_WORKED_EXAMPLES_FOR_PARSER
            if self.negative_example:
                prompt += "\n" + _NEGATIVE_EXAMPLE_FOR_PARSER
        else:
            prompt = _SCHEMA_PROMPT % numbered

        data = reason = None
        json_attempts = 0
        for json_attempts in range(1, MAX_PARSE_ATTEMPTS + 1):
            data, reason = self._attempt_json(prompt)
            if data is not None:
                break
        if data is None:
            doc = self.fallback.parse(text)
            doc.parse_fallback = True
            doc.parse_fallback_reason = reason
            doc.parse_attempts = json_attempts
            _log_parse_fallback(text, reason, json_attempts, self.enforce_topology)
            return doc

        G, prov, drops = self._build_graph(data, sents)

        if self.enforce_topology:
            from .topology import canonical_violations, ATTACH
            repair_attempts = 0
            violations = canonical_violations(G)
            while violations and repair_attempts < self.topology_repair_attempts:
                repair_attempts += 1
                try:
                    err = _topology_violation_message_for_parser(violations, ATTACH)
                    raw = self.complete(_PARSE_REPAIR_PROMPT % (err, json.dumps(data), numbered),
                                        max_tokens=PARSE_MAX_TOKENS)
                    data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
                except Exception:
                    break
                G, prov, new_drops = self._build_graph(data, sents)
                drops.extend(new_drops)
                violations = canonical_violations(G)
            if violations:
                # Repairs exhausted: drop only the still-violating edges
                # (as for an off-schema edge), not the whole graph.
                bad_ids = {eid for (eid, _s, _ty, _t) in violations}
                for eid in bad_ids:
                    G.edges.pop(eid, None)
                    prov.pop(eid, None)

        return Document(sents, G, prov, parse_attempts=json_attempts, negation_drops=drops)