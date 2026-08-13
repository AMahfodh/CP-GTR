"""Pluggable NL -> Semantic Legal Graph parser.

A Document bundles sentences + typed graph + provenance (edge_id -> sentence
index), so subtractive/substitutive repairs can be realized back onto the right
sentence. Two parsers are provided:
  * RuleParser  -- keyword/regex; zero dependencies; runs anywhere.
  * LLMParser   -- schema-constrained; you supply a complete(prompt)->str client.
Both emit graphs typed over the same TG the rules use.
"""
from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from .graph import Graph, new_id


@dataclass
class Document:
    sentences: list
    graph: Graph
    prov: dict = field(default_factory=dict)     # edge_id -> sentence index


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------------------------------------------------------------- rule parser
class RuleParser:
    """Deterministic keyword parser. Good enough to run the pipeline end-to-end
    and to unit-test realization; NOT the parser you report accuracy for."""

    def parse(self, text) -> Document:
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


class LLMParser:
    """Schema-constrained LLM parser. `complete` is any callable str->str
    (wrap OpenAI, Anthropic, a local model, etc.). Falls back to RuleParser if
    the model returns unparseable output."""

    def __init__(self, complete, fallback=None):
        self.complete = complete
        self.fallback = fallback or RuleParser()

    def parse(self, text) -> Document:
        sents = split_sentences(text)
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
        try:
            raw = self.complete(_SCHEMA_PROMPT % numbered)
            data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        except Exception:
            return self.fallback.parse(text)

        TG_NODES = {"DataSubject", "ConsentProcess", "DataProcessing",
                    "ParentalConsent", "Withdrawal", "Operator", "ThirdParty",
                    "RetentionClause", "DurationIndefinite", "DurationBounded"}
        TG_EDGES = {"subjectOf", "grants", "requires", "permits",
                    "sellsTo", "hasDuration"}
        G, prov, idmap = Graph(), {}, {}
        for n in data.get("nodes", []):
            if n["type"] in TG_NODES:              # reject off-schema nodes
                idmap[n["id"]] = G.add_node(new_id("n"), n["type"])
        for e in data.get("edges", []):
            if e["type"] in TG_EDGES and e["src"] in idmap and e["tgt"] in idmap:
                eid = G.add_edge(new_id("e"), idmap[e["src"]], idmap[e["tgt"]], e["type"])
                prov[eid] = int(e.get("sentence", 0))
        return Document(sents, G, prov)