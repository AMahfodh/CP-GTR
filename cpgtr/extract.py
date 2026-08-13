"""Stage-1 LLM rule extractor: provisions.json -> candidate Rule objects.

This is the component that replaces library.synthesize(). Supply a
complete(prompt)->str client; the emitted JSON is parsed into Rule spans that
then go UNCHANGED into certify.admit(). Off-schema symbols are rejected here,
matching the paper's 'no latitude over the symbol set'.

Performance/reliability design (2026-08-07, after a 571-call real run showed
p50 latency 10.8s but p90 605s / max 1255s with no correlation to prompt
size -- see cpgtr/llm.py's module docstring for the full diagnosis):
  - Concurrent (ThreadPoolExecutor) rather than sequential, so a handful of
    slow outlier calls don't serially block the whole run.
  - Batched (BATCH_SIZE provisions per call, one JSON array back) to cut
    total call count, with automatic fallback to per-provision calls for any
    batch whose response isn't a valid, correctly-sized array.
  - Incrementally cached, keyed by a content hash of
    (model, temperature, max_tokens, provision), and written to cache_path
    as JSONL (one line per successful extraction, flushed immediately) --
    not a single JSON blob written at the end. A crashed or interrupted run
    never loses completed work; re-running with an unchanged cache_path and
    the same inputs makes zero further LLM calls for anything already cached.
  - Prompts send only the single provision's text, capped short (see
    PROMPT_SPAN_CHARS) -- not neighbouring provisions or the full statute.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .graph import Graph, new_id
from .rules import Rule

TG_NODES = {"DataSubject", "ConsentProcess", "DataProcessing", "ParentalConsent",
            "Withdrawal", "Operator", "ThirdParty", "RetentionClause",
            "DurationIndefinite", "DurationBounded"}
TG_EDGES = {"subjectOf", "grants", "requires", "permits", "sellsTo", "hasDuration"}

# Sent in the prompt (not the same as segment_corpus.py's 2000-char storage
# cap): measured against real extraction, a full statutory span is far more
# context than the schema-constrained extraction task needs, and latency did
# not correlate with prompt size in the 571-call benchmark -- this is capped
# for cost/consistency, not because it was identified as the latency cause.
PROMPT_SPAN_CHARS = 800

BATCH_SIZE = 5
MAX_WORKERS = 8
# Measured from the 571-call benchmark: completion_tokens p90=638, max=878
# for SINGLE-provision responses. Scaled per batch item plus headroom for
# JSON array wrapping, not guessed.
MAX_TOKENS_PER_ITEM = 1024

_SCHEMA_NOTE = """An id used in both L and R must denote the exact same node
type (or the exact same src/tgt/type for an edge) in both -- do not reuse an
id for two structurally different elements. Preserve an id across L and R iff
the element is NOT deleted."""

_ITEM_SCHEMA = """{"name": str,
 "L": {"nodes":[{"id","type"}], "edges":[{"id","src","tgt","type"}]},
 "R": {"nodes":[...], "edges":[...]},
 "nac": {"nodes":[...], "edges":[...]},        // optional, superset of L
 "age_op": "<"|"<="|"="|">="|">"|null, "age": int|null,
 "jurisdiction": "US"|"EU"|"UK"|"US-CA"|null,
 "template": str,
 "confidence": float}                          // your own confidence in [0,1] that this
                                                 // rule correctly and completely captures
                                                 // the provision's requirement"""

_EXTRACT_PROMPT = """Convert this statutory provision into a DPO repair rule as
STRICT JSON. A rule has L (pattern to find), R (pattern after repair), K
(preserved elements: ids present in BOTH L and R), NAC (forbidden context), a
context predicate, and a natural-language template for the repair.

Allowed node types: %s
Allowed edge types: %s

JSON schema:
%s
%s JSON only.

PROVISION (%s, tau=%s):
%s
"""

_BATCH_EXTRACT_PROMPT = """Convert EACH of the following %d statutory provisions
into a DPO repair rule as STRICT JSON. Return a JSON ARRAY with EXACTLY %d
objects, one per provision, in the same order. Each object has this schema:

%s
%s

Allowed node types: %s
Allowed edge types: %s

Return ONLY the JSON array, no commentary.

PROVISIONS:
%s
"""

_REPAIR_PROMPT = """Your previous JSON output for this statutory provision was
rejected for a structural reason:

ERROR: %s

YOUR PREVIOUS OUTPUT:
%s

Fix ONLY this specific error and return the corrected JSON object (same
schema, JSON only, no commentary). %s

ORIGINAL PROVISION (%s, tau=%s):
%s
"""


def _mk_graph(spec):
    """Build a Graph from an LLM-emitted {"nodes","edges"} spec, rejecting
    off-schema types AND structurally malformed edges.

    The edge-endpoint check (src/tgt must reference a node actually declared
    in this same spec) is not optional bookkeeping: a real extraction run
    against Llama-3.3-70B-T produced a rule whose R declared an edge to a
    node id that was never listed under "nodes" (a genuine model mistake,
    not a schema-vocabulary violation -- the node TYPE would have been fine,
    the id just didn't exist). Without this check, _mk_graph builds a Graph
    with a dangling edge reference, and the resulting malformed Rule doesn't
    fail until deep inside strongly_joinable's isomorphism search (a bare
    KeyError from graph.iso's edges_ok, not a clean, attributable parse
    error) -- turning one bad extraction into a crash of the entire
    admission run instead of a single skipped candidate.
    """
    G = Graph()
    for n in spec.get("nodes", []):
        if n["type"] not in TG_NODES:
            raise ValueError(f"off-schema node {n['type']}")
        G.add_node(n["id"], n["type"])
    for e in spec.get("edges", []):
        if e["type"] not in TG_EDGES:
            raise ValueError(f"off-schema edge {e['type']}")
        if e["src"] not in G.nodes or e["tgt"] not in G.nodes:
            raise ValueError(
                f"edge {e.get('id', '?')!r} references an undeclared node "
                f"(src={e['src']!r}, tgt={e['tgt']!r}, "
                f"declared nodes={sorted(G.nodes)})"
            )
        G.add_edge(e["id"], e["src"], e["tgt"], e["type"])
    return G


def _make_phi(age_op, age, jur):
    ops = {"<": lambda a, c: a < c, "<=": lambda a, c: a <= c,
           "=": lambda a, c: a == c, ">=": lambda a, c: a >= c,
           ">": lambda a, c: a > c}

    def phi(A, J):
        if jur and J != jur:
            return False
        if age_op and age is not None and not ops[age_op](A, age):
            return False
        return True
    return phi


def json_to_rule(d, source=None):
    """Build a Rule from a parsed extraction spec.

    K (the preserved interface) is conventionally "ids present in both L and
    R" (see the extraction prompt above and rules.py's module docstring: "K
    elements share the same id across L, K, R"). A real extraction run
    against Llama-3.3-70B-T produced a rule that reused an edge id ("1")
    across L and R for two STRUCTURALLY DIFFERENT edges (different target
    node, different type) -- a real model mistake, not a schema-vocabulary
    violation. Naive id-intersection treats id reuse alone as "preserved,"
    so apply_rule() left that edge untouched even though its L-side target
    node was being deleted, producing a graph with a dangling edge that
    doesn't fail until deep inside a later isomorphism check (an
    unattributable KeyError, not a clean, skippable parse error). The checks
    below require an id present in both L and R to actually denote the SAME
    node type / edge (src,tgt,type) in both, matching the manuscript's own
    requirement that the K-inclusion morphisms be type-preserving (sec:
    synthesis item 1) -- an id reused for inconsistent content is rejected
    as a parse failure, the same category as an off-schema symbol.
    """
    L = _mk_graph(d["L"])
    R = _mk_graph(d["R"])

    K_nodes = set(L.nodes) & set(R.nodes)
    for n in K_nodes:
        if L.nodes[n] != R.nodes[n]:
            raise ValueError(
                f"node {n!r} is type {L.nodes[n]!r} in L but {R.nodes[n]!r} "
                f"in R -- not a valid preserved interface element"
            )

    K_edges = set(L.edges) & set(R.edges)
    for e in K_edges:
        if L.edges[e] != R.edges[e]:
            raise ValueError(
                f"edge {e!r} is {L.edges[e]!r} in L but {R.edges[e]!r} in R "
                f"-- not a valid preserved interface element"
            )

    nacs = [_mk_graph(d["nac"])] if d.get("nac") else []
    return Rule(d.get("name", new_id("r")), L, K_nodes, K_edges, R, nacs,
                phi=_make_phi(d.get("age_op"), d.get("age"), d.get("jurisdiction")),
                template=d.get("template", ""), source=source,
                confidence=_parse_confidence(d.get("confidence")))


def _parse_confidence(v):
    """Coerce the extractor's self-reported confidence to a float in [0,1],
    or None if absent/unparseable. Not a validation failure on its own (a
    malformed confidence doesn't invalidate an otherwise-good rule) -- a
    None here just means certify.admit()'s descending-confidence sort
    (Algorithm 3) places this candidate after every candidate that DID
    report one, via the (confidence, name) sort key's stable tie-break."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def _cap(text, n=PROMPT_SPAN_CHARS):
    return text[:n]


def _single_prompt(p):
    return _EXTRACT_PROMPT % (sorted(TG_NODES), sorted(TG_EDGES), _ITEM_SCHEMA,
                              _SCHEMA_NOTE, p["jurisdiction"], p["tau"], _cap(p["text"]))


def _batch_prompt(ps):
    body = "\n\n".join(
        f"[{i}] (jurisdiction={p['jurisdiction']}, tau={p['tau']}):\n{_cap(p['text'])}"
        for i, p in enumerate(ps)
    )
    return _BATCH_EXTRACT_PROMPT % (len(ps), len(ps), _ITEM_SCHEMA, _SCHEMA_NOTE,
                                     sorted(TG_NODES), sorted(TG_EDGES), body)


def _provision_hash(model, temperature, max_tokens, p):
    basis = "|".join([model, repr(temperature), repr(max_tokens),
                       str(p.get("jurisdiction")), str(p.get("tau")), _cap(p["text"])])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _validate(d, source):
    """Try building a Rule purely to validate `d`; returns an error string,
    or None if `d` is valid. Doesn't keep the Rule -- callers reconstruct
    from the spec later (specs, not Rule objects, are what gets cached)."""
    try:
        json_to_rule(d, source=source)
        return None
    except Exception as e:
        return str(e)


def _extract_one(complete, p, repair_attempts=1):
    """Extract a single provision, retrying with an explicit repair prompt
    (showing the model its own bad output and the specific error) up to
    `repair_attempts` times if validation fails, before giving up. Returns
    the valid spec dict, or None."""
    try:
        raw = complete(_single_prompt(p))
        d = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception as e:
        print(f"  [skip] {p.get('id', '?')}: {e}")
        return None

    err = _validate(d, p.get("source"))
    attempts = 0
    while err is not None and attempts < repair_attempts:
        attempts += 1
        try:
            raw = complete(_REPAIR_PROMPT % (err, json.dumps(d), _SCHEMA_NOTE,
                                             p["jurisdiction"], p["tau"], _cap(p["text"])))
            d = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        except Exception as e:
            print(f"  [skip] {p.get('id', '?')}: repair attempt failed: {e}")
            return None
        err = _validate(d, p.get("source"))

    if err is not None:
        print(f"  [skip] {p.get('id', '?')}: {err}")
        return None
    return d


def _extract_batch(complete, ps, repair_attempts=1):
    """Extract a batch; returns a list of spec-dicts-or-None (same length and
    order as `ps`), or None entirely if the response wasn't a parseable,
    correctly-sized JSON array (caller falls back to per-provision calls).

    Passes an explicit, batch-scaled max_tokens: a real Groq run at the
    single-item default (MAX_TOKENS_PER_ITEM) truncated 5-item batch
    responses mid-JSON (repeatable "Expecting ',' delimiter" parse
    failures), silently forcing every batch into the slower per-provision
    fallback. +20% headroom for the JSON array's own wrapping/whitespace.
    """
    try:
        batch_max_tokens = int(MAX_TOKENS_PER_ITEM * len(ps) * 1.2)
        raw = complete(_batch_prompt(ps), max_tokens=batch_max_tokens)
        arr = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))
    except Exception as e:
        print(f"  [batch failed, falling back to per-provision] {e}")
        return None
    if not isinstance(arr, list) or len(arr) != len(ps):
        print(f"  [batch returned {len(arr) if isinstance(arr, list) else 'non-array'}, "
              f"expected {len(ps)} -- falling back to per-provision]")
        return None

    out = []
    for p, d in zip(ps, arr):
        err = _validate(d, p.get("source"))
        attempts = 0
        while err is not None and attempts < repair_attempts:
            attempts += 1
            try:
                raw = complete(_REPAIR_PROMPT % (err, json.dumps(d), _SCHEMA_NOTE,
                                                 p["jurisdiction"], p["tau"], _cap(p["text"])))
                d = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            except Exception as e:
                print(f"  [skip] {p.get('id', '?')}: repair attempt failed: {e}")
                d, err = None, None
                break
            err = _validate(d, p.get("source"))
        if err is not None:
            print(f"  [skip] {p.get('id', '?')}: {err}")
            d = None
        out.append(d)
    return out


def _load_cache(cache_path):
    """JSONL cache: one {"hash","spec","source"} object per line. Tolerates
    a truncated last line (e.g. process killed mid-write) by skipping it
    rather than failing the whole load."""
    cache = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[rec["hash"]] = {"spec": rec["spec"], "source": rec["source"]}
    return cache


def extract(complete, provisions_path="corpus/provisions.json", provisions=None,
            cache_path=None, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS,
            repair_attempts=1):
    """Extract candidate Rule objects from the segmented corpus: concurrent,
    batched, incrementally cached (see module docstring for the full design
    rationale and the benchmark that motivated it).

    Args:
        complete: str->str LLM client (must expose `.model`, e.g. from
            cpgtr.llm.make_complete -- used only to key the cache correctly).
        provisions_path: path to a provisions.json-shaped file, used when
            `provisions` is not given.
        provisions: an already-loaded list of provision dicts (e.g. a
            deduplicated, stratified sample) -- takes precedence over
            provisions_path when given.
        cache_path: if given, loaded first (resuming any already-completed
            work with zero LLM calls for cached items) and appended to
            incrementally as each new item completes -- an interrupted run
            never loses completed work.
        batch_size: provisions sent per LLM call (JSON array response).
        max_workers: ThreadPoolExecutor worker count.
        repair_attempts: extra LLM calls per failing candidate, showing it
            the specific validation error, before giving up on it.

    Each returned Rule's `.source` is set from its provision's "source"
    field, so callers can attribute proposed/certified counts back to the
    originating statute (Table 4, tab:corpus).
    """
    if provisions is None:
        provisions = json.load(open(provisions_path))
    provisions = list(provisions)

    model = getattr(complete, "model", "unknown-model")
    temperature = 0.0
    max_tokens = MAX_TOKENS_PER_ITEM

    cache = _load_cache(cache_path)
    results = {}          # hash -> {"spec", "source"}
    to_extract = []        # (hash, provision)
    for p in provisions:
        h = _provision_hash(model, temperature, max_tokens, p)
        if h in cache:
            results[h] = cache[h]
        else:
            to_extract.append((h, p))

    n_batches = -(-len(to_extract) // batch_size) if to_extract else 0
    print(f"[extract] {len(results)} loaded from cache, {len(to_extract)} to "
          f"extract ({n_batches} calls at batch_size={batch_size}, "
          f"{max_workers} workers)")

    cache_lock = threading.Lock()

    def _persist(h, entry):
        results[h] = entry
        if cache_path:
            with cache_lock:
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"hash": h, "spec": entry["spec"],
                                        "source": entry["source"]},
                                       ensure_ascii=False) + "\n")

    def process_batch(batch):
        if len(batch) == 1:
            h, p = batch[0]
            d = _extract_one(complete, p, repair_attempts)
            if d is not None:
                _persist(h, {"spec": d, "source": p.get("source")})
            return
        ds = _extract_batch(complete, [p for _, p in batch], repair_attempts)
        if ds is None:
            for h, p in batch:
                d = _extract_one(complete, p, repair_attempts)
                if d is not None:
                    _persist(h, {"spec": d, "source": p.get("source")})
            return
        for (h, p), d in zip(batch, ds):
            if d is not None:
                _persist(h, {"spec": d, "source": p.get("source")})

    if to_extract:
        batches = [to_extract[i:i + batch_size] for i in range(0, len(to_extract), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_batch, b) for b in batches]
            for fut in as_completed(futures):
                fut.result()   # re-raise any worker exception

    candidates = []
    for h, entry in results.items():
        try:
            candidates.append(json_to_rule(entry["spec"], source=entry["source"]))
        except Exception as e:
            print(f"  [skip, cached] {entry['spec'].get('name', '?')}: {e}")
    return candidates


def load_cached_candidates(cache_path):
    """Reconstruct candidate Rule objects from cache_path -- no LLM calls.
    Not a pure replay: a cache written before a json_to_rule() validation
    was tightened may contain specs that were valid THEN but are correctly
    rejected NOW -- skipped exactly as extract() itself would skip them, not
    silently included or fatally raised, so tightening a check never
    requires re-extraction to take effect on an existing cache."""
    candidates = []
    for entry in _load_cache(cache_path).values():
        try:
            candidates.append(json_to_rule(entry["spec"], source=entry["source"]))
        except Exception as e:
            print(f"  [skip, cached] {entry['spec'].get('name', '?')}: {e}")
    return candidates
