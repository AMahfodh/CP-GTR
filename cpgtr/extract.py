"""Stage 1: schema-constrained LLM extraction of candidate repair rules.

Role in the pipeline: turns each statutory provision in
``corpus/provisions.json`` into a candidate `Rule` (a DPO graph-rewrite rule
with L/K/R patterns, optional NAC, a context predicate over (age, jurisdiction)
and a natural-language repair template). The candidates are passed unchanged to
the admission gate (`certify.admit`). `library.synthesize()` is only an offline
stand-in for demo runs.

Inputs: a ``complete(prompt) -> str`` client (see `llm.make_complete`) and a
list of provisions (``jurisdiction``, ``tau``, ``text``, ``source``).
Output: a list of `Rule` objects, each tagged with its originating ``source``.

Schema constraint: the model has no latitude over the symbol set. Node and edge
types must come from `TG_NODES` / `TG_EDGES`, edges must reference declared
nodes, ids shared by L and R must denote the same element, and (optionally)
every edge must be one of the canonical (source, edge, target) triples in
`cpgtr.topology.ATTACH`. A candidate that fails validation is sent back to the
model once or more with the specific error (a repair prompt) before it is
dropped.

Throughput and reliability:
  - Provisions are processed concurrently (ThreadPoolExecutor) so a few slow
    calls do not block the run.
  - Provisions are sent in batches (`BATCH_SIZE` per call, one JSON array
    back); a batch whose response is not a valid, correctly sized array falls
    back to one call per provision.
  - Results are cached incrementally as JSONL (one line per successful
    extraction, flushed immediately), keyed by a hash of (model, temperature,
    max_tokens, provision). An interrupted run loses no completed work, and a
    re-run with the same cache and inputs makes no further LLM calls.
  - Prompts contain only the provision's own text, truncated to
    `PROMPT_SPAN_CHARS` characters.
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

# Maximum number of characters of a provision's text included in a prompt.
# A statutory span is far more context than the extraction task needs; the cap
# keeps cost and prompt size consistent across provisions.
PROMPT_SPAN_CHARS = 800

BATCH_SIZE = 5
MAX_WORKERS = 8
# Completion budget per provision, used (scaled by batch size) for batch calls
# and as the basis of the cache key in `_provision_hash`. Single-provision
# calls use the larger SINGLE_ITEM_MAX_TOKENS below.
MAX_TOKENS_PER_ITEM = 1024

# max_tokens passed to complete() for a single-provision call (initial and
# repair). Reasoning-tuned models such as gpt-oss-120b spend part of the same
# max_tokens budget on a hidden reasoning trace, so a budget that fits the
# visible JSON answer can still truncate the response mid-JSON; 4096 leaves
# room for both. Batch calls scale their budget per item (see _extract_batch).
SINGLE_ITEM_MAX_TOKENS = 4096

_SCHEMA_NOTE = """An id used in both L and R must denote the exact same node
type (or the exact same src/tgt/type for an edge) in both -- do not reuse an
id for two structurally different elements. Preserve an id across L and R iff
the element is NOT deleted."""

def _canonical_schema_block():
    """Render the canonical (source, edge, target) triples as prompt text.

    The text is generated from `cpgtr.topology.ATTACH` (loaded from
    schema/attach.yaml), the same table the validator uses, so the prompt
    cannot drift from the topology check."""
    from .topology import ATTACH
    lines = ["Every edge in L, K, and R must be one of these EXACT "
             "(source type, edge type, target type) triples -- no other "
             "combination is admissible, even if the node/edge types are "
             "individually allowed:"]
    for (s, e, t) in sorted(ATTACH):
        lines.append(f"  {s} --{e}--> {t}")
    return "\n".join(lines)


# One minimal worked L/R example per admissible ATTACH triple, so the model
# sees every canonical arrangement at least once rather than only a prose
# description. The examples are small schema demonstrations, not realistic
# rules.
_ATTACH_WORKED_EXAMPLES = """Worked examples, one per admissible triple:

DataSubject--subjectOf-->ConsentProcess:
  L: {"nodes":[{"id":"ds1","type":"DataSubject"},{"id":"cp1","type":"ConsentProcess"}],
      "edges":[{"id":"e1","src":"ds1","tgt":"cp1","type":"subjectOf"}]}

ConsentProcess--grants-->DataProcessing:
  L: {"nodes":[{"id":"cp1","type":"ConsentProcess"}],"edges":[]}
  R: {"nodes":[{"id":"cp1","type":"ConsentProcess"},{"id":"dp1","type":"DataProcessing"}],
      "edges":[{"id":"e1","src":"cp1","tgt":"dp1","type":"grants"}]}

ConsentProcess--permits-->Withdrawal:
  L: {"nodes":[{"id":"cp1","type":"ConsentProcess"}],"edges":[]}
  R: {"nodes":[{"id":"cp1","type":"ConsentProcess"},{"id":"w1","type":"Withdrawal"}],
      "edges":[{"id":"e1","src":"cp1","tgt":"w1","type":"permits"}]}

ConsentProcess--requires-->ParentalConsent:
  L: {"nodes":[{"id":"cp1","type":"ConsentProcess"}],"edges":[]}
  R: {"nodes":[{"id":"cp1","type":"ConsentProcess"},{"id":"pc1","type":"ParentalConsent"}],
      "edges":[{"id":"e1","src":"cp1","tgt":"pc1","type":"requires"}]}

RetentionClause--hasDuration-->DurationBounded:
  L: {"nodes":[{"id":"rc1","type":"RetentionClause"},{"id":"di1","type":"DurationIndefinite"}],
      "edges":[{"id":"e1","src":"rc1","tgt":"di1","type":"hasDuration"}]}
  R: {"nodes":[{"id":"rc1","type":"RetentionClause"},{"id":"db1","type":"DurationBounded"}],
      "edges":[{"id":"e1","src":"rc1","tgt":"db1","type":"hasDuration"}]}

RetentionClause--hasDuration-->DurationIndefinite:
  L: {"nodes":[{"id":"rc1","type":"RetentionClause"}],"edges":[]}
  R: {"nodes":[{"id":"rc1","type":"RetentionClause"},{"id":"di1","type":"DurationIndefinite"}],
      "edges":[{"id":"e1","src":"rc1","tgt":"di1","type":"hasDuration"}]}

Operator--sellsTo-->ThirdParty:
  L: {"nodes":[{"id":"op1","type":"Operator"},{"id":"tp1","type":"ThirdParty"}],
      "edges":[{"id":"e1","src":"op1","tgt":"tp1","type":"sellsTo"}]}
  R: {"nodes":[{"id":"op1","type":"Operator"},{"id":"tp1","type":"ThirdParty"}],"edges":[]}
"""

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

# Both extraction prompts below state the allowed node/edge vocabularies, the
# exact admissible (source, edge, target) triples, and one worked example per
# triple, so a topology-violating candidate is a genuine deviation from the
# instructions rather than a consequence of an underspecified schema. A
# candidate that still violates the topology is caught by
# json_to_rule(..., enforce_topology=True) and sent back through the same
# repair loop as any other structural error (see _topology_rejection_message).
_EXTRACT_PROMPT = """Convert this statutory provision into a DPO repair rule as
STRICT JSON. A rule has L (pattern to find), R (pattern after repair), K
(preserved elements: ids present in BOTH L and R), NAC (forbidden context), a
context predicate, and a natural-language template for the repair.

Allowed node types: %s
Allowed edge types: %s

%s

%s
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

%s

%s
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
    """Build a `Graph` from an LLM-emitted ``{"nodes", "edges"}`` spec.

    Raises ValueError for an off-schema node or edge type, and for an edge
    whose src/tgt is not a node declared in the same spec. The endpoint check
    matters: a dangling edge would otherwise surface much later as an
    unattributable KeyError inside the isomorphism search, aborting the whole
    admission run instead of skipping one bad candidate.
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


def _topology_rejection_message(v_L, v_R, attach):
    """Build the feedback message for a topology-violating candidate.

    Names the offending (source, edge, target) triples in L and R and lists
    every admissible triple, so the repair prompt (which inserts this text as
    its ERROR) tells the model what to change. `v_L` / `v_R` are the violation
    lists from `canonical_violations`; `attach` is the admissible triple set."""
    lines = ["Non-canonical edge(s) -- every edge must be one of the "
             "admissible (source, edge, target) triples listed below."]
    for side, violations in (("L", v_L), ("R", v_R)):
        for (eid, src_t, ety, tgt_t) in violations:
            lines.append(f"  offending, in {side}: {src_t} --{ety}--> {tgt_t} (edge id {eid!r})")
    lines.append("Admissible triples (use ONE of these instead):")
    for (s, e, t) in sorted(attach):
        lines.append(f"  {s} --{e}--> {t}")
    return "\n".join(lines)


def _make_phi(age_op, age, jur):
    """Build the context predicate ``phi(A, J)`` from an optional age
    comparison (``age_op``, ``age``) and optional jurisdiction ``jur``.

    ``phi`` is True when the jurisdiction matches (if given) and the age
    comparison holds (if given)."""
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


def json_to_rule(d, source=None, enforce_topology=False):
    """Build a `Rule` from a parsed extraction spec.

    The preserved interface K is the set of node and edge ids present in both
    L and R. An id shared by L and R must denote the same node type, or the
    same (src, tgt, type) for an edge, on both sides; otherwise ValueError is
    raised. Without this check a reused id would be treated as "preserved"
    although the element differs, leaving a dangling edge after rule
    application.

    Args:
        d: parsed spec with ``L``, ``R``, optional ``nac``, ``age_op``,
            ``age``, ``jurisdiction``, ``template``, ``confidence`` and
            ``name``.
        source: originating statute, stored on the rule.
        enforce_topology: if True, an L or R containing an edge outside the
            canonical ATTACH table raises `cpgtr.topology.TopologyRejection`
            (a ValueError subclass, so it is rejected like any other
            off-schema candidate). Default False.
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

    if enforce_topology:
        from .topology import canonical_violations, TopologyRejection, ATTACH
        v_L = canonical_violations(L)
        v_R = canonical_violations(R)
        if v_L or v_R:
            raise TopologyRejection(_topology_rejection_message(v_L, v_R, ATTACH))

    nacs = [_mk_graph(d["nac"])] if d.get("nac") else []
    return Rule(d.get("name", new_id("r")), L, K_nodes, K_edges, R, nacs,
                phi=_make_phi(d.get("age_op"), d.get("age"), d.get("jurisdiction")),
                template=d.get("template", ""), source=source,
                confidence=_parse_confidence(d.get("confidence")))


def _parse_confidence(v):
    """Coerce the extractor's self-reported confidence to a float in [0, 1].

    Returns None if the value is absent or unparseable. A missing confidence
    does not invalidate the rule; it only means the admission gate's
    descending-confidence ordering places the candidate after those that
    reported one."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def _cap(text, n=PROMPT_SPAN_CHARS):
    """Truncate provision text to the prompt span limit."""
    return text[:n]


def _single_prompt(p):
    """Prompt for extracting one provision `p`."""
    return _EXTRACT_PROMPT % (sorted(TG_NODES), sorted(TG_EDGES),
                              _canonical_schema_block(), _ATTACH_WORKED_EXAMPLES,
                              _ITEM_SCHEMA, _SCHEMA_NOTE,
                              p["jurisdiction"], p["tau"], _cap(p["text"]))


def _batch_prompt(ps):
    """Prompt for extracting the list of provisions `ps` as one JSON array."""
    body = "\n\n".join(
        f"[{i}] (jurisdiction={p['jurisdiction']}, tau={p['tau']}):\n{_cap(p['text'])}"
        for i, p in enumerate(ps)
    )
    return _BATCH_EXTRACT_PROMPT % (len(ps), len(ps), _ITEM_SCHEMA, _SCHEMA_NOTE,
                                     sorted(TG_NODES), sorted(TG_EDGES),
                                     _canonical_schema_block(), _ATTACH_WORKED_EXAMPLES,
                                     body)


def _provision_hash(model, temperature, max_tokens, p):
    """Cache key: SHA-256 of the model, sampling settings and provision."""
    basis = "|".join([model, repr(temperature), repr(max_tokens),
                       str(p.get("jurisdiction")), str(p.get("tau")), _cap(p["text"])])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _validate(d, source, enforce_topology=False):
    """Validate spec `d` by building a Rule from it and discarding the result.

    Returns ``(error_message, is_topology_violation)``, or ``(None, False)`` if
    `d` is valid. Specs, not Rule objects, are what get cached, so callers
    rebuild the Rule from the spec later."""
    try:
        json_to_rule(d, source=source, enforce_topology=enforce_topology)
        return None, False
    except Exception as e:
        from .topology import TopologyRejection
        return str(e), isinstance(e, TopologyRejection)


def _record_rejection(stats, is_topology):
    """Thread-safely count a rejected candidate in `stats` (if supplied),
    separating topology rejections from all other rejection reasons."""
    if stats is None:
        return
    key = "topology_rejections" if is_topology else "other_rejections"
    with stats["_lock"]:
        stats[key] = stats.get(key, 0) + 1


def _extract_one(complete, p, repair_attempts=1, enforce_topology=False, stats=None):
    """Extract one provision, with repair retries.

    If the response fails validation, the model is shown its own output and the
    specific error (repair prompt) up to `repair_attempts` times before the
    candidate is dropped. Returns the valid spec dict, or None."""
    try:
        raw = complete(_single_prompt(p), max_tokens=SINGLE_ITEM_MAX_TOKENS)
        d = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception as e:
        print(f"  [skip] {p.get('id', '?')}: {e}")
        return None

    err, is_topo = _validate(d, p.get("source"), enforce_topology)
    attempts = 0
    while err is not None and attempts < repair_attempts:
        attempts += 1
        try:
            raw = complete(_REPAIR_PROMPT % (err, json.dumps(d), _SCHEMA_NOTE,
                                             p["jurisdiction"], p["tau"], _cap(p["text"])),
                            max_tokens=SINGLE_ITEM_MAX_TOKENS)
            d = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        except Exception as e:
            print(f"  [skip] {p.get('id', '?')}: repair attempt failed: {e}")
            _record_rejection(stats, False)
            return None
        err, is_topo = _validate(d, p.get("source"), enforce_topology)

    if err is not None:
        print(f"  [skip] {p.get('id', '?')}: {err}")
        _record_rejection(stats, is_topo)
        return None
    return d


def _extract_batch(complete, ps, repair_attempts=1, enforce_topology=False, stats=None):
    """Extract a batch of provisions in one call.

    Returns a list of spec dicts or None (same length and order as `ps`), or
    None overall if the response is not a parseable JSON array of the right
    length; the caller then falls back to per-provision calls. Items that fail
    validation get the same repair retries as `_extract_one`.

    The completion budget scales with batch size (plus 20% for the array's own
    wrapping), because a single-item budget truncates multi-item responses
    mid-JSON.
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
        err, is_topo = _validate(d, p.get("source"), enforce_topology)
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
                _record_rejection(stats, False)
                break
            err, is_topo = _validate(d, p.get("source"), enforce_topology)
        if err is not None:
            print(f"  [skip] {p.get('id', '?')}: {err}")
            _record_rejection(stats, is_topo)
            d = None
        out.append(d)
    return out


def _load_cache(cache_path):
    """Load the JSONL cache: one ``{"hash", "spec", "source"}`` object per line.

    Returns a dict keyed by hash. Unparseable lines (e.g. a last line truncated
    when a process was killed mid-write) are skipped rather than failing the
    load."""
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
            repair_attempts=1, enforce_topology=False, stats=None):
    """Extract candidate `Rule` objects from the segmented corpus.

    Extraction is concurrent, batched and incrementally cached (see the module
    docstring).

    Args:
        complete: str->str LLM client. Should expose `.model` (as clients from
            `cpgtr.llm.make_complete` do); it is used only to key the cache.
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
        enforce_topology: reject candidates whose edges are not canonical
            (see `json_to_rule`).
        repair_attempts: extra LLM calls per failing candidate, showing it
            the specific validation error, before giving up on it.
        stats: optional output dict. If given, it is updated in place with
            "topology_rejections" and "other_rejections" counts: candidates
            that never validated even after `repair_attempts` retries, split
            into topology violations and every other reason (JSON parse
            failure, off-schema symbol, malformed K, ...).

    Each returned Rule's `.source` is set from its provision's "source"
    field, so proposed/certified counts can be attributed back to the
    originating statute.
    """
    if provisions is None:
        provisions = json.load(open(provisions_path))
    provisions = list(provisions)

    if stats is not None:
        stats.setdefault("_lock", threading.Lock())
        stats.setdefault("topology_rejections", 0)
        stats.setdefault("other_rejections", 0)

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
            d = _extract_one(complete, p, repair_attempts, enforce_topology, stats)
            if d is not None:
                _persist(h, {"spec": d, "source": p.get("source")})
            return
        ds = _extract_batch(complete, [p for _, p in batch], repair_attempts, enforce_topology, stats)
        if ds is None:
            for h, p in batch:
                d = _extract_one(complete, p, repair_attempts, enforce_topology, stats)
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
            candidates.append(json_to_rule(entry["spec"], source=entry["source"],
                                            enforce_topology=enforce_topology))
        except Exception as e:
            print(f"  [skip, cached] {entry['spec'].get('name', '?')}: {e}")
    return candidates


def load_cached_candidates(cache_path):
    """Rebuild candidate `Rule` objects from the cache at `cache_path`.

    Makes no LLM calls. Specs are re-validated with `json_to_rule`, so a cached
    spec that no longer passes validation is skipped (as `extract` would skip
    it) instead of being included or raising."""
    candidates = []
    for entry in _load_cache(cache_path).values():
        try:
            candidates.append(json_to_rule(entry["spec"], source=entry["source"]))
        except Exception as e:
            print(f"  [skip, cached] {entry['spec'].get('name', '?')}: {e}")
    return candidates
