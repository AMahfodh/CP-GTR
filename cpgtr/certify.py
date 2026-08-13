"""Termination-by-construction (repair-typing) + Alg. 3 admission gate.


"""
from __future__ import annotations
import itertools
from .rules import Rule
from .cpa import critical_pairs, strongly_joinable
from .context import representative_contexts, jointly_satisfiable


def edge_types(G):
    return {ty for (_, _, ty) in G.edges.values()}


def node_types(G):
    return set(G.nodes.values())


def r1_satisfied(rule: Rule) -> bool:
    """(R1) firing discharges the match: deletion does it directly; an additive
    rule needs a NAC that its RHS satisfies by construction."""
    if rule.del_nodes() or rule.del_edges():
        return True
    for nac in rule.nacs:
        nac_extra_types = {ty for e, (s, t, ty) in nac.edges.items()
                           if e not in rule.L.edges}
        if nac_extra_types & edge_types(rule.R):
            return True
    return False


def creation_dependency(r: Rule, r_prime: Rule) -> bool:
    """r ~> r' ("r may enable r'"), CP-GTR_V2.tex Definition "Creation
    dependency": some element of r's freshly created structure has a type
    that occurs in r''s left-hand side, AND the two guards are jointly
    satisfiable (some (A,J) makes both Phi_r and Phi_r' true).

    The joint-satisfiability clause matters: two rules that can never both be
    active for the same (A,J) can never actually interact on any real host,
    so a bare type coincidence between them must not count as a
    stratification dependency. Dropping this clause (as an earlier version of
    this module did) makes the stratification needlessly conservative -- and,
    worse, can make some jointly-unsatisfiable-guard rule sets appear
    cyclic when no real interaction is possible.
    """
    created_e = edge_types(r.R) - edge_types(r.L)
    created_n = node_types(r.R) - node_types(r.L)
    overlap = bool((created_e & edge_types(r_prime.L)) or
                   (created_n & node_types(r_prime.L)))
    if not overlap:
        return False
    return jointly_satisfiable(r.phi, r_prime.phi)


def _incident_signature(node_id, G):
    """For a node in graph G, the set of (direction, edge_type, other_node_type)
    describing its incident edges -- a cheap structural fingerprint used by
    creation_dependency_instance() to check more than bare type coincidence."""
    sig = set()
    for (s, t, ty) in G.edges.values():
        if s == node_id:
            sig.add(("out", ty, G.nodes.get(t)))
        elif t == node_id:
            sig.add(("in", ty, G.nodes.get(s)))
    return sig


def creation_dependency_instance(r: Rule, r_prime: Rule) -> bool:
    """Tightened creation-dependency test (an OPTION alongside
    creation_dependency(), not a replacement -- see certify.stratify()'s
    `instance_level=` parameter). The type-level test above registers a
    dependency whenever a created element's TYPE occurs anywhere in r_prime's
    L, which on a single-domain corpus where type reuse is pervasive (the
    same handful of node/edge types recur across dozens of unrelated
    provisions) can flag a dependency between rules that could never actually
    interact, because the created element's own local structure doesn't
    match what r_prime's left-hand side actually needs at that node.

    Requires that a created element could plausibly COMPLETE a real match of
    r_prime's L, not just share a type with something in it: for each node r
    creates, its incident-edge signature *within R* (what it's actually
    connected to, right after r fires) must be a superset of the incident
    signature some same-typed node in r_prime's L requires *within L* -- if
    L's node needs a specific incident edge that the created node doesn't
    actually have, the created node cannot be the completion L is asking
    for, regardless of type. A same-typed L-node with NO incident-edge
    requirements at all is left as a bare type match (nothing further to
    check). Created edges are covered by the same node-signature check on
    their endpoints, since an edge's only "structure" beyond its own type is
    what it's attached to.

    This is a heuristic tightening, not a formal subgraph-isomorphism
    decision procedure -- it reduces false positives from type reuse but
    does not attempt full pattern matching against r_prime's whole L (e.g. it
    does not check multi-node conjunctive requirements across L). Report
    both this and the type-level rate side by side (Task 6); do not silently
    replace one with the other.
    """
    if not jointly_satisfiable(r.phi, r_prime.phi):
        return False

    for cn in r.add_nodes():
        cn_type = r.R.nodes[cn]
        cn_sig = _incident_signature(cn, r.R)
        for ln, ln_type in r_prime.L.nodes.items():
            if ln_type != cn_type:
                continue
            ln_sig = _incident_signature(ln, r_prime.L)
            if not ln_sig or ln_sig <= cn_sig:
                return True

    created_edge_types = edge_types(r.R) - edge_types(r.L)
    if created_edge_types:
        for (s, t, ty) in r_prime.L.edges.values():
            if ty not in created_edge_types:
                continue
            # the edge's endpoints in L must themselves be typed compatibly
            # with what r actually produced (endpoint types, not just the
            # edge's own type) -- otherwise the edge type alone is exactly
            # the bare-type-reuse false positive this function exists to avoid.
            s_ty, t_ty = r_prime.L.nodes.get(s), r_prime.L.nodes.get(t)
            r_node_types = node_types(r.R)
            if s_ty in r_node_types and t_ty in r_node_types:
                return True

    return False


# --------------------------------------------------------------------------- #
# Small graph primitives for cycle detection / minimum feedback vertex set,   #
# used only when the round-based peeling below stalls.                       #
# --------------------------------------------------------------------------- #
def _sccs(nodes, edge_fn):
    """Tarjan's strongly-connected-components algorithm. `edge_fn(n)` yields
    successor ids. Returns a list of components (each a list of ids),
    including trivial singletons."""
    index_counter = [0]
    stack, lowlink, index, on_stack = [], {}, {}, set()
    result = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edge_fn(v):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    for v in nodes:
        if v not in index:
            strongconnect(v)
    return result


def _has_cycle(nodes_subset, edge_fn):
    subset = set(nodes_subset)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes_subset}

    def dfs(v):
        color[v] = GRAY
        for w in edge_fn(v):
            if w not in subset:
                continue
            if color[w] == GRAY:
                return True
            if color[w] == WHITE and dfs(w):
                return True
        color[v] = BLACK
        return False

    return any(color[v] == WHITE and dfs(v) for v in nodes_subset)


def _min_feedback_vertex_set(nodes, edge_fn, brute_force_limit=16):
    """Smallest subset of `nodes` whose removal makes the induced subgraph
    acyclic. Exact via brute-force subset search up to `brute_force_limit`
    nodes (ample for the small rule-interaction cycles expected in this
    domain -- legal repair rules have |L| <= 6 elements and realistic cycles
    involve a handful of rules). Falls back to a greedy "repeatedly drop the
    highest-out-degree node" heuristic beyond that size: always sound (the
    remainder is acyclic) but not guaranteed minimum.
    """
    nodes = list(nodes)
    if not _has_cycle(nodes, edge_fn):
        return []

    if len(nodes) <= brute_force_limit:
        for size in range(1, len(nodes) + 1):
            for combo in itertools.combinations(nodes, size):
                remainder = [n for n in nodes if n not in combo]
                if not _has_cycle(remainder, edge_fn):
                    return list(combo)
        return nodes  # unreachable: removing every node is always acyclic

    removed, remaining = [], list(nodes)
    while _has_cycle(remaining, edge_fn):
        outdeg = {n: sum(1 for w in edge_fn(n) if w in remaining) for n in remaining}
        worst = max(remaining, key=lambda n: outdeg[n])
        removed.append(worst)
        remaining.remove(worst)
    return removed


# --------------------------------------------------------------------------- #
def stratify(rules, dmax=None, instance_level=False):
    """Compute the pointwise-least GLOBAL stratification rho for `rules` as a
    whole (CP-GTR Definition "Stratification and repair measure" +
    Definition "Repair-typed rule" + Proposition "Existence and computation of
    a stratification"), via round-based layered elimination.

    This is a fixed point computed once over the WHOLE proposed set, not
    incrementally per admitted candidate -- see the manuscript's "Why the
    stratification must be global" paragraph and docs/manuscript_excerpts.md
    section 2 for the r1~>r2~>r1 counterexample that an incremental,
    per-candidate check misses.

    Round k peels off every remaining rule that does not creation-depend
    (r ~> r', Definition "Creation dependency") on any rule that is *still
    unranked* -- an unranked rule could end up at any rank >= k, so admitting
    a dependency into it would violate the "strictly lower rank" requirement.
    This computes the same quantity as the manuscript's value-iteration
    formula rho^(t+1)(r) = max({0} u {rho^(t)(r')+1 : r~>r'}) -- both are
    standard longest-path-in-DAG computations and agree exactly (verified);
    peeling was independently arrived at before this module was checked
    against the manuscript's formal statement of it.

    When a round makes no progress, `remaining` contains at least one cycle
    in ~>. Per the Proposition, no stratification exists for a cyclic rule
    set; rather than rejecting everything left, a MINIMUM feedback vertex set
    of each offending strongly-connected component is rejected (rho=None) and
    peeling resumes -- removing those specific rules may unblock the rest of
    the component or rules depending on it.

    Mutates `.rho` on every rule in `rules` (None if unstratifiable, including
    rules that fail R1 outright, or that were selected into a minimum
    feedback vertex set). Returns (cyclic_ids, cycle_peers):
      cyclic_ids:  set of rule ids rejected specifically because they lie in
                   a cycle (a subset of the rules left with rho=None), for
                   callers that want to distinguish "no self-discharging
                   NAC" (R1) rejections from "cyclic creation dependency"
                   rejections.
      cycle_peers: {rule_id: [other rule names in the same strongly-connected
                   component]}, for a rejected rule -- this is the actual
                   counterexample a refinement step needs (Stage 3: "the
                   offending critical pair is serialized and returned to the
                   extractor as a counterexample" -- the analogous object
                   for a stratification rejection is the cycle itself, not
                   just the fact that one exists). Only populated for rules
                   in `cyclic_ids`.

    instance_level: use creation_dependency_instance() (Task 6: requires the
        created element's incident structure to actually be compatible with
        what the target rule's left-hand side needs, not just a bare type
        match) instead of the default type-level creation_dependency(). An
        OPTION, not the default -- pass True only to compare rejection rates
        (see run_e2e.py's dependency-rate comparison), not to silently
        change what a real run certifies.
    """
    dep_fn = creation_dependency_instance if instance_level else creation_dependency
    rules = list(rules)
    if dmax is None:
        # Proposition "Existence and computation of a stratification": the
        # iteration reaches a fixed point in at most |R| rounds, and a rank
        # reaching |R| itself witnesses a cycle. |R| is therefore the natural,
        # always-sufficient bound -- not an arbitrary small constant.
        dmax = len(rules)

    remaining = []
    for r in rules:
        if not r1_satisfied(r):
            r.rho = None
        else:
            remaining.append(r)

    cyclic_ids = set()
    cycle_peers = {}
    k = 0
    while remaining:
        if k > dmax:
            for r in remaining:
                r.rho = None
            break

        this_round = [
            r for r in remaining
            if not any(dep_fn(r, other)
                       for other in remaining if other is not r)
        ]

        if this_round:
            for r in this_round:
                r.rho = k
            this_round_ids = {id(r) for r in this_round}
            remaining = [r for r in remaining if id(r) not in this_round_ids]
            k += 1
            continue

        # No progress: find the cycle(s) and reject a minimum feedback vertex
        # set of each, per Proposition "Existence and computation of a
        # stratification".
        rid_to_rule = {id(r): r for r in remaining}

        def edge_fn(rid, _map=rid_to_rule):
            r = _map[rid]
            return [id(o) for o in remaining
                    if o is not r and dep_fn(r, o)]

        sccs = _sccs(list(rid_to_rule), edge_fn)
        round_cyclic = set()
        for comp in sccs:
            if len(comp) > 1 or (len(comp) == 1 and comp[0] in edge_fn(comp[0])):
                fvs = _min_feedback_vertex_set(comp, edge_fn)
                round_cyclic.update(fvs)
                comp_names = [rid_to_rule[rid].name for rid in comp]
                for rid in fvs:
                    my_name = rid_to_rule[rid].name
                    cycle_peers[rid] = [n for n in comp_names if n != my_name]

        if not round_cyclic:
            # Defensive: this_round was empty so something must be cyclic;
            # avoid looping forever if that invariant is ever violated.
            for r in remaining:
                r.rho = None
            break

        for rid in round_cyclic:
            rid_to_rule[rid].rho = None
        cyclic_ids |= round_cyclic
        remaining = [r for r in remaining if id(r) not in round_cyclic]

    return cyclic_ids, cycle_peers


def admit(candidates, dmax=None, use_cpa=True, use_strat=True, verbose=True,
          contexts=None):
    """Algorithm 3 (Bootstrap-and-Refine). Stage 3 (Refine) itself lives in
    cpgtr/refine.py, which wraps this function rather than being inlined
    here -- admit() stays LLM-agnostic (pure graph algorithm), and refine.py
    consumes the `detail` field this function attaches to each rejection to
    build the formal counterexample Stage 3 re-prompts with.

    Args:
        contexts: optional override of the (A,J) points checked for
            confluence. Defaults to the CONTEXT CELLS of `candidates`
            (CP-GTR_V2.tex "Context cells"): one representative point per
            maximal region where the active rule set is constant. This is
            what makes the certified guarantee hold for every (A,J) in
            N x J rather than only a sampled grid -- pass an explicit list
            only for smaller/faster ad hoc checks (e.g. tests).

    Returns (certified_rules, log) where log records the decision per
    candidate as (name, status, modality_or_None, detail). detail is:
      - for "reject:no-measure": {"cycle_peers": [other rule names in the
        same creation-dependency cycle]} -- the counterexample Stage 3 needs
        for a stratification rejection.
      - for "reject:non-joinable-CP": {"peer": conflicting rule's name,
        "S": overlap graph, "H1"/"H2": the two divergent outcomes} -- the
        formal critical pair Stage 3 needs (CP-GTR.tex: "the offending
        critical pair is serialized... as a formal object, not a
        natural-language error string").
      - otherwise: None.
    """
    # Algorithm 3, line "sort R_prop by descending extractor confidence"
    # (CP-GTR_V2:30): the confluence loop below only checks each
    # candidate against ALREADY-ADMITTED rules (`cert`), never against
    # not-yet-processed ones, so which of several mutually-conflicting
    # candidates gets admitted is decided by processing order. On real
    # extracted data (not the small hand-built library) this is not a
    # theoretical concern: the same 89-candidate real set certified anywhere
    # from 12 to 17 rules depending purely on input order before this sort
    # was added. Confidence ties (or missing confidence -- hand-written
    # library.py rules, or a cached spec predating this field) break on
    # `name` so the order is fully deterministic regardless of what order
    # the caller happens to hand candidates in (e.g. ThreadPoolExecutor
    # completion order during real extraction).
    candidates = sorted(candidates,
                         key=lambda r: (-(r.confidence if r.confidence is not None else -1),
                                        r.name))
    if contexts is None:
        contexts = representative_contexts(candidates)

    cert, log = [], []

    cycle_peers = {}
    if use_strat:
        _, cycle_peers = stratify(candidates, dmax)   # sets .rho on every candidate up front

    for r in candidates:
        if use_strat:
            if r.rho is None:
                detail = {"cycle_peers": cycle_peers.get(id(r), [])}
                log.append((r.name, "reject:no-measure", None, detail))
                if verbose:
                    print(f"[strat-reject] {r.name}")
                continue
        else:
            # --Strat ablation: naive size measure (accept if |R| <= |L|)
            if len(r.R.nodes) + len(r.R.edges) > len(r.L.nodes) + len(r.L.edges):
                log.append((r.name, "reject:size", None, None))
                continue
            r.rho = 0

        admit_ok, witness, cpa_timeout = True, None, False
        if use_cpa:
            try:
                for A, J in contexts:
                    if not r.phi(A, J):
                        continue
                    active = [x for x in cert if x.phi(A, J)] + [r]
                    for p in active:
                        for (S, H1, H2, pers) in critical_pairs(r, p):
                            if not strongly_joinable(H1, H2, pers, active):
                                admit_ok, witness = False, {"peer": p.name, "S": S, "H1": H1, "H2": H2}
                                break
                        if not admit_ok:
                            break
                    if not admit_ok:
                        break
            except RuntimeError:
                # StronglyJoinable's NormalForms search assumes a terminating
                # rule set ("Termination of the search" paragraph). Under
                # --Strat that precondition can fail -- conservatively treat
                # "couldn't establish joinability within budget" as not
                # joinable, rather than crashing the whole run.
                admit_ok, cpa_timeout = False, True

        if admit_ok:
            cert.append(r)
            log.append((r.name, "admit", r.modality(), None))
            if verbose:
                print(f"[admit]        {r.name}  (rank {r.rho}, {r.modality()})")
        elif cpa_timeout:
            log.append((r.name, "reject:cpa-nonterm", None, None))
            if verbose:
                print(f"[cpa-reject]   {r.name}  (confluence search did not "
                      f"terminate within budget)")
        else:
            log.append((r.name, "reject:non-joinable-CP", None, witness))
            if verbose:
                print(f"[cpa-reject]   {r.name}  (non-joinable critical pair "
                      f"vs {witness['peer'] if witness else '?'})")
    return cert, log
