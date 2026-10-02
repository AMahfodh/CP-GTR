"""Certification: termination by stratification, and the admission gate.

Implements the two certification checks applied to candidate rules:

* Termination. Rules are ranked by a global stratification over the creation
  dependency relation (`creation_dependency`); a rule set with a dependency
  cycle has no stratification, so a minimum feedback vertex set of each cycle
  is rejected (`stratify`).
* Confluence. Each candidate is checked against the already-admitted rules for
  strong joinability of every critical pair, at every context cell
  (`admit`, using `cpgtr.cpa` and `cpgtr.context`).

`admit` is the paper's Algorithm 3 (Bootstrap-and-Refine) without the refinement
stage, which is implemented separately in `cpgtr/refine.py`.
"""
from __future__ import annotations
import itertools
import time
from .rules import Rule
from .cpa import critical_pairs, strongly_joinable
from .context import representative_contexts, jointly_satisfiable


def edge_types(G):
    """Set of edge types occurring in graph `G`."""
    return {ty for (_, _, ty) in G.edges.values()}


def node_types(G):
    """Set of node types occurring in graph `G`."""
    return set(G.nodes.values())


def nac_is_vacuous(rule: Rule) -> bool:
    """True if some NAC of `rule` has no element outside its left-hand side.

    Such a NAC is satisfied by the rule's own match alone, so `nac_violated`
    (cpgtr/rules.py) reports it violated on every host where L matches. The
    rule can therefore never fire, under any host or context. Admission
    rejects these rules rather than certifying dead code.
    """
    return any(not nac.nodes and not nac.edges for nac in rule.nacs)


def r1_satisfied(rule: Rule) -> bool:
    """Condition (R1): firing the rule discharges the match it fired on.

    A deleting rule discharges its match directly. An additive rule needs a NAC
    whose extra edge types are produced by the rule's RHS, so the NAC is
    violated by the rule's own result and the rule cannot fire on it again.
    """
    if rule.del_nodes() or rule.del_edges():
        return True
    for nac in rule.nacs:
        nac_extra_types = {ty for e, (s, t, ty) in nac.edges.items()
                           if e not in rule.L.edges}
        if nac_extra_types & edge_types(rule.R):
            return True
    return False


def _deleted_types(r: Rule):
    """Return (node types, edge types) of the elements `r` deletes (L \\ K)."""
    node_types_del = {r.L.nodes[n] for n in r.del_nodes()}
    edge_types_del = {r.L.edges[e][2] for e in r.del_edges()}
    return node_types_del, edge_types_del


def _creates(r: Rule, r_prime: Rule) -> bool:
    """Mechanism 1, enabling by creation: `r` creates an element whose type
    occurs in the left-hand side of `r_prime`."""
    created_e = edge_types(r.R) - edge_types(r.L)
    created_n = node_types(r.R) - node_types(r.L)
    return bool((created_e & edge_types(r_prime.L)) or
                (created_n & node_types(r_prime.L)))


def _removes_nac_witness(r: Rule, r_prime: Rule) -> bool:
    """Mechanism 2, enabling by removing a NAC witness.

    `r` deletes an element whose type occurs in a NAC of `r_prime` outside
    that rule's left-hand side. Deleting it can turn a violated NAC into a
    satisfied one, newly enabling `r_prime`.
    """
    del_n_types, del_e_types = _deleted_types(r)
    if not del_n_types and not del_e_types:
        return False
    for nac in r_prime.nacs:
        extra_n_types = {nac.nodes[n] for n in nac.nodes if n not in r_prime.L.nodes}
        extra_e_types = {ty for e, (s, t, ty) in nac.edges.items() if e not in r_prime.L.edges}
        if (del_n_types & extra_n_types) or (del_e_types & extra_e_types):
            return True
    return False


def _releases_dangling(r: Rule, r_prime: Rule) -> bool:
    """Mechanism 3, enabling by releasing the dangling condition.

    `r` deletes an edge with an endpoint whose type is the type of a node that
    `r_prime` deletes. Removing that edge can lift the dangling-edge obstruction
    that blocked `r_prime`.
    """
    rp_del_node_types = {r_prime.L.nodes[n] for n in r_prime.del_nodes()}
    if not rp_del_node_types:
        return False
    for e in r.del_edges():
        s, t, ty = r.L.edges[e]
        if r.L.nodes.get(s) in rp_del_node_types or r.L.nodes.get(t) in rp_del_node_types:
            return True
    return False


def creation_dependency(r: Rule, r_prime: Rule) -> bool:
    """Creation dependency r ~> r' ("r may enable r'").

    True when r' may become newly matchable, or newly discharge its (R1)
    obligation, because of what r does, via at least one of three mechanisms,
    and the two guards are jointly satisfiable (some (A, J) makes both
    context predicates true):

      (1) r creates a type that r' needs in its left-hand side;
      (2) r deletes a NAC witness of r' (outside r' 's left-hand side), so
          that NAC goes from violated to satisfied;
      (3) r deletes an edge whose endpoint type is a node type that r'
          deletes, releasing a dangling-condition obstruction on r'.

    Joint satisfiability is required because two rules that can never be
    active for the same (A, J) cannot interact on any host, so a bare type
    coincidence between them is not a real dependency; counting it would make
    the stratification needlessly conservative and could make rule sets look
    cyclic when no interaction is possible.
    """
    if not (_creates(r, r_prime) or _removes_nac_witness(r, r_prime)
            or _releases_dangling(r, r_prime)):
        return False
    return jointly_satisfiable(r.phi, r_prime.phi)


def _incident_signature(node_id, G):
    """Set of (direction, edge_type, other_node_type) for the edges of `node_id`.

    A cheap structural fingerprint used by `creation_dependency_instance`.
    """
    sig = set()
    for (s, t, ty) in G.edges.values():
        if s == node_id:
            sig.add(("out", ty, G.nodes.get(t)))
        elif t == node_id:
            sig.add(("in", ty, G.nodes.get(s)))
    return sig


def creation_dependency_instance(r: Rule, r_prime: Rule) -> bool:
    """Tighter, instance-level variant of `creation_dependency` (an option).

    The type-level test registers a dependency whenever a created element's
    type occurs anywhere in r_prime's left-hand side. When the same few
    node/edge types recur across many unrelated provisions, this flags
    dependencies between rules that cannot actually interact. This variant
    additionally requires that a created element could plausibly complete a
    match of r_prime's L: for each node r creates, its incident-edge signature
    in R must be a superset of the signature of some same-typed node in
    r_prime's L (an L-node with no incident edges is a bare type match).
    Created edge types are accepted when both endpoint types of a matching
    L-edge occur in R.

    This is a heuristic, not a subgraph-isomorphism test: it does not match
    r_prime's whole L (for example, multi-node conjunctive requirements are
    not checked). Results should be reported alongside the type-level test.
    Guard joint-satisfiability is required as in `creation_dependency`.
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
            # The edge's endpoint types must also occur in R; the edge type
            # alone is the bare-type-reuse case this variant is meant to avoid.
            s_ty, t_ty = r_prime.L.nodes.get(s), r_prime.L.nodes.get(t)
            r_node_types = node_types(r.R)
            if s_ty in r_node_types and t_ty in r_node_types:
                return True

    return False


# --------------------------------------------------------------------------- #
# Small graph primitives for cycle detection and minimum feedback vertex set, #
# used only when the round-based peeling in stratify() stalls.                #
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
    """True if the subgraph induced on `nodes_subset` contains a cycle (DFS)."""
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
    """Smallest subset of `nodes` whose removal makes the induced subgraph acyclic.

    Exact (brute-force subset search) for up to `brute_force_limit` nodes, which
    is ample for the small rule-interaction cycles that occur in practice.
    Beyond that size it repeatedly drops the highest-out-degree node: the
    remainder is always acyclic, but the set is not guaranteed minimum.
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
    """Compute the least global stratification rho for `rules`, in rounds.

    Round k assigns rank k to every remaining rule that does not
    creation-depend (r ~> r') on any other still-unranked rule: an unranked
    rule may end up at any rank >= k, so a dependency on it would violate the
    requirement that dependencies point to strictly lower ranks. This is the
    longest-path computation of rho(r) = max({0} U {rho(r') + 1 : r ~> r'}).

    The ranking is computed once over the whole proposed set, not incrementally
    per admitted candidate: an incremental check can admit both rules of a
    two-rule cycle r1 ~> r2 ~> r1, depending on the order they are examined.

    When a round makes no progress, the remaining rules contain a cycle in ~>
    and no stratification exists for them. Rather than rejecting everything
    left, a minimum feedback vertex set of each strongly-connected component
    is rejected (rho = None) and peeling resumes, since removing those rules
    may unblock the rest of the component and the rules that depend on it.

    Mutates `.rho` on every rule in `rules` (None if unstratifiable: the rule
    fails R1, or was selected into a minimum feedback vertex set).

    Args:
        rules: iterable of candidate rules.
        dmax: round limit; defaults to len(rules), which always suffices (a
            rank reaching |R| witnesses a cycle).
        instance_level: use `creation_dependency_instance` instead of the
            type-level `creation_dependency`. Intended for comparing rejection
            rates, not for the default certification run.

    Returns:
        (cyclic_ids, cycle_peers). `cyclic_ids` is the set of `id(rule)` for
        rules rejected because they lie on a dependency cycle (as opposed to
        failing R1). `cycle_peers` maps `id(rule)` for those rules to the names
        of the other rules in the same strongly-connected component, which is
        the counterexample the refinement stage needs.
    """
    dep_fn = creation_dependency_instance if instance_level else creation_dependency
    rules = list(rules)
    if dmax is None:
        # The iteration reaches a fixed point in at most |R| rounds, and a
        # rank reaching |R| witnesses a cycle, so |R| is always sufficient.
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

        # No progress: find the cycles and reject a minimum feedback vertex
        # set of each.
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
            # Defensive: an empty round implies a cycle, so this should not
            # happen; reject the rest rather than loop forever.
            for r in remaining:
                r.rho = None
            break

        for rid in round_cyclic:
            rid_to_rule[rid].rho = None
        cyclic_ids |= round_cyclic
        remaining = [r for r in remaining if id(r) not in round_cyclic]

    return cyclic_ids, cycle_peers


def admit(candidates, dmax=None, use_cpa=True, use_strat=True, verbose=True,
          contexts=None, edgefree="keep", deadline=None):
    """Admission gate: the paper's Algorithm 3 without the refinement stage.

    Candidates are processed in descending extractor confidence. Each is
    checked for (1) a repair rank from the global stratification and (2)
    strong joinability of all its critical pairs against the already-admitted
    rules, at every context in `contexts` where it is active. Refinement
    (re-prompting the extractor with a rejection) lives in `cpgtr/refine.py`,
    which wraps this function; `admit` itself is a pure graph algorithm and
    attaches a `detail` payload to each rejection for that purpose.

    Args:
        candidates: iterable of candidate `Rule` objects.
        dmax: round limit passed to `stratify`.
        use_cpa: if False, skip the confluence check (ablation).
        use_strat: if False, replace stratification with a naive size measure
            (accept if |R| <= |L|) (ablation).
        verbose: print one line per decision.
        contexts: list of (A, J) points checked for confluence. Defaults to
            the context cells of `candidates` (one representative point per
            maximal region where the active rule set is constant), so the
            guarantee holds for every (A, J) rather than a sampled grid. Pass
            an explicit list only for smaller, faster ad hoc checks.
        edgefree: "keep" sends candidates whose L has no edge through the
            normal pipeline; "reject" rejects them up front as
            "reject:edgefree". An edge-free L matches any host containing
            that node type.
        deadline: optional wall-clock cutoff in `time.time()` seconds. It is
            checked once per candidate, before that candidate's confluence
            work starts, never mid-candidate. Once it has passed, the current
            and all remaining candidates are logged "budget-exceeded" and the
            function returns what was decided so far. This differs from
            "reject:cpa-nonterm", where one candidate's own confluence search
            exceeded its step budget.

    Returns:
        (certified_rules, log). `log` has one (name, status, modality, detail)
        entry per candidate. Statuses are "admit", "reject:no-measure",
        "reject:non-joinable-CP", "reject:cpa-nonterm", "reject:vacuous-nac",
        "reject:edgefree", "reject:size" and "budget-exceeded". `detail` is:
          - for "reject:no-measure": {"cycle_peers": names of the other rules
            in the same dependency cycle, "reason": "cycle" or "r1"};
          - for "reject:non-joinable-CP": {"peer": conflicting rule's name,
            "S": overlap graph, "H1"/"H2": the two divergent outcomes}, the
            critical pair serialized as a formal counterexample;
          - otherwise None.
    """
    # Process candidates in descending extractor confidence. The confluence
    # loop below only compares a candidate with already-admitted rules, so
    # when several candidates conflict, which one is admitted depends on
    # processing order; a fixed order makes the certified set independent of
    # input order. Confidence ties break on Rule.content_hash() rather than
    # `.name`, since names need not be unique across extracted candidates.
    candidates = sorted(candidates,
                         key=lambda r: (-(r.confidence if r.confidence is not None else -1),
                                        r.content_hash()))

    # Edge-free candidates: with edgefree="keep" they go through the normal
    # R1 / stratification / confluence checks like any other candidate; with
    # "reject" they are removed here, before stratification, on their own
    # "reject:edgefree" channel so the two settings can be compared.
    edgefree_rejected = []
    if edgefree not in ("keep", "reject"):
        raise ValueError(f"edgefree must be 'keep' or 'reject', got {edgefree!r}")
    if edgefree == "reject":
        kept = []
        for r in candidates:
            if not r.L.edges:
                edgefree_rejected.append(r)
            else:
                kept.append(r)
        candidates = kept

    # A rule with a vacuous NAC can never fire (see nac_is_vacuous). Reject it
    # on its own channel, before context-cell enumeration, so it does not
    # shape which rules count as active in any cell.
    vacuous_nac_rejected = [r for r in candidates if nac_is_vacuous(r)]
    if vacuous_nac_rejected:
        vacuous_ids = {id(r) for r in vacuous_nac_rejected}
        candidates = [r for r in candidates if id(r) not in vacuous_ids]

    if contexts is None:
        contexts = representative_contexts(candidates)

    cert, log = [], []
    for r in edgefree_rejected:
        log.append((r.name, "reject:edgefree", None, None))
        if verbose:
            print(f"[edgefree-reject] {r.name}  (L has no edge)")
    for r in vacuous_nac_rejected:
        log.append((r.name, "reject:vacuous-nac", None, None))
        if verbose:
            print(f"[vacuous-nac-reject] {r.name}  (NAC has no element outside L)")

    cycle_peers = {}
    cyclic_ids = set()
    if use_strat:
        cyclic_ids, cycle_peers = stratify(candidates, dmax)  # sets .rho on every candidate

    for idx, r in enumerate(candidates):
        if deadline is not None and time.time() >= deadline:
            for rest in candidates[idx:]:
                log.append((rest.name, "budget-exceeded", None, None))
            if verbose:
                print(f"[budget]       wall-clock deadline reached with "
                      f"{len(candidates) - idx} candidate(s) unprocessed")
            break
        if use_strat:
            if r.rho is None:
                # Both reasons share one status string; `detail["reason"]`
                # distinguishes a rule that individually fails R1 from one
                # selected into a minimum feedback vertex set of a cycle.
                detail = {"cycle_peers": cycle_peers.get(id(r), []),
                          "reason": "cycle" if id(r) in cyclic_ids else "r1"}
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
                # The normal-form search in strongly_joinable assumes a
                # terminating rule set. Without stratification that can fail,
                # so treat "joinability not established within budget" as not
                # joinable rather than aborting the run.
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
