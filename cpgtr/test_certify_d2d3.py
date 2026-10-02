"""Unit tests for certification: creation dependency and admission rejections.

Covers the three creation-dependency mechanisms of `certify.creation_dependency`
(creation, removal of a NAC witness, release of a dangling-edge obstruction),
rejection of rules with a vacuous NAC, the "r1" / "cycle" reasons attached to
"reject:no-measure", and a regression check on the extracted rule pool.

Uses only the standard library. Run from the repository root with:
    python -m unittest cpgtr.test_certify_d2d3
"""
from __future__ import annotations
import os
import unittest

from cpgtr.graph import Graph
from cpgtr.rules import Rule
from cpgtr.certify import (
    creation_dependency, nac_is_vacuous, admit, r1_satisfied,
)


def _rule(name, l_nodes, l_edges, k_nodes, k_edges, r_nodes, r_edges,
          nacs=None, phi=None, confidence=1.0):
    L, R = Graph(), Graph()
    for n, ty in l_nodes.items():
        L.add_node(n, ty)
    for e, (s, t, ty) in l_edges.items():
        L.add_edge(e, s, t, ty)
    for n, ty in r_nodes.items():
        R.add_node(n, ty)
    for e, (s, t, ty) in r_edges.items():
        R.add_edge(e, s, t, ty)
    return Rule(name=name, L=L, K_nodes=set(k_nodes), K_edges=set(k_edges),
                R=R, nacs=nacs or [], phi=phi or (lambda A, J: True),
                confidence=confidence)


class TestD1Unchanged(unittest.TestCase):
    def test_d1_creation_still_detected(self):
        # r creates a Foo node r' needs; no deletions or NACs involved.
        r = _rule('r', {'x': 'Bar'}, {}, {'x'}, set(),
                   {'x': 'Bar', 'y': 'Foo'}, {})
        rp = _rule("r'", {'y': 'Foo'}, {}, set(), set(), {'y': 'Foo'}, {})
        self.assertTrue(creation_dependency(r, rp))

    def test_no_dependency_when_unrelated(self):
        r = _rule('r', {'x': 'Bar'}, {}, {'x'}, set(), {'x': 'Bar', 'y': 'Foo'}, {})
        rp = _rule("r'", {'z': 'Baz'}, {}, set(), set(), {'z': 'Baz'}, {})
        self.assertFalse(creation_dependency(r, rp))


class TestD2RemovesNacWitness(unittest.TestCase):
    def test_deleting_nac_witness_creates_dependency(self):
        # r deletes a 'Flag' node and r' has a NAC forbidding a 'Flag' node
        # beyond its own L. The check is a conservative over-approximation
        # keyed on type coincidence alone, so the pair is flagged dependent.
        r = _rule('r', {'f': 'Flag'}, {}, set(), set(), {}, {})
        rp_L = {'x': 'Bar'}
        rp_nac = Graph()
        rp_nac.add_node('x', 'Bar')
        rp_nac.add_node('f', 'Flag')
        rp = _rule("r'", rp_L, {}, {'x'}, set(), rp_L, {}, nacs=[rp_nac])
        self.assertTrue(creation_dependency(r, rp))

    def test_no_deletion_no_d2(self):
        # r creates (does not delete) a same-typed node: no NAC-witness
        # dependency.
        r = _rule('r', {'x': 'Bar'}, {}, {'x'}, set(), {'x': 'Bar', 'f': 'Flag'}, {})
        rp_L = {'y': 'Bar'}
        rp_nac = Graph()
        rp_nac.add_node('y', 'Bar')
        rp_nac.add_node('f', 'Flag')
        rp = _rule("r'", rp_L, {}, {'y'}, set(), rp_L, {}, nacs=[rp_nac])
        self.assertFalse(creation_dependency(r, rp))

    def test_joint_satisfiability_gates_d2(self):
        r = _rule('r', {'f': 'Flag'}, {}, set(), set(), {}, {},
                   phi=lambda A, J: J == 'EU')
        rp_L = {'x': 'Bar'}
        rp_nac = Graph()
        rp_nac.add_node('x', 'Bar')
        rp_nac.add_node('f', 'Flag')
        rp = _rule("r'", rp_L, {}, {'x'}, set(), rp_L, {}, nacs=[rp_nac],
                    phi=lambda A, J: J == 'UK')
        self.assertFalse(creation_dependency(r, rp))


class TestD3ReleasesDangling(unittest.TestCase):
    def test_deleting_edge_to_type_rprime_deletes(self):
        # r deletes an edge (a->b) whose target type is 'Node', and r' deletes
        # a node of type 'Node'; deleting the edge can release r' 's dangling
        # condition, so the pair is flagged dependent.
        r = _rule('r', {'a': 'Foo', 'b': 'Node'}, {'e': ('a', 'b', 'links')},
                   {'a'}, set(), {'a': 'Foo'}, {})
        rp = _rule("r'", {'c': 'Node'}, {}, set(), set(), {}, {})
        self.assertTrue(creation_dependency(r, rp))

    def test_no_d3_when_rprime_deletes_nothing(self):
        r = _rule('r', {'a': 'Foo', 'b': 'Node'}, {'e': ('a', 'b', 'links')},
                   {'a'}, set(), {'a': 'Foo'}, {})
        rp = _rule("r'", {'c': 'Node'}, {}, {'c'}, set(), {'c': 'Node'}, {})
        self.assertFalse(creation_dependency(r, rp))


class TestVacuousNac(unittest.TestCase):
    def test_empty_graph_nac_is_vacuous(self):
        r = _rule('dead', {'x': 'Foo'}, {}, {'x'}, set(), {'x': 'Foo'}, {},
                   nacs=[Graph()])
        self.assertTrue(nac_is_vacuous(r))

    def test_no_nacs_is_not_vacuous(self):
        r = _rule('live', {'x': 'Foo'}, {}, set(), set(), {}, {}, nacs=[])
        self.assertFalse(nac_is_vacuous(r))

    def test_real_nac_with_extra_element_is_not_vacuous(self):
        r_L = {'x': 'Foo'}
        nac = Graph()
        nac.add_node('x', 'Foo')
        nac.add_node('y', 'Bar')
        r = _rule('live', r_L, {}, {'x'}, set(), r_L, {}, nacs=[nac])
        self.assertFalse(nac_is_vacuous(r))

    def test_admission_rejects_vacuous_nac_rule(self):
        # A deleting rule (satisfies R1 via deletion) that also carries a
        # vacuous NAC can never fire, so it must be rejected as
        # "reject:vacuous-nac" rather than admitted.
        dying = _rule('dying', {'x': 'Foo'}, {}, set(), set(), {}, {},
                       nacs=[Graph()])
        self.assertTrue(r1_satisfied(dying))  # deletes -- R1 holds
        cert, log = admit([dying], verbose=False)
        self.assertEqual(cert, [])
        statuses = {name: status for name, status, _mod, _detail in log}
        self.assertEqual(statuses['dying'], 'reject:vacuous-nac')

    def test_admission_keeps_rule_with_no_nac(self):
        # A well-formed additive rule (R1 satisfied via a non-vacuous NAC
        # matching its own effect) must be admitted normally.
        L = {'x': 'Foo'}
        nac = Graph()
        nac.add_node('x', 'Foo')
        nac.add_node('y', 'Bar')
        nac.add_edge('e', 'x', 'y', 'has')
        fine = _rule('fine', L, {}, {'x'}, set(), {'x': 'Foo', 'y': 'Bar'},
                      {'e': ('x', 'y', 'has')}, nacs=[nac])
        self.assertTrue(r1_satisfied(fine))
        self.assertFalse(nac_is_vacuous(fine))
        cert, log = admit([fine], verbose=False)
        statuses = {name: status for name, status, _mod, _detail in log}
        self.assertEqual(statuses['fine'], 'admit')
        self.assertEqual(len(cert), 1)


class TestNoMeasureReasonSplit(unittest.TestCase):
    """"reject:no-measure" must distinguish, via detail['reason'], an
    individual (R1) failure from membership in a creation-dependency cycle."""

    def test_r1_failure_reason(self):
        # No deletion and no NAC: fails (R1) outright and never reaches the
        # round-based peeling, so the reason cannot be "cycle".
        r = _rule('no_r1', {'x': 'Foo'}, {}, {'x'}, set(),
                   {'x': 'Foo', 'y': 'Bar'}, {}, nacs=[])
        cert, log = admit([r], verbose=False)
        detail = next(d for n, s, m, d in log if n == 'no_r1')
        self.assertEqual(detail['reason'], 'r1')

    def test_cycle_reason(self):
        # Two rules that mutually creation-depend (each creates a type the
        # other's L needs) and each satisfy (R1) via an unrelated deletion,
        # so the round-based peeling runs and finds a genuine 2-cycle.
        a = _rule('a', {'p': 'Foo', 'junk': 'Junk'}, {}, {'p'}, set(),
                   {'p': 'Foo', 'q': 'Bar'}, {})
        b = _rule('b', {'q': 'Bar', 'junk2': 'Junk'}, {}, {'q'}, set(),
                   {'q': 'Bar', 'p2': 'Foo'}, {})
        cert, log = admit([a, b], verbose=False)
        statuses = {n: (s, d) for n, s, m, d in log}
        # A 2-cycle's minimum feedback vertex set has size 1 (as with the
        # r_FLIP1/r_FLIP2 fixtures): exactly one rule is rejected as the
        # cycle-breaker and the other is then ranked and admitted. The
        # rejected rule must carry reason='cycle'.
        rejected = [n for n in ('a', 'b') if statuses[n][0] == 'reject:no-measure']
        self.assertEqual(len(rejected), 1, statuses)
        self.assertEqual(statuses[rejected[0]][1]['reason'], 'cycle')


@unittest.skipUnless(
    os.path.exists('cache/canonical_extraction_cache.jsonl'),
    'regression fixture needs the real 200-candidate extraction cache',
)
class TestUSCARegression(unittest.TestCase):
    """On the real extracted candidate pool, two rules (named in the test
    below) form a walkable non-termination cycle at context cell J='US-CA'.
    A creation-dependency test that only considered rule creation would miss
    it; with all three mechanisms and vacuous-NAC rejection active, admission
    must not certify both halves of the cycle."""

    def test_uscacycle_no_longer_slips_through(self):
        from cpgtr.extract import _load_cache, json_to_rule
        cache = _load_cache('cache/canonical_extraction_cache.jsonl')
        candidates = [json_to_rule(e['spec'], source=e['source'], enforce_topology=True)
                      for e in cache.values()]
        cert, log = admit(candidates, verbose=False)
        cert_names = {r.name for r in cert}
        # The cycle's two rules must not both survive to the certified set.
        self.assertFalse(
            {'Provision0_OptOutIdentifierRestriction', 'InterpreterAssistanceRequirement'}
            <= cert_names,
            'both halves of the US-CA cycle were certified -- (D2)/(D3) did not catch it',
        )


if __name__ == '__main__':
    unittest.main()
