"""Hand-written fixture rules for the worked example and mechanism checks.

Contains the small rule library of the paper's worked example (additive,
subtractive and substitutive rules, plus a deliberately conflicting rule that
certification must reject) and seeded adversarial fixtures used to exercise
individual certification mechanisms (confluence and stratification).

These are fixtures, not extractor output. `synthesize()` returns the worked-
example candidate stream for offline smoke tests; reported results use rules
from the LLM extractor (`cpgtr/extract.py`). `lib.synthesize()` is a
deliberate placeholder that raises NotImplementedError.
"""
from .graph import Graph
from .rules import Rule




class _Library:
    def synthesize(self):
        raise NotImplementedError(
            "Return a small list of TOY rules matching your rule type. "
            "For smoke-testing plumbing only — never for reported results."
        )


lib = _Library()



def _g(nodes, edges):
    G = Graph()
    for i, t in nodes:
        G.add_node(i, t)
    for i, s, t, ty in edges:
        G.add_edge(i, s, t, ty)
    return G


def r_PC():                       # additive: missing parental consent
    L = _g([("DS", "DataSubject"), ("CP", "ConsentProcess")],
           [("e1", "DS", "CP", "subjectOf")])
    R = _g([("DS", "DataSubject"), ("CP", "ConsentProcess"), ("PC", "ParentalConsent")],
           [("e1", "DS", "CP", "subjectOf"), ("e2", "CP", "PC", "requires")])
    nac = _g([("DS", "DataSubject"), ("CP", "ConsentProcess"), ("PCn", "ParentalConsent")],
             [("e1", "DS", "CP", "subjectOf"), ("na", "CP", "PCn", "requires")])
    return Rule("r_PC", L, {"DS", "CP"}, {"e1"}, R, [nac],
                phi=lambda A, J: A < 16 and J == "EU",
                template="Because you are under 16, this consent must be given or "
                         "authorised by the holder of parental responsibility (GDPR Art. 8).",
                confidence=1.0)


def r_WD():                       # additive, disjoint from r_PC -> should admit
    L = _g([("CP", "ConsentProcess")], [])
    R = _g([("CP", "ConsentProcess"), ("W", "Withdrawal")],
           [("e2", "CP", "W", "permits")])
    nac = _g([("CP", "ConsentProcess"), ("Wn", "Withdrawal")],
             [("na", "CP", "Wn", "permits")])
    return Rule("r_WD", L, {"CP"}, set(), R, [nac],
                phi=lambda A, J: J == "EU",
                template="You may withdraw your consent at any time (GDPR Art. 7(3)).",
                confidence=1.0)


def r_BAD():                      # deletes ConsentProcess: conflicts with r_PC
    # The deliberately bad candidate of the worked example. Its L is a copy of
    # r_PC's under the identical guard, so the two are symmetrically
    # incompatible and whichever is admitted first survives. Confidence 0.0
    # makes it the last candidate considered, so certification rejects r_BAD
    # rather than r_PC.
    L = _g([("DS", "DataSubject"), ("CP", "ConsentProcess")],
           [("e1", "DS", "CP", "subjectOf")])
    R = _g([("DS", "DataSubject")], [])
    return Rule("r_BAD", L, {"DS"}, set(), R, [],
                phi=lambda A, J: A < 16 and J == "EU",
                confidence=0.0)


def r_SALE():                     # subtractive: delete unlawful third-party sale
    L = _g([("OP", "Operator"), ("TP", "ThirdParty")],
           [("e1", "OP", "TP", "sellsTo")])
    R = _g([("OP", "Operator"), ("TP", "ThirdParty")], [])
    return Rule("r_SALE", L, {"OP", "TP"}, set(), R, [],
                phi=lambda A, J: A < 16 and J in ("EU", "US"),
                template="We do not sell or share your personal information with "
                         "third parties for their own marketing purposes.",
                confidence=1.0)


def r_RET():                      # substitutive: bound an indefinite retention
    L = _g([("RC", "RetentionClause"), ("DI", "DurationIndefinite")],
           [("e1", "RC", "DI", "hasDuration")])
    R = _g([("RC", "RetentionClause"), ("DB", "DurationBounded")],
           [("e2", "RC", "DB", "hasDuration")])
    nac = _g([("RC", "RetentionClause"), ("DBn", "DurationBounded")],
             [("na", "RC", "DBn", "hasDuration")])
    return Rule("r_RET", L, {"RC"}, set(), R, [nac],
                phi=lambda A, J: J == "EU",
                template="We retain it for no longer than 24 months, after which "
                         "it is erased (GDPR Art. 5(1)(e)).",
                confidence=1.0)


# --------------------------------------------------------------------------- #
# Seeded mechanism-validation fixtures: adversarial rules authored to exercise
# individual certification checks, not outputs of the extractor. The full
# pipeline rejects r_BAD and r_RET_BAD (confluence) and one of r_FLIP1 and
# r_FLIP2 (stratification); the ablations admit them. r_NOTIFY produces
# a multi-step cascade (cascade depth > 1): r_RET creates a DurationBounded
# node, which is what r_NOTIFY's left-hand side needs, so r_NOTIFY becomes
# matchable only after r_RET has fired (a creation dependency r_RET ~> r_NOTIFY).
# A purely additive library such as r_PC and r_WD alone cannot cascade, because
# nothing one rule adds is a precondition for another.
# --------------------------------------------------------------------------- #
def r_NOTIFY():                   # additive; matchable only after r_RET fires
    L = _g([("RC", "RetentionClause"), ("DB", "DurationBounded")],
           [("e1", "RC", "DB", "hasDuration")])
    R = _g([("RC", "RetentionClause"), ("DB", "DurationBounded"), ("N", "Notification")],
           [("e1", "RC", "DB", "hasDuration"), ("e2", "RC", "N", "notifies")])
    nac = _g([("RC", "RetentionClause"), ("DB", "DurationBounded"), ("Nn", "Notification")],
             [("e1", "RC", "DB", "hasDuration"), ("na", "RC", "Nn", "notifies")])
    return Rule("r_NOTIFY", L, {"RC", "DB"}, {"e1"}, R, [nac],
                phi=lambda A, J: J == "EU",
                template="We will notify you that your data's retention period "
                         "is now bounded to 24 months.",
                confidence=1.0)


def r_RET_BAD():                  # deletes DurationIndefinite; conflicts with r_RET
    # Mirrors r_BAD's relationship to r_PC, but for retention. r_BAD and r_PC
    # share the same guard, so whichever is admitted first always fires first
    # and admitting r_BAD without the confluence check changes no observable
    # outcome. r_RET_BAD has a wider guard (EU, UK) than r_RET (EU only), so on
    # a UK host it fires uncontested and admitting it without the confluence
    # check produces a visible difference regardless of candidate order.
    L = _g([("RC", "RetentionClause"), ("DI", "DurationIndefinite")],
           [("e1", "RC", "DI", "hasDuration")])
    R = _g([("RC", "RetentionClause")], [])
    return Rule("r_RET_BAD", L, {"RC"}, set(), R, [],
                phi=lambda A, J: J in ("EU", "UK"),
                confidence=0.0)   # adversarial fixture, as for r_BAD


# Two ping-pong rules that loop under a naive size measure but are rejected by
# stratification; they exercise the no-stratification ablation.
def r_FLIP1():
    L = _g([("X", "TagA")], [])
    R = _g([("X", "TagB")], [])           # relabel via delete+add, size unchanged
    return Rule("r_FLIP1", L, set(), set(), R, [], phi=lambda A, J: True)


def r_FLIP2():
    L = _g([("X", "TagB")], [])
    R = _g([("X", "TagA")], [])
    return Rule("r_FLIP2", L, set(), set(), R, [], phi=lambda A, J: True)


def synthesize():
    """Worked-example candidate stream (highest confidence first).

    Offline stand-in for extractor output, for smoke tests only.
    """
    return [r_PC(), r_WD(), r_SALE(), r_RET(), r_BAD()]


