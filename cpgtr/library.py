"""Seed candidate rules -- the stand-in for the Stage-1 LLM extractor output.

REPLACE `synthesize()` with a real extractor: prompt a frozen LLM under the TG
schema, parse to Rule objects, then feed them to certify.admit unchanged.
"""
from .graph import Graph
from .rules import Rule




class _Library:
    def synthesize(self):
        # Replace `...` with toy rules in whatever shape admit() expects.
        # Kept deliberately tiny and obviously synthetic.
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
    # Deliberately the lowest-confidence fixture in this library: r_BAD's L
    # is a strict copy of r_PC's, under the identical guard, so the two are
    # symmetrically incompatible -- whichever is admitted first survives, the
    # other is rejected as non-joinable. There is no structural sense in
    # which r_BAD is "more wrong" than r_PC other than the narrative role it
    # plays (CP-GTR_V2.tex's Box 2/3 worked example: certification must
    # reject the deliberately-injected bad candidate). confidence=0.0 encodes
    # exactly that role via Algorithm 3's own admission-order mechanism,
    # rather than relying on synthesize()'s list literal happening to place
    # it last (see this module's synthesize() docstring, which predates the
    # confidence field but already documented "highest confidence first").
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


def r_RET_BAD():                  # deletes DurationIndefinite outright: conflicts with r_RET
    # Deliberately mirrors r_BAD's relationship to r_PC (Box 3), but for
    # retention instead of consent, and with one addition that matters
    # operationally: r_BAD's LHS is a strict copy of r_PC's, so whichever of
    # the two is admitted first always fires first on any host and the other
    # never gets a chance to run -- meaning admitting r_BAD under --CPA
    # produces no OBSERVABLE difference on hosts where r_PC is also present.
    # r_RET_BAD instead carries a WIDER guard (EU, UK) than r_RET (EU only),
    # so on a UK host it fires completely uncontested -- no ordering trick
    # needed, and the --CPA/full-pipeline divergence is real and reproducible
    # regardless of candidate order.
    L = _g([("RC", "RetentionClause"), ("DI", "DurationIndefinite")],
           [("e1", "RC", "DI", "hasDuration")])
    R = _g([("RC", "RetentionClause")], [])
    return Rule("r_RET_BAD", L, {"RC"}, set(), R, [],
                phi=lambda A, J: J in ("EU", "UK"),
                confidence=0.0)   # deliberately-injected adversarial fixture, same
                                  # reasoning as r_BAD's confidence=0.0 above


# Two ping-pong rules that LOOP under a naive size measure but are rejected by
# stratification -- used to make the --Strat ablation produce a real incident.
def r_FLIP1():
    L = _g([("X", "TagA")], [])
    R = _g([("X", "TagB")], [])           # relabel via delete+add, size unchanged
    return Rule("r_FLIP1", L, set(), set(), R, [], phi=lambda A, J: True)


def r_FLIP2():
    L = _g([("X", "TagB")], [])
    R = _g([("X", "TagA")], [])
    return Rule("r_FLIP2", L, set(), set(), R, [], phi=lambda A, J: True)


def synthesize():
    """Extractor stand-in: ordered candidate stream (highest confidence first)."""
    return [r_PC(), r_WD(), r_SALE(), r_RET(), r_BAD()]


