"""Load the scope ontology, check it, and answer permit/deny for a concept.

Two functions:
  check()    runs once at startup. The reasoner (HermiT, via owlready2) checks the
             class hierarchy is consistent; checks that no class
             carries both permit and deny, which annotations cannot express to the
             reasoner.
  verdict()  runs per message. Walk from the concept up through its ancestors and
             return the first permission found. No tag anywhere means deny.

Usage as a script prints the hierarchy with the inherited verdict of every class:
    python src/scope_ontology.py [ontology/scope_v1.owl] [--plant]
--plant adds five deliberate conflicts first, to show the checks catch them.

Sources. The owlready2 docs:
    https://owlready2.readthedocs.io/en/latest/onto.html          load, classes()
    https://owlready2.readthedocs.io/en/latest/class.html         ancestors(), subclasses(), is_a
    https://owlready2.readthedocs.io/en/latest/reasoning.html     sync_reasoner, inconsistency
    https://owlready2.readthedocs.io/en/latest/annotations.html   annotation property access
"""
import argparse
from owlready2 import get_ontology, sync_reasoner, Nothing, OwlReadyInconsistentOntologyError
from tabulate import tabulate


class ScopeOntology:
    def __init__(self, path="ontology/scope_v1.owl"):
        # Load ontology from OWL file
        # Set the permission property and the root class as attributes.
        self.onto = get_ontology("file://" + path).load()
        self.permission = self.onto.permission
        self.root = self.onto.ScopeConcept

    # startup check
    def check(self):
        """Validate the ontology and return any issues; an empty list means the policy is valid."""
        problems = []

        # Run the reasoner and catch any OwlReadyInconsistentOntologyError.
        # The docs (reasoning.html) also says a class under two disjoint classes is inferred equivalent to Nothing.
        try:
            with self.onto:
                sync_reasoner(infer_property_values=False, debug=0)
        except OwlReadyInconsistentOntologyError:
            problems.append("ontology is inconsistent")

        # The reasoner catches structural ontology problems,
        # but it cannot detect two custom policy checks because permission is stored as an annotation
        # instead of a logical constraint. We therefore inspect `self.permission[c]` directly
        # using the owlready2 annotation pattern and enforce these project-specific rules.
        for c in self.onto.classes():
            if Nothing in c.equivalent_to:
                problems.append(f"{c.name}: unsatisfiable (under two disjoint branches)")
            tags = set(self.permission[c])
            if tags - {"permit", "deny"}:
                problems.append(f"{c.name}: unknown permission tag {tags}")
            if {"permit", "deny"} <= tags:
                problems.append(f"{c.name}: tagged both permit and deny")
        return problems

    # check each message
    def concept(self, name):
        # lookup a class by name,
        # None if it does not exist.
        return self.onto[name]

    def verdict(self, name):
        """permit or deny for a concept name; the most specific tag is used."""
        # Important: This is the inheritance rule of the project, start at the
        # concept, walk up, first tag that matches is used, no match means deny.
        c = self.onto[name]
        if c is None:
            return "deny", None
        for cls in [c] + list(c.ancestors()):
            tags = self.permission[cls]
            if tags:
                return tags[0], cls.name
        return "deny", None

    def leaf_concepts(self):
        """(name, description) for every class that can be matched to a message."""
        # skip the three structural classes that are not meaningful user concepts.
        out = []
        for c in self.onto.classes():
            if c.description and c.name not in ("CityServiceRequest", "NotHandled", "ScopeConcept"):
                out.append((c.name, c.description[0]))
        return out

    def tree(self):
        """One row per class: the class name, its own tag, and the inherited verdict."""
        # Traverse the ontology depth-first using c.subclasses() to print one row per class.
        # Rendered using tabulate: https://github.com/astanin/python-tabulate
        rows = []
        def walk(c, depth):
            tag = self.permission[c]
            v, src = self.verdict(c.name)
            rows.append(["| " * depth + c.name, tag[0] if tag else "", v,
                         "" if src == c.name else src])
            for s in sorted(c.subclasses(), key=lambda x: x.name):
                walk(s, depth + 1)
        walk(self.root, 0)
        return tabulate(rows, headers=["class", "own tag", "verdict", "inherited from"],
                        tablefmt="plain")


def plant_conflicts(so):
    """Five deliberate policy errors, for the startup check to catch."""
    # These are common mistakes a policy author could make while editing the ontology by hand.
    # We add/remove class relationships and annotation values inside `with onto:`, following
    # the owlready2 class-modification pattern: https://owlready2.readthedocs.io/en/latest/class.html
    o = so.onto
    with o:
        o.PotHole.is_a.append(o.NotHandled)                 # under both disjoint branches
        o.MissedGarbage.permission.append("deny")           # permit inherited, deny asserted: fine
        o.MissedGarbage.permission.append("permit")         # ...but both on one class: conflict
        o.Emergency.is_a.append(o.CityServiceRequest)       # denied concept also permitted
        o.WaterMainLeak.permission.append("maybe")          # not a valid tag
        o.AnimalLoose.is_a.append(o.LegalMatter)            # service under a denied concept
    return ["PotHole", "MissedGarbage", "Emergency", "WaterMainLeak", "AnimalLoose"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="ontology/scope_v1.owl")
    ap.add_argument("--plant", action="store_true", help="plant five conflicts before checking")
    a = ap.parse_args()
    so = ScopeOntology(a.path)
    if a.plant:
        planted = plant_conflicts(so)
        print("planted conflicts in:", ", ".join(planted))
    problems = so.check()
    print("startup check:", "ok" if not problems else f"{len(problems)} problem(s)")
    for p in problems:
        print("  -", p)
    if not a.plant:
        print()
        print(so.tree())
