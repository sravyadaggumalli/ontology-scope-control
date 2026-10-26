"""Build the scope ontology (arm C) from the Dallas taxonomy and the concept list.

Hierarchy:
    ScopeConcept
      CityServiceRequest   [permit]
        <Department>           one class per Dallas department that owns a concept
          <Concept>                leaf, description = Dallas service names
            <Concept>Hazardous     only if Dallas has a "- Hazardous" variant; inherits permit
      NotHandled           [deny]
        Emergency, LegalMatter, PrivatePropertyDispute, OtherJurisdiction, OtherAgency

Permission is an annotation property on classes. Only the two top-level classes
carry a tag; everything else inherits it, except the overrides listed in
ontology/overrides.csv (concept, parent, permission, description).

Inputs:  data/taxonomy_clean.csv, eval/concepts_v1.csv, ontology/overrides.csv
Output:  ontology/scope_v1.owl

Usage:
    python src/build_ontology.py
"""
import re, types
import pandas as pd
from owlready2 import get_ontology, Thing, AnnotationProperty, AllDisjoint

IRI = "http://example.org/scope311#"

def camel(s):
    return re.sub(r"[^A-Za-z0-9]+", " ", s).title().replace(" ", "")

def main():
    tax = pd.read_csv("data/taxonomy_clean.csv")
    concepts = pd.read_csv("eval/concepts_v1.csv")
    overrides = pd.read_csv("ontology/overrides.csv")

    # department and request count per Dallas service name
    dept_of = {s.strip().lower(): d for d, s in zip(tax.department, tax.service)}
    count_of = {s.strip().lower(): n for s, n in zip(tax.service, tax["count"])}

    onto = get_ontology(IRI)
    with onto:
        class ScopeConcept(Thing): pass
        class CityServiceRequest(ScopeConcept): pass
        class NotHandled(ScopeConcept): pass
        class permission(AnnotationProperty): pass
        class description(AnnotationProperty): pass
        class source(AnnotationProperty): pass
        AllDisjoint([CityServiceRequest, NotHandled])
        CityServiceRequest.permission = ["permit"]
        NotHandled.permission = ["deny"]

        departments = {}
        def department_class(name):
            if name not in departments:
                departments[name] = types.new_class(camel(name), (CityServiceRequest,))
                departments[name].description = [name]
                departments[name].source = ["dallas"]
            return departments[name]

        classes = {}
        for _, r in concepts.iterrows():
            names = [n.strip() for n in r.description.split(";")]
            if r.permitted:
                # parent = the department that owns most of this concept's requests
                owners = {}
                for n in names:
                    d = dept_of.get(n.lower())
                    if d:
                        owners[d] = owners.get(d, 0) + count_of.get(n.lower(), 0)
                parent = department_class(max(owners, key=owners.get)) if owners else CityServiceRequest
                routine = [n for n in names if "hazardous" not in n.lower()]
                hazardous = [n for n in names if "hazardous" in n.lower()]
                cls = types.new_class(r.concept, (parent,))
                cls.description = ["; ".join(routine or names)]
                cls.source = ["dallas"]
                if hazardous:
                    hz = types.new_class(r.concept + "Hazardous", (cls,))
                    hz.description = ["; ".join(hazardous)]
                    hz.source = ["dallas"]
                    classes[hz.name] = hz
            else:
                cls = types.new_class(r.concept, (NotHandled,))
                cls.description = [r.description]
                cls.source = ["authored"]
            classes[r.concept] = cls

        for _, r in overrides.iterrows():
            parent = classes.get(r.parent) or onto[r.parent]
            cls = types.new_class(r.concept, (parent,))
            cls.description = [r.description]
            cls.source = ["authored"]
            if isinstance(r.permission, str) and r.permission:
                cls.permission = [r.permission]
            classes[r.concept] = cls

    onto.save(file="ontology/scope_v1.owl", format="rdfxml")
    n = len(list(onto.classes()))
    print(f"wrote ontology/scope_v1.owl: {n} classes, {len(departments)} departments, "
          f"{sum(1 for c in classes.values() if c.name.endswith('Hazardous'))} hazardous leaves, "
          f"{len(overrides)} overrides")

if __name__ == "__main__":
    main()
