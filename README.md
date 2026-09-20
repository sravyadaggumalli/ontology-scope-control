# Ontology-Based Scope Control for Conversational Agents

CSC789 AI Capstone (Track A, Research), DSU, Fall 2026
Student: Sravya Daggumalli

## Objective

Test whether an ontology with inherited permissions gives better conversational
scope control for a city 311 assistant than a prose system prompt, a blocklist,
or a flat topic list, on the same set of questions. The question is whether  hierarchy and inheritance help, and how much is lost when questions are matched to concepts automatically.

## Approaches compared

| ID | Approach | LLM used |
|----|----------|----------|
| A | Prose system prompt | yes |
| B | Prose prompt + explicit blocklist | yes |
| R | Flat list of the ontology's concepts, no hierarchy, embedding match | no |
| C | Ontology with inherited permissions, embedding match | no |
| D | Same ontology, gold concept supplied (diagnostic upper bound) | no |

All approaches return `allow` or `refuse` with a short reason. No chat interface
or answer generation is in the scope of the project

## Data

Dallas 311 Service Requests, Oct 2020 to present (Dallas OpenData, dataset
`d7e7-envw`), used in aggregate only: department and request-type counts pulled
with one grouped query. No individual records, addresses, or locations. See
`data/NOTE.md` for the query and normalization rules.

Out-of-scope boundary cases are written from public 311 documentation. The scope
model is a fictional city, not Dallas's actual policy.

## Evaluation set

160 questions in four groups of 40 (direct in-scope, inheritance-dependent
in-scope, paraphrased in-scope, out-of-scope), with a subset written
independently by someone with no knowledge of the ontology, and deliberately
hard out-of-scope cases (near-misses and ambiguous questions). Each question is
labeled allow/refuse plus its concept; a second person labels 40 independently.
10 per group are held out for tuning, 30 per group form the fixed test set.

If time permits a reduced version will be repeated on a second domain to check generality.

## Main tools

Python 3, Protege (OWL), owlready2 (HermiT/Pellet), sentence-transformers,
a small hosted LLM for the prompt baselines.

## Layout

    docs/       proposal, progress reports
    data/       311 Dallas OpenData and cleanup notes
    ontology/   OWL files
    src/        matching, permission lookup, baselines
    eval/       question set, labels, results
