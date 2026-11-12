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

`eval/questions_v0.csv` is the original 42-question pilot and used for every design decision so far, therefore it is the tuning split. `eval/questions_v1.csv`
is the full set of 196 questions: columns id, group, question, label, concept, source
(author or independent), split (tune or test). 132 were drafted as candidates from the
Dallas service list ranked by request volume, so every permitted concept is covered,
then reviewed and labeled by the author; 22 were written independently by a family
member without seeing the concept list. The 154 non-pilot questions are the test split,
assigned once by `src/split.py` with seed 20261111 and committed before any test run.
Nothing is tuned on the test split. A second labeler on a sample is pending.

## Main tools

Python 3, Protege (OWL), owlready2 (HermiT/Pellet), sentence-transformers,
a small hosted LLM for the prompt baselines.

## Layout

    docs/       proposal, progress reports
    data/       311 Dallas OpenData and cleanup notes
    ontology/   scope_v1.owl (pilot), scope_v2.owl (current), both generated; overrides.csv (hand-written exceptions)
    src/        matching, permission lookup, baselines, ontology build and runs
    eval/       question set, labels, results

## Running

    pip install -r requirements.txt     # owlready2 needs Java for the reasoner
    python src/clean_taxonomy.py        # data/taxonomy_clean.csv
    python src/build_ontology.py        # ontology/scope_v2.owl from eval/concepts_v2.csv
    python src/scope_ontology.py        # startup check and tree print
    python src/scope_ontology.py --plant   # same, with five planted conflicts
    python src/check_questions.py       # sanity checks on eval/questions_v1.csv
    python src/split.py                 # assign tune/test once, fixed seed (already done)

    # all five approaches on the full set (API key needed for A, B and the LLM matcher)
    for i in 1 2 3; do python src/baseline_prompt.py eval/questions_v1.csv --out eval/results_baseline_A_run$i.csv; done
    for i in 1 2 3; do python src/baseline_prompt.py eval/questions_v1.csv --blocklist --out eval/results_baseline_B_run$i.csv; done
    python src/baseline_flat.py eval/questions_v1.csv --out eval/results_baseline_R.csv
    python src/run_ontology.py --arm C eval/questions_v1.csv
    python src/run_ontology.py --arm C eval/questions_v1.csv --matcher llm --out eval/results_ontology_C_llm_run1.csv
    python src/run_ontology.py --arm D eval/questions_v1.csv

    python src/compare.py --split test --ref C-llm eval/results_*.csv   # table, McNemar, docs/figures/results_test.png
    python src/plot_results.py          # pilot figure (docs/figures/baseline_results.png)
    python src/baseline_prompt.py eval/questions_v0.csv --blocklist --plant   # conflict experiment for the prompt (planned)

## Ontology (arm C)

`build_ontology.py` generates the OWL file from the cleaned Dallas taxonomy and
the concept list. `concepts_v1.csv` is the 30-concept pilot list; `concepts_v2.csv`
adds nine high-volume services not in the pilot(parking, line locate,
encampments, animal nuisance and bites, noise, parks, construction, street
lighting), improves coverage from 68% to 86% of all requests (73 of 270 services). The pilot
results in eval/ were produced with v1; the full evaluation uses v2. The tree is ScopeConcept, then two disjoint branches:
CityServiceRequest (tagged permit) with one class per Dallas department and one
leaf per concept, and NotHandled (tagged deny) with the refused concepts. A
concept gets a Hazardous subclass when Dallas has a hazardous variant of the
request type. Permission is an annotation on the two branch roots only; every
other class inherits it, except the exceptions in `ontology/overrides.csv`.

`scope_ontology.py` loads the file, runs HermiT once at startup to check the
hierarchy, checks that no class carries two tags, and answers
permit/deny for a concept by walking up from the concept until a tag is found.
