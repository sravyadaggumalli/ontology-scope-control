# Dataset

**Source:** Dallas OpenData, "311 Service Requests October 1, 2020 to Present",
https://www.dallasopendata.com/Services/311-Service-Requests/d7e7-envw

**Date Pulled** 2026-09-19

**Query Used:**

    https://www.dallasopendata.com/resource/d7e7-envw.json?$select=department,service_request_type,count(*)&$group=department,service_request_type&$order=count%20DESC&$limit=5000

**File:** `d7e7-envw.json` (435 rows; fields `department`, `service_request_type`, `count`)

## File Contents

- 435 rows (department x request-type pairs)
- 2,644,595 requests in total since Oct 2020

## Preprocessing

Normalization steps:

1. Drop rows with any null service_request_type.
2. Filter out test records (TEST CRM, CSR Test).
3. Normalize service names by stripping the trailing department code
   (- TPW, - PBW, etc.).
4. Normalize department labels: Sanitation Services -> Sanitation,
   Dallas Water Utilities -> Water Utilities,
   Transportation and Public Works -> Transportation and Public Works.
5. Group by (department, service) and sum the counts.

This should give us approx. 200 unique service names.

## Notes

- "Code Concern - CCS" is a catch-all with about 27% of all requests. Its
  subtypes are not in the data and can be taken from the Code Compliance
  public pages.
- Department is the only hierarchy in the data. The mid-level grouping of
  services into concepts will be authored.

## Usage terms

Public open data; check Licensing and Attribution on Dallas OpenData website for the terms of use. 

## (Update 09-28-2026)

### After normalization 
services: 270, departments: 23, requests: 2,643,848
services with >= 50 requests: 201
services with <  10 requests: 58
largest single service: 'Code Concern' = 703,947 (26.6% of all)

## Pilot questions

eval/questions_v0.csv: 42 questions written and labeled by me.
11 direct, 10 inheritance, 10 paraphrase, 11 out-of-scope.
Full 160-question set ~ Week 6.

## Baseline runs

1) Arm R, flat list: bge-small-en-v1.5, threshold 0.55, one run.
   - in-scope refused 0/31, out-of-scope allowed 5/11 (all near misses).
2) Arm A, prose prompt: gpt-6-luna, one run. Model rejects temperature=0, so
  runs are not deterministic; final experiment will use 3 runs, majority vote.
   - in-scope refused 5/31 (all hazards treated as emergencies),
  out-of-scope allowed 1/11 (Plano pothole).


## (Update 10-05-2026)

## Baseline runs

| Run | Model / concepts | Threshold | Out-of-scope allowed | In-scope refused | Concept correct |
|---|---|---|---|---|---|
| R | bge-small, concepts_v0 (hand-written) | 0.55 | 5/11 | 0/31 | 30/31 |
| R_v1 | bge-small, concepts_v1 (taxonomy names) | 0.55 | 6/11 | 1/31 | 19/31 |
| R_t065 | bge-small, concepts_v0 | 0.65 | 5/11 | 1/31 | 30/31 |
| A_run1 | gpt-6-luna | default | 1/11 | 6/31 | - |
| A_run2 | gpt-6-luna | default | 0/11 | 6/31 | - |
| A_run3 | gpt-6-luna | default | 1/11 | 4/31 | - |
| A majority | gpt-6-luna, 3 runs | default | 1/11 | 6/31 | - |

Prompt model rejects temperature=0, so runs are at the default; 3 of 42
verdicts changed across runs (G3-01, G3-05, G4-03).

Notes:

- The flat list and the prompt fail in opposite directions. The flat list
  allowed everything in scope but also let 5 of the 11 out-of-scope questions
  through. The prompt caught 10 of 11 out-of-scope but refused 6 in-scope
  questions.

- All 5 flat-list misses are near misses: the question uses 311 words but is
  not a 311 request (neighbor's tree, pothole in Plano, dog-bite lawsuit, dog
  attack in progress, sprinkler smell). Each one matched a permitted concept
  with a score between 0.66 and 0.73. Similarity matching gets the topic
  right and misses the location, intent, or responsibility.

- All 6 prompt refusals are hazards that the model escalated to "call 911":
  guardrail, limb hitting trucks, chained dog, pothole with cars hitting it,
  aggressive dog, smashed railing. Dallas 311 has request types for all of
  these. G1-01 (plain pothole) was allowed every run; G3-01 (same pothole,
  "cars keep hitting it") was refused. The wording triggers it, not the topic.

- R_v1 (descriptions from Dallas service names only) changed the verdicts
  very little (6/11, 1/31) but 12 of 31 in-scope questions matched the wrong
  concept, e.g. illegal dumping -> AlleyRepair, bent stop sign ->
  StreetSpillageDebris, tap water smell -> WaterMainLeak. The verdict stayed
  right because most concepts are permitted, so a wrong match usually still returns "allow". Concept accuracy tells the real story: 30/31 with my own hand-written descriptions vs 19/31 with the city's own labels. The only difference between the two runs is the description text, so the hand-written synonyms, which used the same everyday words as the questions, were doing the matching. That is the leakage, and it is why concepts_v1 is the list we are going to use from now on.

- Threshold 0.65 did not help much. Near misses score 0.66-0.73, above some real
  requests (median mowing, 0.60). The 5 correct refusals at 0.55 came from
  matching a denied concept; at 0.65 they came from no match at all.

- Prompt runs are not repeatable like expected: 3 of 42 verdicts flipped (G3-01, G3-05,
  G4-03). Reasons also vary between runs with the same verdict, and G1-11's
  reason was "cloudy tap water" in all three runs for a question about smell.

- Both baselines let the Plano pothole through by majority.

Results: eval/results_baseline_*.csv. Figure: docs/figures/baseline_results.png
(src/plot_results.py).