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
