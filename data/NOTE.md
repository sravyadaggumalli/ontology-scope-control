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
