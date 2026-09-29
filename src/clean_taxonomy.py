"""Preprocess the Dallas 311 taxonomy data into a normalized service list.

Input : data/d7e7-envw.json  (department, service_request_type, count)
Output: data/taxonomy_clean.csv (department, service, count)
"""
import json, re, sys
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/d7e7-envw.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/taxonomy_clean.csv"

df = pd.DataFrame(json.load(open(SRC)))
df["count"] = pd.to_numeric(df["count"])
print(f"raw: {len(df)} rows, {df.service_request_type.nunique()} types, {df.department.nunique()} departments")

# Step 1. drop rows with missing type
df = df.dropna(subset=["service_request_type"])

# Step 2. filter test records
df = df[~df.service_request_type.str.contains(r"TEST CRM|CSR Test", case=False, regex=True)]

# Step 3. normalize service name: strip trailing department code ("for example - TPW")
df["service"] = df.service_request_type.str.replace(r"\s*[-–]\s*[A-Z&/ ]{2,12}$", "", regex=True).str.strip()

# Step 4. normalize/merge department labels
DEPT_MAP = {
    "Sanitation Services": "Sanitation",
    "Dallas Water Utilities": "Water Utilities",
    "Transportation and Public Works Department": "Transportation and Public Works",
    "Transportation": "Transportation and Public Works",
    "Convention & Event Services": "Convention and Event Services",
}
df["department"] = df.department.replace(DEPT_MAP)

# Step 5. aggregate duplicates created by Step 3 and Step 4
out = (df.groupby(["department", "service"], as_index=False)["count"].sum()
         .sort_values("count", ascending=False))

# Step 6. Output the cleaned taxonomy
print(f"clean: {len(out)} services, {out.department.nunique()} departments")
print(f"services with >=50 requests: {(out['count'] >= 50).sum()}")
out.to_csv(DST, index=False)
