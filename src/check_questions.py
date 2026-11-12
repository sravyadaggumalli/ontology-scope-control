"""Sanity checks for a question file before it is split or run.

Checks: required columns, valid group names, label is allow/refuse, concept
exists in the concept list (or NONE), label agrees with the concept's permitted
flag, duplicate ids, duplicate or near-duplicate question text, and prints the
count per group, per source, and per concept.

Usage:
    python src/check_questions.py [eval/questions_v1.csv] [--concepts eval/concepts_v2.csv]
"""
import argparse, sys
import pandas as pd

GROUPS = ["direct", "inherit", "paraphrase", "outscope"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="eval/questions_v1.csv")
    ap.add_argument("--concepts", default="eval/concepts_v2.csv")
    a = ap.parse_args()
    q = pd.read_csv(a.questions)
    c = pd.read_csv(a.concepts)
    permitted = dict(zip(c.concept, c.permitted))
    problems = []

    for col in ["id", "group", "question", "label", "concept", "source", "split"]:
        if col not in q.columns:
            problems.append(f"missing column {col}")
    if problems:
        sys.exit("\n".join(problems))

    bad = q[~q.group.isin(GROUPS)]
    problems += [f"{r.id}: unknown group {r.group}" for r in bad.itertuples()]
    bad = q[~q.label.isin(["allow", "refuse"])]
    problems += [f"{r.id}: label must be allow or refuse, got {r.label}" for r in bad.itertuples()]
    for r in q.itertuples():
        if r.concept == "NONE":
            if r.label != "refuse":
                problems.append(f"{r.id}: concept NONE must be refuse")
            continue
        if r.concept not in permitted:
            problems.append(f"{r.id}: concept {r.concept} not in {a.concepts}")
        elif permitted[r.concept] and r.label != "allow":
            problems.append(f"{r.id}: {r.concept} is permitted but label is refuse")
        elif not permitted[r.concept] and r.label != "refuse":
            problems.append(f"{r.id}: {r.concept} is denied but label is allow")
    dup = q[q.id.duplicated(keep=False)]
    problems += [f"duplicate id {i}" for i in sorted(set(dup.id))]
    norm = q.question.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")
    dup = q[norm.duplicated(keep=False)]
    problems += [f"duplicate question text: {r.id}" for r in dup.itertuples()]
    bad = q[~q.source.isin(["author", "independent"])]
    problems += [f"{r.id}: source must be author or independent" for r in bad.itertuples()]

    print(f"{len(q)} questions in {a.questions}")
    print("\nper group:\n" + q.group.value_counts().reindex(GROUPS).fillna(0).astype(int).to_string())
    print("\nper source:\n" + q.source.value_counts().to_string())
    print("\nper concept (in-scope only):")
    print(q[q.label == "allow"].concept.value_counts().to_string())
    missing = [k for k, v in permitted.items() if v and k not in set(q[q.label == "allow"].concept)]
    if missing:
        print("\npermitted concepts with no question:", ", ".join(missing))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nno problems")

if __name__ == "__main__":
    main()
