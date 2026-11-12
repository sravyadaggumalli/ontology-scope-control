"""Assign each question to the tuning or the test split, once, with a fixed seed.

Rules:
  - every question from the pilot file (questions_v0.csv) goes to tune, because
    every design decision so far was made while looking at them
  - the remaining questions in each group are shuffled with the seed and the
    first --tune-per-group go to tune, the rest to test
  - a row that already has a split keeps it, so running this again never moves
    a question between splits; use --reset to start over (only before any
    test-set run exists)

Usage:
    python src/split.py [eval/questions_v1.csv] [--seed 20261111] [--tune-per-group 10]

    Shuffle uses numpy.random.default_rng(seed).shuffle, https://numpy.org/doc/stable/reference/random/generator.html
"""
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="eval/questions_v1.csv")
    ap.add_argument("--pilot", default="eval/questions_v0.csv")
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--tune-per-group", type=int, default=10)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    q = pd.read_csv(a.questions)
    q["split"] = q["split"].fillna("").astype(str)
    if a.reset:
        q["split"] = ""
    pilot_ids = set(pd.read_csv(a.pilot).id)
    q.loc[q.id.isin(pilot_ids) & (q.split == ""), "split"] = "tune"

    rng = np.random.default_rng(a.seed)
    for g, sub in q.groupby("group"):
        todo = sub[sub.split == ""].index.to_numpy().copy()
        rng.shuffle(todo)
        have_tune = (sub.split == "tune").sum()
        n_tune = max(0, a.tune_per_group - have_tune) if have_tune < a.tune_per_group else 0
        q.loc[todo[:n_tune], "split"] = "tune"
        q.loc[todo[n_tune:], "split"] = "test"

    q.to_csv(a.questions, index=False)
    print(f"seed {a.seed}")
    print(pd.crosstab(q.group, q.split).to_string())
    print(f"\ntotal: tune {int((q.split=='tune').sum())}, test {int((q.split=='test').sum())}")

if __name__ == "__main__":
    main()
