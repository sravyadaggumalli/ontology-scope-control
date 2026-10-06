"""Baseline R: flat list of permitted concepts matched by embedding similarity.

# Embedding + cosine similarity pattern is based on the sentence-transformers
# semantic search example:
#   https://www.sbert.net/examples/applications/semantic-search/README.html
# Model Used: BAAI/bge-small-en-v1.5, https://huggingface.co/BAAI/bge-small-en-v1.5

No LLM. Each concept has a label plus a few synonyms; the question is embedded
and compared against every concept description. Above the threshold -> allow
(if the concept is permitted) or refuse (if denied); below -> refuse.

Usage:
    python src/baseline_flat.py eval/questions_v0.csv [--threshold 0.55] [--concepts eval/concepts_v1.csv]
Output:
    eval/results_baseline_R.csv and a per-group metrics table on stdout
"""
import argparse, sys
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# Concept list is loaded from a CSV (concept, permitted, description).
#   eval/concepts_v0.csv  pilot list, descriptions written by hand
#   eval/concepts_v1.csv  descriptions taken from Dallas service names only

def metrics(df):
    rows = []
    for g, sub in df.groupby("group", sort=False):
        oos = sub[sub.label == "refuse"]; ins = sub[sub.label == "allow"]
        rows.append({"group": g, "n": len(sub),
                     "out-of-scope allowed": f"{(oos.pred=='allow').sum()}/{len(oos)}" if len(oos) else "-",
                     "in-scope refused":     f"{(ins.pred=='refuse').sum()}/{len(ins)}" if len(ins) else "-",
                     "unmatched": int((sub.matched == "NONE").sum())})
    oos = df[df.label == "refuse"]; ins = df[df.label == "allow"]
    rows.append({"group": "all", "n": len(df),
                 "out-of-scope allowed": f"{(oos.pred=='allow').sum()}/{len(oos)}",
                 "in-scope refused": f"{(ins.pred=='refuse').sum()}/{len(ins)}",
                 "unmatched": int((df.matched == "NONE").sum())})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="eval/questions_v0.csv")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--concepts", default="eval/concepts_v0.csv")
    ap.add_argument("--out", default="eval/results_baseline_R.csv")
    a = ap.parse_args()

    concepts = pd.read_csv(a.concepts)
    m = SentenceTransformer(a.model)
    labels = concepts.concept.tolist()
    permitted = dict(zip(concepts.concept, concepts.permitted.astype(bool)))
    cvec = m.encode([f"{c}: {d}" for c, d in zip(concepts.concept, concepts.description)],
                    normalize_embeddings=True)

    df = pd.read_csv(a.questions)
    qvec = m.encode(df.question.tolist(), normalize_embeddings=True)
    sims = qvec @ cvec.T
    best = sims.argmax(1); score = sims.max(1)
    df["matched"] = [labels[i] if s >= a.threshold else "NONE" for i, s in zip(best, score)]
    df["score"] = score.round(3)
    df["pred"] = ["allow" if (c != "NONE" and permitted[c]) else "refuse" for c in df.matched]
    df["correct"] = df.pred == df.label
    df.to_csv(a.out, index=False)
    for _, r in df.iterrows():
        print(f"{r.id}  gold={r.label:6s} pred={r.pred:6s} match={r.matched:24s} {r.score:.2f}")
    print(f"\nmodel: {a.model}  concepts: {a.concepts}  threshold: {a.threshold}  accuracy: {df.correct.mean():.2f}\n")
    print(metrics(df).to_string(index=False))
