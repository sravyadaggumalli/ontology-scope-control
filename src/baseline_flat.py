"""Baseline R: flat list of permitted concepts matched by embedding similarity.

# Embedding + cosine similarity pattern is based on the sentence-transformers
# semantic search example:
#   https://www.sbert.net/examples/applications/semantic-search/README.html
# Model Used: BAAI/bge-small-en-v1.5, https://huggingface.co/BAAI/bge-small-en-v1.5

No LLM. Each concept has a label plus a few synonyms; the question is embedded
and compared against every concept description. Above the threshold -> allow
(if the concept is permitted) or refuse (if denied); below -> refuse.

Usage:
    python src/baseline_flat.py eval/questions_v0.csv [--threshold 0.55]
Output:
    eval/results_baseline_R.csv and a per-group metrics table on stdout
"""
import argparse, sys
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# Flat concept list: same concepts the ontology will have, no hierarchy.
# (label, permitted?, description used for matching)
CONCEPTS = [
    ("PotHole", True, "pothole; hole in the road; road surface damage"),
    ("StreetRepair", True, "street repair; broken pavement; road damage"),
    ("AlleyRepair", True, "alley repair; broken concrete in alley"),
    ("GuardrailRepair", True, "guardrail repair; bent railing on bridge or overpass"),
    ("TrafficSignalFlashing", True, "traffic signal flashing red; traffic light malfunction; blinking light at intersection"),
    ("TrafficSignMaintenance", True, "traffic sign damaged; stop sign bent, missing, or obscured"),
    ("GraffitiTrafficSignal", True, "graffiti on traffic signal or sign; spray paint on public property"),
    ("TreeDownLowLimbs", True, "tree down; low hanging limb over street; branch blocking road"),
    ("StreetSpillageDebris", True, "spilled gravel, dirt or debris in road; material dropped from truck"),
    ("MedianMaintenance", True, "median not mowed; right of way maintenance; overgrown median"),
    ("MissedGarbage", True, "missed garbage pickup; trash not collected"),
    ("MissedRecycle", True, "missed recycling pickup"),
    ("MissedBrushBulk", True, "missed bulk trash or brush pickup; bulk pile still on curb"),
    ("RollCartMaintenance", True, "roll cart replacement; broken trash or recycling bin lid; new cart"),
    ("DeadAnimalPickup", True, "dead animal in street; deceased animal removal"),
    ("IllegalDumping", True, "illegal dumping; couch, mattress, tires dumped in alley or street"),
    ("StormDrainCleaning", True, "storm drain clogged; inlet stopped up; street flooding when it rains"),
    ("CreekCulvertBlockage", True, "creek or culvert blocked with debris; creek backing up"),
    ("WaterMainLeak", True, "water main leak or break; water running down street"),
    ("WaterQuality", True, "tap water smells or tastes bad; rotten egg smell in water; discolored water"),
    ("AnimalLoose", True, "loose dog; stray dog roaming neighborhood"),
    ("AnimalLooseCat", True, "stray cat; cat with kittens; feral cats"),
    ("AnimalAggressiveBehavior", True, "aggressive dog; dog chasing or threatening people"),
    ("AnimalLackOfCare", True, "animal neglect; dog chained outside without water or shelter"),
    ("HighWeeds", True, "high weeds; overgrown grass on property; uncut lawn"),
    ("Emergency", False, "emergency in progress; fire; gas smell; someone injured or bleeding; call 911"),
    ("PrivatePropertyDispute", False, "dispute with neighbor; neighbor's tree or fence; private property disagreement"),
    ("LegalMatter", False, "lawsuit; court; fight a ticket; landlord deposit; legal advice"),
    ("OtherJurisdiction", False, "another city or county such as Plano, Fort Worth, Irving"),
    ("OtherAgency", False, "state or federal service; driver's license; DMV; passport"),
]

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
    ap.add_argument("--out", default="eval/results_baseline_R.csv")
    a = ap.parse_args()

    m = SentenceTransformer(a.model)
    labels = [c[0] for c in CONCEPTS]; permitted = {c[0]: c[1] for c in CONCEPTS}
    cvec = m.encode([f"{c[0]}: {c[2]}" for c in CONCEPTS], normalize_embeddings=True)

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
    print(f"\nmodel: {a.model}  threshold: {a.threshold}  accuracy: {df.correct.mean():.2f}\n")
    print(metrics(df).to_string(index=False))
