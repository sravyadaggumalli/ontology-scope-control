"""ARM C & D: Run the ontology on the question set.

D  The correct concept for each question is read from the question file and
   handed to the ontology. No matching happens. This only tests whether the
   tree and its permissions give the right allow/refuse for every concept.

C  The concept is found automatically: each question is embedded and compared
   against every ontology class that has a description, using the same model
   and threshold as the flat baseline. The best match is handed to the
   ontology. Comparing C with D shows how much is lost to matching.

Usage:
    python src/run_ontology.py --arm D                      # no model needed
    python src/run_ontology.py --arm C [--threshold 0.55]
Output:
    eval/results_ontology_<arm>.csv and a per-group metrics table on stdout

Sources. Matching reuses the sentence-transformers semantic search util:
    https://www.sbert.net/examples/applications/semantic-search/README.html
    https://www.sbert.net/docs/package_reference/util.html  (util.semantic_search)
metrics() is a shared method so all runs are scored identically.
"""
import argparse, sys
import pandas as pd
sys.path.insert(0, "src")
from scope_ontology import ScopeOntology
from baseline_flat import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="eval/questions_v0.csv")
    ap.add_argument("--arm", choices=["C", "D"], required=True)
    ap.add_argument("--ontology", default="ontology/scope_v1.owl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"eval/results_ontology_{a.arm}.csv"

    # startup check
    so = ScopeOntology(a.ontology)
    problems = so.check()
    if problems:
        sys.exit("policy rejected at startup:\n  " + "\n  ".join(problems))
    print("startup check: ok")

    df = pd.read_csv(a.questions)

    if a.arm == "D":
        # Find the labeled concept from the question file. If a
        # question has no concept in the ontology (e.g. weather) it is NONE.
        df["matched"] = [c if so.concept(c) is not None else "NONE" for c in df.concept]
        df["score"] = 1.0
    else:
        # From the sbert semantic search example: encode the corpus and the
        # queries with convert_to_tensor=True, then util.semantic_search gives
        # the top_k corpus ids and cosine scores per query.
        from sentence_transformers import SentenceTransformer, util
        names, descs = zip(*so.leaf_concepts())
        m = SentenceTransformer(a.model)
        cvec = m.encode([f"{n}: {d}" for n, d in zip(names, descs)], convert_to_tensor=True)
        qvec = m.encode(df.question.tolist(), convert_to_tensor=True)
        hits = util.semantic_search(qvec, cvec, top_k=1)       # one list of hits per question
        best = [h[0]["corpus_id"] for h in hits]
        score = [h[0]["score"] for h in hits]
        # The ontology corpus is built as “Name: description” strings,
        # and if the top match falls below the similarity threshold,
        # it is treated as NONE, matching the same fallback rule used in arm R.
        df["matched"] = [names[i] if s >= a.threshold else "NONE" for i, s in zip(best, score)]
        df["score"] = [round(s, 3) for s in score]

    # The ontology resolves the decision: NONE is rejected immediately,
    # while verdict() walks the tree and records the class that granted or denied permission.
    verdicts = [so.verdict(c) if c != "NONE" else ("deny", None) for c in df.matched]
    df["pred"] = ["allow" if v == "permit" else "refuse" for v, _ in verdicts]
    df["decided_by"] = [src or "no match" for _, src in verdicts]
    df["correct"] = df.pred == df.label
    df.to_csv(out, index=False)

    # logging for each question, followed by the shared metrics table.
    for _, r in df.iterrows():
        print(f"{r.id}  gold={r.label:6s} pred={r.pred:6s} match={r.matched:26s} by={r.decided_by}")
    ins = df[df.label == "allow"]
    print(f"\narm {a.arm}  ontology: {a.ontology}  accuracy: {df.correct.mean():.2f}  "
          f"right concept: {(ins.matched == ins.concept).sum()}/{len(ins)}\n")
    print(metrics(df).to_string(index=False))


if __name__ == "__main__":
    main()
