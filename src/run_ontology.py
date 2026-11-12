"""ARM C & D: Run the ontology on the question set.

D  The correct concept for each question is read from the question file and
   handed to the ontology. No matching happens. This only tests whether the
   tree and its permissions give the right allow/refuse for every concept.

C  The concept is found automatically: each question is embedded and compared
   against every ontology class that has a description, using the same model
   and threshold as the flat baseline. The best match is handed to the
   ontology. Comparing C with D shows how much is lost to matching.

Usage:
    python src/run_ontology.py --arm D                        # no model needed
    python src/run_ontology.py --arm C                        # embedding matcher, concepts only, clean descriptions
    python src/run_ontology.py --arm C --candidates all --raw-desc   # the pilot configuration
    python src/run_ontology.py --arm C --matcher llm          # the hosted LLM picks the concept (needs API key)
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
    ap.add_argument("--ontology", default="ontology/scope_v2.owl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--matcher", choices=["embed", "llm"], default="embed",
                    help="arm C only: embedding similarity (default) or the hosted LLM picks the concept")
    ap.add_argument("--llm", default="gpt-6-luna")
    ap.add_argument("--candidates", choices=["all", "concepts"], default="concepts",
                    help="arm C only: which classes the matcher may pick (see ScopeOntology.candidates)")
    ap.add_argument("--raw-desc", action="store_true", help="arm C only: keep the Routine/Hazardous suffixes")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or (f"eval/results_ontology_{a.arm}.csv" if a.arm == "D"
                    else f"eval/results_ontology_C_{a.matcher}.csv")

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
    elif a.matcher == "embed":
        # From the sbert semantic search example: encode the corpus and the
        # queries with convert_to_tensor=True, then util.semantic_search gives
        # the top_k corpus ids and cosine scores per query.
        from sentence_transformers import SentenceTransformer, util
        names, descs = zip(*so.candidates(a.candidates, clean=not a.raw_desc))
        m = SentenceTransformer(a.model)
        cvec = m.encode([f"{n}: {d}" for n, d in zip(names, descs)], convert_to_tensor=True)
        qvec = m.encode(df.question.tolist(), convert_to_tensor=True)
        hits = util.semantic_search(qvec, cvec, top_k=1)       # one list of hits per question
        best = [h[0]["corpus_id"] for h in hits]
        score = [h[0]["score"] for h in hits]
        # The ontology corpus is built as "Name: description" strings,
        # and if the top match falls below the similarity threshold,
        # it is treated as NONE, matching the same fallback rule used in arm R.
        df["matched"] = [names[i] if s >= a.threshold else "NONE" for i, s in zip(best, score)]
        df["score"] = [round(s, 3) for s in score]
    else:
        # Same hosted model as baselines A and B, but it only picks a concept
        # name from the list; the tree still makes the allow/refuse decision.
        # Chat-completions call follows the openai quickstart, as in baseline_prompt.py.
        import json, re, time
        from openai import OpenAI
        client = OpenAI()
        cands = so.candidates(a.candidates, clean=not a.raw_desc)
        names = [n for n, _ in cands]
        listing = "\n".join(f"- {n}: {d}" for n, d in cands)
        prompt = ("You classify resident messages to a city 311 assistant into exactly one "
                  "concept from the list below, or NONE if no concept fits. Pick the concept "
                  "whose description best matches what the resident is reporting. Do not "
                  "decide whether the message should be handled; only name the concept.\n\n"
                  f"{listing}\n\nRespond with JSON only: {{\"concept\": \"<name or NONE>\"}}")
        matched = []
        for i, q in enumerate(df.question):
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=a.llm,
                        messages=[{"role": "system", "content": prompt},
                                  {"role": "user", "content": q}],
                        response_format={"type": "json_object"})
                    text = r.choices[0].message.content
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(2)
            try:
                name = str(json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("concept", "NONE")).strip()
            except Exception:
                name = "NONE"
            matched.append(name if name in names else "NONE")
        df["matched"] = matched
        df["score"] = 1.0

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
    cfg = "" if a.arm == "D" else f"  matcher: {a.matcher}  candidates: {a.candidates}  clean_desc: {not a.raw_desc}"
    print(f"\narm {a.arm}  ontology: {a.ontology}{cfg}  accuracy: {df.correct.mean():.2f}  "
          f"right concept: {(ins.matched == ins.concept).sum()}/{len(ins)}\n")
    print(metrics(df).to_string(index=False))


if __name__ == "__main__":
    main()
