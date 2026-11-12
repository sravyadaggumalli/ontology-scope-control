"""Compare all approaches on the same questions and the same split.

Reads the question file for the split column, then every result file given,
takes a majority verdict when an approach has several runs, and reports per
approach: out-of-scope allowed, in-scope refused, concept accuracy (strict, and
counting a Hazardous subclass of the right concept as right), unmatched, and
accuracy. Then McNemar's exact test between each approach and a reference.

Result files are matched to approaches by name:
    results_baseline_A*.csv   -> A        results_baseline_B*.csv -> B
    results_baseline_R*.csv   -> R        results_ontology_C_embed*.csv -> C-embed
    results_ontology_C_llm*.csv -> C-llm  results_ontology_D*.csv -> D
Files named results_*_pilot* or results_*_v1* from the pilot are ignored unless
passed explicitly.

Usage:
    python src/compare.py --split test [--questions eval/questions_v1.csv] [--ref C-llm] eval/results_*.csv
    python src/compare.py --split tune eval/results_*.csv
Writes docs/figures/results_<split>.png and prints the tables (also as LaTeX with --latex).

McNemar's test: exact binomial on the discordant pairs, as in
https://en.wikipedia.org/wiki/McNemar%27s_test#Variations
bar labels follow the matplotlib bar_label example:
https://matplotlib.org/stable/gallery/lib/bar_label_demo.html
majority vote: DataFrame.mode(axis=1), https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mode.html
"""
import argparse, glob, math, re
import pandas as pd
import matplotlib.pyplot as plt

def arm_of(path):
    base = path.split("/")[-1]
    for pat, arm in [(r"results_baseline_A", "A"), (r"results_baseline_B", "B"),
                     (r"results_baseline_R", "R"), (r"results_ontology_C_llm", "C-llm"),
                     (r"results_ontology_C_embed", "C-embed"), (r"results_ontology_C\b|results_ontology_C\.csv", "C-embed"),
                     (r"results_ontology_D", "D")]:
        if re.search(pat, base):
            return arm
    return None

def majority(dfs):
    votes = pd.concat([d.set_index("id").pred for d in dfs], axis=1)
    out = dfs[0].set_index("id").copy()
    out["pred"] = votes.mode(axis=1)[0]
    out["flipped"] = votes.nunique(axis=1) > 1
    return out.reset_index()

def mcnemar_exact(a_correct, b_correct):
    """two-sided exact p-value on discordant pairs"""
    b = int(((a_correct) & (~b_correct)).sum())   # a right, b wrong
    c = int(((~a_correct) & (b_correct)).sum())   # a wrong, b right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--questions", default="eval/questions_v1.csv")
    ap.add_argument("--split", default="test", choices=["test", "tune", "all"])
    ap.add_argument("--ref", default="C-llm", help="approach the others are tested against")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    q = pd.read_csv(a.questions)
    if a.split != "all":
        q = q[q.split == a.split]
    ids = set(q.id)
    gold = q.set_index("id")

    groups = {}
    for f in sorted(set(sum([glob.glob(x) for x in a.files], []))):
        arm = arm_of(f)
        if arm:
            d = pd.read_csv(f)
            if "id" in d and ids <= set(d.id):
                groups.setdefault(arm, []).append(d[d.id.isin(ids)])
            else:
                print(f"skip {f}: does not cover the {a.split} split")
    if not groups:
        raise SystemExit("no result files cover this split")

    rows, correct = [], {}
    for arm in ["A", "B", "R", "C-embed", "C-llm", "D"]:
        if arm not in groups:
            continue
        d = majority(groups[arm]).set_index("id").loc[gold.index]
        lab = gold.label; con = gold.concept
        oos = lab == "refuse"; ins = lab == "allow"
        strict = (d.matched == con) if "matched" in d else pd.Series(False, index=d.index)
        lenient = strict | (d.matched == con + "Hazardous") if "matched" in d else strict
        unmatched = (d.matched == "NONE") if "matched" in d else pd.Series(False, index=d.index)
        c = d.pred == lab
        correct[arm] = c
        rows.append({
            "approach": arm, "runs": len(groups[arm]),
            "out-of-scope allowed": f"{int((d.pred[oos]=='allow').sum())}/{int(oos.sum())}",
            "in-scope refused": f"{int((d.pred[ins]=='refuse').sum())}/{int(ins.sum())}",
            "concept acc. (strict)": f"{int(strict[ins].sum())}/{int(ins.sum())}" if "matched" in d else "--",
            "concept acc. (lenient)": f"{int(lenient[ins].sum())}/{int(ins.sum())}" if "matched" in d else "--",
            "unmatched": int(unmatched.sum()) if "matched" in d else "--",
            "accuracy": round(c.mean(), 3),
            "verdicts flipped": int(d.flipped.sum()) if len(groups[arm]) > 1 else "--",
        })
    res = pd.DataFrame(rows).set_index("approach")
    print(f"\n{a.split} split, n = {len(gold)} ({int((gold.label=='allow').sum())} in-scope, {int((gold.label=='refuse').sum())} out-of-scope)\n")
    print(res.to_string())

    # per group accuracy
    pg = pd.DataFrame({arm: correct[arm].groupby(gold.group).mean().round(2) for arm in correct})
    print("\naccuracy per group:\n" + pg.to_string())

    if a.ref in correct:
        print(f"\nMcNemar exact test vs {a.ref} (b = ref right/other wrong, c = ref wrong/other right):")
        for arm, c in correct.items():
            if arm == a.ref: continue
            b, cc, p = mcnemar_exact(correct[a.ref], c)
            print(f"  {a.ref} vs {arm:8s} b={b:3d} c={cc:3d} p={p:.3f}")

    if a.latex:
        print("\n" + res.drop(columns=["runs"]).to_latex())

    # figure: the two error rates per approach
    fig_rows = []
    for arm in res.index:
        o = res.loc[arm, "out-of-scope allowed"].split("/"); i = res.loc[arm, "in-scope refused"].split("/")
        fig_rows.append({"approach": arm, "out-of-scope allowed": 100*int(o[0])/int(o[1]),
                         "in-scope refused": 100*int(i[0])/int(i[1])})
    fig = pd.DataFrame(fig_rows).set_index("approach")
    ax = fig.plot.bar(rot=0, figsize=(8, 4), color=["#2A78D6", "#EB6834"])
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.0f}%", (p.get_x() + p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)
    plt.ylabel("error rate (%)")
    plt.title(f"Error rates on the {a.split} split (n={len(gold)})")
    plt.tight_layout()
    out = f"docs/figures/results_{a.split}.png"
    plt.savefig(out, dpi=150)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
