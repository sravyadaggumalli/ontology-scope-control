"""Summarize the baseline runs and plot the two error rates per run.

Reads eval/results_baseline_*.csv and writes docs/figures/baseline_results.png.
Prompt runs (results_baseline_A_run*.csv) are combined by majority vote.
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt

# flat-list runs: one file each
runs = {}
for f in sorted(glob.glob("eval/results_baseline_R*.csv")):
    name = f.split("results_baseline_")[1][:-4]
    runs[name] = pd.read_csv(f)

# prompt runs: majority vote across the run files
a_files = sorted(glob.glob("eval/results_baseline_A_run*.csv"))
if a_files:
    dfs = [pd.read_csv(f) for f in a_files]
    votes = pd.concat([d.pred for d in dfs], axis=1)
    maj = dfs[0].copy()
    maj["pred"] = votes.mode(axis=1)[0]
    flipped = (votes.nunique(axis=1) > 1).sum()
    print(f"prompt: {flipped} of {len(maj)} verdicts changed across {len(dfs)} runs")
    runs["A (majority)"] = maj

# error rates per run
rows = []
for name, df in runs.items():
    oos = df[df.label == "refuse"]
    ins = df[df.label == "allow"]
    rows.append({
        "run": name,
        "out_of_scope_allowed": (oos.pred == "allow").mean() * 100,
        "in_scope_refused": (ins.pred == "refuse").mean() * 100,
    })
res = pd.DataFrame(rows).set_index("run")
print(res.round(1))

# plot
res.plot.bar(rot=15, figsize=(7, 4))
plt.ylabel("error rate (%)")
plt.title("Baseline error rates, 42-question pilot")
plt.tight_layout()
plt.savefig("docs/figures/baseline_results.png", dpi=150)
print("wrote docs/figures/baseline_results.png")
