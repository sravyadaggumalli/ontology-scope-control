"""Initial analysis of the cleaned Dallas 311 taxonomy.

Input : data/taxonomy_clean.csv
Output: docs/figures/taxonomy_overview.png and printed summary stats
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/taxonomy_clean.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/taxonomy_overview.png"

df = pd.read_csv(SRC)
total = df["count"].sum()

# Summary Stats
print(f"services: {len(df)}   departments: {df.department.nunique()}   requests: {total:,}")
print(f"services with >= 50 requests: {(df['count'] >= 50).sum()}")
print(f"services with <  10 requests: {(df['count'] < 10).sum()}")
top = df.iloc[0]
print(f"largest single service: '{top.service}' = {top['count']:,} ({100*top['count']/total:.1f}% of all)")
print("\nrequests by department:")
by_dept = df.groupby("department")["count"].agg(["sum", "size"]).sort_values("sum", ascending=False)
print(by_dept.head(12).to_string())

# figure: two panels
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e6e3"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("white")

# panel 1: requests by department (listing top 10, remaining are folded into Other)
k = 10
d = by_dept["sum"]
shown = d.head(k).copy()
if len(d) > k:
    shown["Other (%d depts)" % (len(d) - k)] = d.iloc[k:].sum()
shown = shown[::-1] / 1e3
ax1.barh(shown.index, shown.values, color=BLUE, height=0.62)
for y, v in enumerate(shown.values):
    ax1.text(v + 8, y, f"{v:,.0f}k", va="center", fontsize=7.5, color=MUTED)
ax1.set_xlabel("requests since Oct 2020 (thousands)", fontsize=8, color=MUTED)
ax1.set_title("Requests by department", fontsize=9.5, color=INK, loc="left")
ax1.tick_params(labelsize=7.5, colors=INK, length=0)
ax1.set_xlim(0, shown.max() * 1.18)

# panel 2: distribution of requests across services, sorted by volume (log scale)
counts = df["count"].sort_values(ascending=False).reset_index(drop=True)
ax2.plot(counts.index + 1, counts.values, color=BLUE, lw=2)
ax2.set_yscale("log")
ax2.axhline(50, color=MUTED, lw=1, ls=":")
ax2.text(len(counts) * 0.98, 60, "50-request cutoff", ha="right", fontsize=7.5, color=MUTED)
ax2.set_xlabel("service rank", fontsize=8, color=MUTED)
ax2.set_ylabel("requests (log)", fontsize=8, color=MUTED)
ax2.set_title("Distribution of requests across services", fontsize=9.5, color=INK, loc="left")
ax2.tick_params(labelsize=7.5, colors=INK, length=0)
ax2.grid(axis="y", color=GRID, lw=0.8)
ax2.set_axisbelow(True)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nwrote {OUT}")
