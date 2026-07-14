"""
Read results.csv and produce two figures:

1. accuracy_vs_epsilon.png
   Line plot: x = ε, y = accuracy.
   Horizontal dashed line = non-private baseline.
   Points labelled with accuracy values.

2. privacy_cost.png
   Bar chart showing accuracy loss relative to baseline (utility cost of DP).
"""
import pathlib
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS = pathlib.Path("benchmark/results")

# ── Load ──────────────────────────────────────────────────────────────────
rows = []
with open(RESULTS / "results.csv") as f:
    for r in csv.DictReader(f):
        rows.append({
            "label":    r["label"],
            "epsilon":  float(r["epsilon"]) if r["epsilon"] != "inf" else None,
            "private":  int(r["private"]),
            "accuracy": float(r["accuracy"]),
        })

baseline = next(r for r in rows if r["private"] == 0)
dp_rows  = sorted([r for r in rows if r["private"] == 1],
                  key=lambda r: r["epsilon"])

eps_vals = [r["epsilon"] for r in dp_rows]
acc_vals = [r["accuracy"] for r in dp_rows]
baseline_acc = baseline["accuracy"]

# ── Figure 1: Accuracy vs ε ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(eps_vals, acc_vals, marker="o", linewidth=2,
        color="#028090", label="DP-FedAvg (Gaussian)")
ax.axhline(baseline_acc, linestyle="--", color="#1C3A5E",
           linewidth=1.5, label=f"Non-private baseline ({baseline_acc:.3f})")

# Annotate each DP point.
for eps, acc in zip(eps_vals, acc_vals):
    ax.annotate(f"{acc:.3f}", xy=(eps, acc),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color="#028090")

ax.set_xscale("log")
ax.set_xticks(eps_vals)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("Privacy budget ε  (smaller = stronger privacy)", fontsize=11)
ax.set_ylabel("Test accuracy", fontsize=11)
ax.set_title("Accuracy vs. Privacy Budget — DP-FedAvg on Adult (4 workers)",
             fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(max(0, min(acc_vals) - 0.05), min(1.0, baseline_acc + 0.05))
ax.grid(True, which="both", linestyle=":", alpha=0.5)

plt.tight_layout()
out1 = RESULTS / "accuracy_vs_epsilon.png"
fig.savefig(out1, dpi=150)
print(f"Saved {out1}")
plt.close()

# ── Figure 2: Utility cost (accuracy drop) ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

drops = [baseline_acc - acc for acc in acc_vals]
colors = ["#B91C1C" if d > 0.02 else "#028090" for d in drops]
# Position bars at their true ε value on a log-scaled x-axis (rather than
# evenly-spaced categorical slots) so the visual spacing between 0.5→1 and
# 4→8 reflects the same 2x ratio. Bar widths scale with x so they stay a
# constant fraction of their slot in log space instead of shrinking/growing.
widths = [e * 0.4 for e in eps_vals]
bars = ax.bar(eps_vals, drops, color=colors, width=widths,
              edgecolor="white")
ax.set_xscale("log")
ax.set_xticks(eps_vals)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

for bar, drop in zip(bars, drops):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{drop:.3f}", ha="center", va="bottom", fontsize=9)

ax.legend(["drop > 0.02", "drop <= 0.02"])

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Privacy budget ε", fontsize=11)
ax.set_ylabel("Accuracy drop vs. baseline", fontsize=11)
ax.set_title("Utility Cost of Differential Privacy — DP-FedAvg on Adult",
             fontsize=11)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
out2 = RESULTS / "privacy_cost.png"
fig.savefig(out2, dpi=150)
print(f"Saved {out2}")
plt.close()

# ── Console summary table ─────────────────────────────────────────────────
print()
print(f"{'ε':>8}  {'accuracy':>10}  {'drop':>8}")
print("-" * 34)
print(f"{'baseline':>8}  {baseline_acc:10.4f}  {'—':>8}")
for eps, acc, drop in zip(eps_vals, acc_vals, drops):
    print(f"{eps:>8.1f}  {acc:10.4f}  {drop:8.4f}")

