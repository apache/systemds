"""
Parse per-run accuracy files into a single results.csv.

Output columns: label, epsilon, private, accuracy
"""
import pathlib, csv, re

RESULTS = pathlib.Path("benchmark/results")

rows = []

def parse_acc(path: pathlib.Path) -> float:
    txt = path.read_text().strip()
    # SystemDS writes a bare float.
    return float(txt)

# Non-private baseline.
baseline_path = RESULTS / "acc_baseline.txt"
if baseline_path.exists():
    rows.append(dict(label="baseline", epsilon="inf",
                     private=0, accuracy=parse_acc(baseline_path)))

# DP runs.
for eps in [0.5, 1, 4, 8]:
    p = RESULTS / f"acc_eps_{eps}.txt"
    if p.exists():
        rows.append(dict(label=f"ε={eps}", epsilon=eps,
                         private=1, accuracy=parse_acc(p)))
    else:
        print(f"Warning: {p} not found — skipping")

out = RESULTS / "results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label","epsilon","private","accuracy"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out}")
for r in rows:
    print(f"  {r['label']:12s}  acc={r['accuracy']:.4f}")

