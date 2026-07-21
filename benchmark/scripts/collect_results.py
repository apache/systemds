#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

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

