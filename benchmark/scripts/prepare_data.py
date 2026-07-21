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
Download the UCI Adult dataset, binarise labels, standardise features,
split into 4 equal horizontal partitions for federated workers, and write
SystemDS .mtd metadata files alongside each CSV shard.

Outputs
-------
benchmark/data/worker{1..4}/X_train.csv  + X_train.csv.mtd
benchmark/data/worker{1..4}/y_train.csv  + y_train.csv.mtd
benchmark/data/X_test.csv                + X_test.csv.mtd
benchmark/data/y_test.csv                + y_test.csv.mtd
benchmark/data/meta.txt                  # n_train, n_test, n_features
"""

import json, os, pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/adult/adult.data"
)
ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/adult/adult.test"
)

COLS = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","label",
]
NUMERIC = ["age","fnlwgt","education_num","capital_gain",
           "capital_loss","hours_per_week"]

DATA_DIR = pathlib.Path("benchmark/data")
N_WORKERS = 4

def download(url, dest):
    import urllib.request
    if not dest.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)

def load_adult(path, skip_rows=0):
    df = pd.read_csv(path, names=COLS, skipinitialspace=True,
                     skiprows=skip_rows, na_values="?").dropna()
    # binarise label: >50K → 1, else 0
    df["label"] = (df["label"].str.strip().str.rstrip(".") == ">50K").astype(float)
    # one-hot encode categoricals
    cats = [c for c in COLS[:-1] if c not in NUMERIC]
    df = pd.get_dummies(df, columns=cats, drop_first=True)
    return df

def write_csv_and_mtd(arr: np.ndarray, path: pathlib.Path, description: str):
    """Write a CSV and a matching SystemDS .mtd metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=",", fmt="%.8f")
    rows, cols = arr.shape
    mtd = {
        "data_type": "matrix",
        "value_type": "double",
        "rows": rows,
        "cols": cols,
        "format": "csv",
        "header": False,
        "description": description,
    }
    with open(str(path) + ".mtd", "w") as f:
        json.dump(mtd, f, indent=2)
    print(f"  {path}  ({rows} × {cols})")

# ── Download ─────────────────────────────────────────────────────────────────
download(ADULT_TRAIN_URL, DATA_DIR / "raw" / "adult.data")
download(ADULT_TEST_URL,  DATA_DIR / "raw" / "adult.test")

train_df = load_adult(DATA_DIR / "raw" / "adult.data")
test_df  = load_adult(DATA_DIR / "raw" / "adult.test", skip_rows=1)

# Align columns (test may have different dummies after get_dummies).
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

# ── Feature / label split ────────────────────────────────────────────────────
feature_cols = [c for c in train_df.columns if c != "label"]
X_train = train_df[feature_cols].values.astype(float)
y_train = train_df["label"].values.reshape(-1, 1).astype(float)
X_test  = test_df[feature_cols].values.astype(float)
y_test  = test_df["label"].values.reshape(-1, 1).astype(float)

# ── Standardise (fit on train only) ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

n_train, n_features = X_train.shape
n_test = X_test.shape[0]
print(f"Train: {n_train} × {n_features}  |  Test: {n_test} × {n_features}")

# ── Partition across workers (equal horizontal splits) ──────────────────────
indices = np.array_split(np.arange(n_train), N_WORKERS)
for i, idx in enumerate(indices, start=1):
    wdir = DATA_DIR / f"worker{i}"
    write_csv_and_mtd(X_train[idx], wdir / "X_train.csv",
                      f"Adult features, worker {i}")
    write_csv_and_mtd(y_train[idx], wdir / "y_train.csv",
                      f"Adult labels,   worker {i}")

# ── Test set (coordinator-local) ────────────────────────────────────────────
write_csv_and_mtd(X_test, DATA_DIR / "X_test.csv",  "Adult test features")
write_csv_and_mtd(y_test, DATA_DIR / "y_test.csv",  "Adult test labels")

# ── Metadata for DML scripts ─────────────────────────────────────────────────
worker_rows = [len(idx) for idx in indices]
with open(DATA_DIR / "meta.txt", "w") as f:
    f.write(f"n_train={n_train}\n")
    f.write(f"n_test={n_test}\n")
    f.write(f"n_features={n_features}\n")
    for i, r in enumerate(worker_rows, start=1):
        f.write(f"worker{i}_rows={r}\n")
print("Wrote benchmark/data/meta.txt")

