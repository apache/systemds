#!/usr/bin/env bash
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

# Runs FedAvg for each epsilon value and the non-private baseline,
# then evaluates accuracy. Results are appended to results/results.csv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JAR="$REPO_ROOT/target/SystemDS.jar"
SCRIPTS="$REPO_ROOT/benchmark/scripts"
DATA="$REPO_ROOT/benchmark/data"
RESULTS="$REPO_ROOT/benchmark/results"
mkdir -p "$RESULTS"

# Read dataset metadata written by prepare_data.py.
source <(grep -E '^(n_train|n_test|n_features|worker[1-4]_rows)=' \
              "$DATA/meta.txt" | sed 's/=/="/;s/$/"/')

COMMON_ARGS="\
  data_dir=$DATA \
  n_features=$n_features \
  w1_rows=$worker1_rows \
  w2_rows=$worker2_rows \
  w3_rows=$worker3_rows \
  w4_rows=$worker4_rows \
  n_rounds=300 \
  lr=1.0 \
  clip_norm=4.0 \
  delta=1e-5"

# ── Start workers ─────────────────────────────────────────────────────────
echo "=== Starting federated workers ==="
bash "$SCRIPTS/start_workers.sh"

run_one() {
  local label="$1"    # e.g. "eps_0.5" or "baseline"
  local extra="$2"    # extra -nvargs for this run
  local model="$RESULTS/model_${label}.csv"
  local acc_file="$RESULTS/acc_${label}.txt"

  echo ""
  echo "--- Training: $label ---"
  java --add-modules=jdk.incubator.vector -jar "$JAR" \
      -f "$SCRIPTS/fedavg_dp.dml" \
      -nvargs $COMMON_ARGS $extra out="$model" \
      2>&1 | tee "$RESULTS/train_${label}.log" | grep -E "FedAvg|error|Error" || true

  echo "  Evaluating …"
  java --add-modules=jdk.incubator.vector -jar "$JAR" \
      -f "$SCRIPTS/eval.dml" \
      -nvargs data_dir="$DATA" model_path="$model" out_acc="$acc_file" \
      2>&1 | grep "Accuracy:"
}

# ── Non-private baseline ──────────────────────────────────────────────────
run_one "baseline" "private=0 epsilon=9999"

# ── DP runs ───────────────────────────────────────────────────────────────
for EPS in 0.5 1 4 8; do
  LABEL="eps_${EPS}"
  run_one "$LABEL" "private=1 epsilon=${EPS}"
done

# ── Stop workers ──────────────────────────────────────────────────────────
echo ""
echo "=== Stopping federated workers ==="
bash "$SCRIPTS/stop_workers.sh"

echo ""
echo "All runs complete. Logs and model files in $RESULTS/"

