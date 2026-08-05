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
BENCHMARK_DIR=benchmark/dp

# 1. Prepare data (once).
python ${BENCHMARK_DIR}/scripts/prepare_data.py

# 2. Run the sweep (starts workers, trains, evaluates, stops workers).
bash ${BENCHMARK_DIR}/scripts/run_sweep.sh

# 3. Collect results into CSV.
python ${BENCHMARK_DIR}/scripts/collect_results.py

# 4. Generate plots.
python ${BENCHMARK_DIR}/scripts/plot.py

# 5. Confirm outputs exist.
ls -lh ${BENCHMARK_DIR}/results/accuracy_vs_epsilon.png ${BENCHMARK_DIR}/results/privacy_cost.png

