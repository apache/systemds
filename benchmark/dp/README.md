<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# DP-FedAvg Benchmark

Benchmarks the `dp_gaussian` built-in by sweeping the privacy budget ε for
federated logistic regression (DP-FedAvg) on the UCI Adult dataset, and
plotting the accuracy/privacy trade-off.

## Setup

Run these commands from the repository root. They create a virtual
environment named `python_venv` and install the Python dependencies listed
in [scripts/requirements.txt](scripts/requirements.txt):

```bash
python3 -m venv benchmark/dp/scripts/python_venv
source benchmark/dp/scripts/python_venv/bin/activate
pip install -r benchmark/dp/scripts/requirements.txt
```

Build SystemDS before running the benchmark, if you haven't already:

```bash
mvn clean package -DskipTests
```

## Running

With `python_venv` activated, run the benchmark from the repository root:

```bash
bash benchmark/dp/scripts/run_benchmark.sh
```

This prepares the dataset, starts the federated workers, sweeps
epsilon over {0.5, 1, 4, 8} plus a non-private baseline, stops the workers, and
generates the plots.

## Outputs

- Prepared dataset and per-worker federated shards: [benchmark/dp/data/](data/)
- Trained models, accuracy logs, and results table: [benchmark/dp/results/](results/)
- Accuracy vs. epsilon plot: [benchmark/dp/results/accuracy_vs_epsilon.png](results/accuracy_vs_epsilon.png)
- Utility cost of privacy plot: [benchmark/dp/results/privacy_cost.png](results/privacy_cost.png)

Deactivate the virtual environment when done:

```bash
deactivate
```
