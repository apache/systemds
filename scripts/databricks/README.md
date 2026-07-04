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
{% endcomment %}
-->

# Running SystemDS on Databricks

Scripts and demo notebooks for deploying and running SystemDS on a Databricks
cluster. Tested against DBR 16.4 LTS (Spark 3.5.2 / Scala 2.12), where the
SystemDS jar runs unchanged.

## Contents

| File | Purpose |
| --- | --- |
| `deploy.sh` | Create a UC volume, upload `SystemDS.jar`, create a single-user cluster, install the Delta Kernel libraries, and import the demo notebooks. |
| `SystemDS_MLContext_Demo.scala` | Notebook: Unity Catalog round-trip using the SystemDS MLContext (Scala) API. Reads a table, runs a configurable DML script, writes the result back. |
| `SystemDS_Delta_E2E.scala` | Notebook: end-to-end Delta → linear regression on one Delta table. SystemDS reads it natively as a frame (`read(format="delta")`) → `transformencode` → `lm`; Spark ML reads the same table → `OneHotEncoder` → `LinearRegression`. Times read + encode + train for both. |
| `demo.dml` | Standalone DML smoke test: reads a matrix from storage, computes column sums and a Gram-matrix trace. |
| `.env.example` | Template for your local configuration. |

## Prerequisites

- The [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) installed and
  authenticated once interactively:

  ```bash
  databricks auth login --profile <your-profile>
  ```

- A built SystemDS jar at `<repo-root>/target/SystemDS.jar`
  (`mvn -q -DskipTests package`), or set `JAR_LOCAL` to point at one.
- `python3` on your `PATH` (used to parse CLI JSON output).

## Configuration

All settings are read from environment variables. The easiest way is a `.env`
file:

```bash
cp scripts/databricks/.env.example scripts/databricks/.env
# then edit scripts/databricks/.env
```

`deploy.sh` looks for a `.env` file by:

1. `ENV_FILE=/abs/path/to/.env` if you set it explicitly, otherwise
2. searching upward from the script's own directory (script dir → repo root →
   any parent directory) for the first `.env` it finds.

Anything already exported in your shell overrides values from `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `PROFILE` | `DEFAULT` | Databricks CLI profile. |
| `CATALOG` / `SCHEMA` / `VOLUME` | `main` / `default` / `systemds` | UC location for the jar volume (and notebook defaults). |
| `POLICY_ID` | _(empty)_ | Compute policy id; leave empty for none. |
| `SPARK_VERSION` | `16.4.x-scala2.12` | DBR runtime. |
| `NODE_TYPE` | `i3.xlarge` | Node type. |
| `NUM_WORKERS` | `0` | Worker count (0 = single node). |
| `AUTOTERMINATION_MINUTES` | `30` | Auto-terminate idle minutes. |
| `CLUSTER_NAME` | `systemds` | Cluster name. |
| `JAR_LOCAL` | `<repo-root>/target/SystemDS.jar` | Jar to upload. |
| `NB_DIR` | `/Users/<you>` | Workspace folder to import notebooks into. |
| `NB_FILES` | _(the 2 demo notebooks)_ | Space-separated notebooks to import; language detected from extension. |
| `DELTA_KERNEL_VERSION` | `3.3.2` | Delta Kernel Maven library version installed by `deploy.sh libs` (>= 3.3.2; must match `pom.xml`). |
| `USER_NAME` | _(auto-detected)_ | Databricks user. |

### Choosing a node type

`NODE_TYPE` is a cloud instance type. The default `i3.xlarge` is a small,
storage-optimized AWS node (4 vCPU / 30.5 GiB RAM / 950 GB local NVMe SSD); the
fast local disk is handy because the notebook spills SystemDS scratch to
`/local_disk0`. With `NUM_WORKERS=0` this single node is both driver and
executor, so it bounds the total memory available to SystemDS.

Some common AWS options (pick more cores/RAM for larger workloads):

| Node type | vCPU | RAM | Local SSD | Notes |
| --- | --- | --- | --- | --- |
| `i3.xlarge` | 4 | 30.5 GiB | 950 GB | Default; storage-optimized. |
| `i3.2xlarge` | 8 | 61 GiB | 1900 GB | Same family, 2× bigger. |
| `i4i.xlarge` | 4 | 32 GiB | 937 GB | Newer gen, faster storage. |
| `m5d.xlarge` | 4 | 16 GiB | 150 GB | General purpose w/ local SSD. |
| `r5d.2xlarge` | 8 | 64 GiB | 300 GB | Memory-optimized w/ local SSD. |

The exact set depends on your cloud (AWS / Azure / GCP) and workspace. List what
your workspace actually offers with:

```bash
databricks clusters list-node-types -p "$PROFILE" -o json
```

References:
[AWS EC2 instance types](https://aws.amazon.com/ec2/instance-types/),
[Azure VM sizes](https://learn.microsoft.com/azure/virtual-machines/sizes),
[GCP machine families](https://cloud.google.com/compute/docs/machine-resource).

## Usage

```bash
cd scripts/databricks
./deploy.sh upload     # create UC volume + copy SystemDS.jar into it
./deploy.sh cluster    # create the single-user cluster + install the jar
./deploy.sh libs       # install the Delta Kernel Maven libraries on the cluster
./deploy.sh import     # import the demo notebooks
./deploy.sh all        # all of the above
```

The created cluster id is written to `scripts/databricks/.cluster_id`.

### Delta Kernel libraries

The Delta notebook (`SystemDS_Delta_E2E`) reads Delta tables natively through the
Spark-free Delta Kernel, which is not on the DBR classpath. `./deploy.sh libs`
installs `io.delta:delta-kernel-defaults` (version `DELTA_KERNEL_VERSION`, default
`3.3.2`) as a cluster Maven library. Use **>= 3.3.2**: earlier releases trip a
classloader conflict with DBR's bundled parquet. The version must match the
`delta-kernel.version` property in the SystemDS `pom.xml`.

## Notebook configuration (`SystemDS_MLContext_Demo`)

The Scala notebook is driven by widgets, so nothing is hardcoded — set them in
the notebook UI or pass them as job parameters:

| Widget | Default | Description |
| --- | --- | --- |
| `catalog` / `schema` | `main` / `default` | Where the input/output tables live. |
| `input_table` / `output_table` | `systemds_input` / `systemds_output` | Table names. |
| `dml_path` | _(blank)_ | DML script to run. Blank uses the built-in z-score demo; otherwise a path readable from the driver (UC volume, `/Workspace`, or `dbfs:`). |
| `exec_type` | `DEFAULT` | SystemDS execution mode. `DEFAULT` lets SystemDS choose the plan; `DRIVER`, `SPARK`, or `DRIVER_AND_SPARK` force a mode. |

Custom DML contract: the script receives the input matrix as `X` and must
produce a matrix `Y` and a scalar `checksum`.

## Notebook configuration (`SystemDS_Delta_E2E`)

| Widget | Default | Description |
| --- | --- | --- |
| `catalog` / `schema` / `volume` | `main` / `default` / `systemds` | UC location; the Delta table is written under the volume. |
| `rows` | `1000000` | Rows in the generated Delta table. |
| `num_numeric` / `num_categorical` / `cardinality` | `100` / `20` / `30` | Feature shape. Defaults are deliberately encode-heavy (700 features) so the SystemDS-vs-Spark difference is visible. |
| `reg` | `1e-3` | L2 regularization for `lm`. |
| `recreate` | `true` | Rewrite the Delta table before running. |
| `statistics` | `true` | Print the SystemDS per-instruction breakdown. |

The encode complexity (categoricals × cardinality), not the row count, drives the
gap: more categoricals blow up Spark's `StringIndexer` + `OneHotEncoder` (each a
shuffle stage), while SystemDS dummycodes in-memory. On a single node, raw rows
instead favor Spark, and very large tables can exhaust driver memory.

Indicative single-node (`i3.xlarge`, 1M rows) numbers — single cold run, no warmup:

| workload | Spark ML | SystemDS | speedup |
| --- | --- | --- | --- |
| `SystemDS_Delta_E2E`, 700 features (read + encode + train) | 116.6 s | 55.4 s | ~2.1× |

The Spark side is the same speed on Spark 3.5.2 (DBR 16.4) and Spark 4.0.0
(DBR 17.3 LTS), so the comparison is not an artifact of an old runtime.

## Notes / gotchas baked into the scripts

- UC clusters only accept JAR libraries from a **UC Volume** (not DBFS, not
  `/Workspace`).
- The cluster must be **SINGLE_USER** (Assigned) mode; shared / USER_ISOLATION
  blocks JAR libraries.
- SystemDS needs the Vector API module plus a full `--add-opens` set at JVM
  launch (configured via `spark.{driver,executor}.extraJavaOptions`), and an
  absolute scratch dir (the notebook pins `sysds.scratch`).

`.env` and `.cluster_id` are git-ignored — they hold personal config and local
state and should never be committed.
