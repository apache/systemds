# Running SystemDS

This guide explains how to run SystemDS regardless of whether you installed it from a Release or built it from Source. All execution modes -local, Spark, and federated- are covered in this document.

---

- [1. Prerequisites](#1-prerequisites)
- [2. Set SYSTEMDS_ROOT and PATH](#2-set-systemds_root-and-path)
- [3. Run a Simple Script Locally](#3-run-a-simple-script-locally)
- [4. Run a Script on Spark](#4-run-a-script-on-spark)
- [5. Run a Script in Federated Mode](#5-run-a-script-in-federated-mode)

---

# 1. Prerequisites

### Java Requirement ###
For compatability with Spark execution and parser components, **Java 17** is strongly recommended for SystemDS.

Verify Java version:

```bash
java -version
```

### Spark (required only for Spark execution) ###

- Use Spark 3.x.
- Spark 4.x is not supported due to ANTLR runtime incompatibilities.

Verify Spark version:

```bash
spark-submit --version
```

---

# 2. Set SYSTEMDS_ROOT and PATH

This step is required for both Release and Source-build installations. Run the following in the root directory of your SystemDS installation:

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH="$SYSTEMDS_ROOT/bin:$PATH"
```

It can be beneficial to enter these into your `~/.profile` or `~/.bashrc` for linux,
(but remember to change `$(pwd` to the full folder path)
or your environment variables in windows to enable reuse between terminals and restarts.

```bash
echo 'export SYSTEMDS_ROOT='$(pwd) >> ~/.bashrc
echo 'export PATH=$SYSTEMDS_ROOT/bin:$PATH' >> ~/.bashrc
```
---
# 3. Run a Simple Script Locally

This mode does not require Spark. It only needs Java 17.

### 3.1 Create and Run a Hello World

```bash
echo 'print("Hello, World!")' > hello.dml
```

Run:

```bash
systemds -f hello.dml
```

Expected output:

```bash
Hello, World!
```

### (Optional) MacOS Note: `realpath: illegal option -- -` Error
If you are running MacOS and encounter an error message similar to `realpath: illegal option -- -` when executing `systemds hello.dml`. You may try to replace the system-wide command `realpath` with the homebrew version `grealpath` that comes with the `coreutils`. Alternatively, you may change all occurrences within the script accordingly, i.e., by prepending a `g` to avoid any side effects.

### 3.2 Run a Real Example

This example demonstrates local execution of a real script `Univar-stats.dml`. The relevant commands to run this example with SystemDS is described in the DML Language reference guide at [DML Language Reference](dml-language-reference.html).

Prepare the data (macOS: use `curl`instead of `wget`):
```bash
# download test data
wget -P data/ http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data

# generate a metadata file for the dataset
echo '{"rows": 306, "cols": 4, "format": "csv"}' > data/haberman.data.mtd

# generate type description for the data
echo '1,1,1,2' > data/types.csv
echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd
```

Execute the DML Script:
```bash
systemds -f scripts/algorithms/Univar-Stats.dml -nvargs \
  X=data/haberman.data \
  TYPES=data/types.csv \
  STATS=data/univarOut.mtx \
  CONSOLE_OUTPUT=TRUE
```

### (Optional) MacOS Note: `SparkException` Error
If SystemDS tries to initialize Spark and you see `SparkException: A master URL must be set in your configuration`, you can force single-node execution without Spark/Hadoop initialization via:

```bash
systemds -exec singlenode -f scripts/algorithms/Univar-Stats.dml -nvargs \
  X=data/haberman.data \
  TYPES=data/types.csv \
  STATS=data/univarOut.mtx \
  CONSOLE_OUTPUT=TRUE
```

The script computes basic statistics (min, max, variance, skewness, etc) for each column of a dataset. Expected output (example):
```bash
-------------------------------------------------
Feature [1]: Scale
 (01) Minimum             | 30.0
 (02) Maximum             | 83.0
 (03) Range               | 53.0
 (04) Mean                | 52.45751633986928
 (05) Variance            | 116.71458266366658
 (06) Std deviation       | 10.803452349303281
 (07) Std err of mean     | 0.6175922641866753
 (08) Coeff of variation  | 0.20594669940735139
 (09) Skewness            | 0.1450718616532357
 (10) Kurtosis            | -0.6150152487211726
 (11) Std err of skewness | 0.13934809593495995
 (12) Std err of kurtosis | 0.277810485320835
 (13) Median              | 52.0
 (14) Interquartile mean  | 52.16013071895425
-------------------------------------------------
Feature [2]: Scale
 (01) Minimum             | 58.0
 (02) Maximum             | 69.0
 (03) Range               | 11.0
 (04) Mean                | 62.85294117647059
 (05) Variance            | 10.558630665380907
 (06) Std deviation       | 3.2494046632238507
 (07) Std err of mean     | 0.18575610076612029
 (08) Coeff of variation  | 0.051698529971741194
 (09) Skewness            | 0.07798443581479181
 (10) Kurtosis            | -1.1324380182967442
 (11) Std err of skewness | 0.13934809593495995
 (12) Std err of kurtosis | 0.277810485320835
 (13) Median              | 63.0
 (14) Interquartile mean  | 62.80392156862745
-------------------------------------------------
Feature [3]: Scale
 (01) Minimum             | 0.0
 (02) Maximum             | 52.0
 (03) Range               | 52.0
 (04) Mean                | 4.026143790849673
 (05) Variance            | 51.691117539912135
 (06) Std deviation       | 7.189653506248555
 (07) Std err of mean     | 0.41100513466216837
 (08) Coeff of variation  | 1.7857418611299172
 (09) Skewness            | 2.954633471088322
 (10) Kurtosis            | 11.425776549251449
 (11) Std err of skewness | 0.13934809593495995
 (12) Std err of kurtosis | 0.277810485320835
 (13) Median              | 1.0
 (14) Interquartile mean  | 1.2483660130718954
-------------------------------------------------
Feature [4]: Categorical (Nominal)
 (15) Num of categories   | 2
 (16) Mode                | 1
 (17) Num of modes        | 1
SystemDS Statistics:
Total execution time:   0,470 sec.
```

To check the location of output file created:
```bash
ls -l data/univarOut.mtx
```
---
# 4. Run a Script on Spark

SystemDS can be executed on Spark using the main executable JAR. The location of this JAR differs depending on whether you installed SystemDS from:

- a **Release archive**, or
- a **Source-build installation** (built with Maven)

### 4.1 Running with a Release installation

If you installed SystemDS from a release archive, the main JAR is located at:

```bash
SystemDS.jar
```

Run:

```bash
spark-submit SystemDS.jar -f hello.dml
```

### 4.2 Running with a Source-build installation

If you cloned the SystemDS repository and built it yourself, you must first run Maven to generate the executable JAR.

```bash
mvn -P distribution package
```
This creates several JAR files in `target/`:

Example output:

```bash
target/systemds-3.3.0-shaded.jar
target/systemds-3.3.0.jar
target/systemds-3.3.0-unshaded.jar
target/systemds-3.3.0-extra.jar
target/SystemDS.jar            <-- main runnable JAR
target/systemds-3.3.0-ropt.jar
target/systemds-3.3.0-javadoc.jar
```

Run:

```bash
spark-submit target/SystemDS.jar -f hello.dml
```
---
# 5. Run a Script in Federated Mode

Federated mode allows SystemDS to execute operations on data located on remote or distributed workers. Federated execution requires:

1. One or more **federated workers**
2. A **driver program** (DML or Python) that sends operations to those workers.

Note: The SystemDS documentation provides federated execution examples primarily via the Python API. This Quickstart demonstrates only how to start a federated worker, and refers users to the official Federated Environment guide for complete end-to-end examples.

### 5.1 Start a federated worker

Run in a separate terminal:

```bash
systemds WORKER 8001
```

This starts a worker on port `8001`.

### 5.2 Next steps and full examples

For complete, runnable examples of federated execution (including data files, metadata, and Python code), see the official [Federated Environment guide](https://systemds.apache.org/docs/2.1.0/api/python/guide/federated.html)

