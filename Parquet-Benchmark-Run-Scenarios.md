# Parquet Benchmark Run Scenarios

This guide explains simple ways to run the test below:
```text
ParquetBenchmarkTest#benchmarkParquetReadWrite
```

You can find the test file under this path:
```text
src/test/java/org/apache/sysds/test/functions/io/ParquetBenchmarkTest.java
```

The benchmark is disabled by default. Every scenario must include:
```text
-Dsysds.test.parquet.benchmark=true
```

After running the tests successfully, you can find the results under the below path:
```text
target/parquet-benchmark.csv
target/parquet-benchmark.json
```

For better analysis of performance, on each row of the results file, you can check rows with:
```json
"is_warmup": false
```

---

## Scenario 1: Quick test
### What it does
Runs a very small dense Double benchmark.

Size:
```text
10000 x 20
```

Profile:
```text
dense_double_only
```

### Why this scenario:
It's there to check that the test compiles, Parquet files can be written/read, and CSV/JSON output is created.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=10000" `
  "-Dsysds.test.parquet.cols=20" `
  "-Dsysds.test.parquet.warmup=0" `
  "-Dsysds.test.parquet.reps=1" `
  "-Dsysds.test.parquet.profiles=dense_double_only" `
  test
```

### macOS
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=10000 \
  -Dsysds.test.parquet.cols=20 \
  -Dsysds.test.parquet.warmup=0 \
  -Dsysds.test.parquet.reps=1 \
  -Dsysds.test.parquet.profiles=dense_double_only \
  test
```

### Expected result
The test should pass and create:
```text
target/parquet-benchmark.csv
target/parquet-benchmark.json
```

Expected number of result rows:
```text
1 profile x 8 operations x 1 measured rep = 8 rows
```

---

## Scenario 2: Final profile benchmark
### What it does
Runs the default final Parquet benchmark across three data profiles:
```text
dense_double_only
mixed_schema
sparse_like_double
```

Default size:
```text
100000 x 50
```

For each profile, the scenario runs:
```text
seq write
seq raw_io_read
seq footer_read
seq read
parallel write
parallel raw_io_read
parallel footer_read
parallel read
```

### Why this scenario
This is intended for a standard final Parquet benchmark. It gives evidence for dense numeric data, mixed type handling, and sparse-like numeric behavior.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=100000" `
  "-Dsysds.test.parquet.cols=50" `
  "-Dsysds.test.parquet.warmup=1" `
  "-Dsysds.test.parquet.reps=3" `
  test
```

### macOS
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=100000 \
  -Dsysds.test.parquet.cols=50 \
  -Dsysds.test.parquet.warmup=1 \
  -Dsysds.test.parquet.reps=3 \
  test
```

### Expected result
Expected number of result rows:
```text
3 profiles x 8 operations x (1 warmup + 3 measured reps) = 96 rows
```

---

## Scenario 3: Dense Double baseline benchmark
### What it does
Runs a larger dense double-only benchmark.

Size:
```text
200000 x 50
```

Profile:
```text
dense_double_only
```

### Why this scenario
This will reproduce the earlier dense-only baseline used to compare sequential and current parallel Parquet read/write behavior.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=200000" `
  "-Dsysds.test.parquet.cols=50" `
  "-Dsysds.test.parquet.warmup=2" `
  "-Dsysds.test.parquet.reps=5" `
  "-Dsysds.test.parquet.profiles=dense_double_only" `
  test
```

### macOS
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=200000 \
  -Dsysds.test.parquet.cols=50 \
  -Dsysds.test.parquet.warmup=2 \
  -Dsysds.test.parquet.reps=5 \
  -Dsysds.test.parquet.profiles=dense_double_only \
  test
```

### Expected result
Expected number of result rows:
```text
1 profile x 8 operations x (2 warmup + 5 measured reps) = 56 rows
```

---

## Scenario 4: Manual multipart parallel-reader experiment
### What it does
Manually creates Parquet input directories with multiple part files:
```text
2, 4, 8 part files
```
Then it benchmarks the current parallel reader on these multi-file inputs.

### Why this scenario
We wanna see whether the current parallel reader benefits from multiple input part files, independent of the current writer partitioning behavior.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=200000" `
  "-Dsysds.test.parquet.cols=50" `
  "-Dsysds.test.parquet.warmup=2" `
  "-Dsysds.test.parquet.reps=5" `
  "-Dsysds.test.parquet.profiles=dense_double_only" `
  "-Dsysds.test.parquet.manual.parts=2,4,8" `
  test
```

### macOS
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=200000 \
  -Dsysds.test.parquet.cols=50 \
  -Dsysds.test.parquet.warmup=2 \
  -Dsysds.test.parquet.reps=5 \
  -Dsysds.test.parquet.profiles=dense_double_only \
  -Dsysds.test.parquet.manual.parts=2,4,8 \
  test
```

### Expected result
Expected number of result rows:
```text
Normal dense benchmark: 8 operations x 7 reps = 56 rows
Manual multipart: 3 part settings x 3 operations x 7 reps = 63 rows
Total = 119 rows
```

---

## Scenario 5: Run only one data profile
### What it does
Runs only one selected data profile, for example:
```text
mixed_schema
```

or:

```text
sparse_like_double
```

### Why this scenario?
When we only want to inspect one behavior: mixed type handling or sparse-like compression/materialization.

### Windows PowerShell: For mixed schema only
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=100000" `
  "-Dsysds.test.parquet.cols=50" `
  "-Dsysds.test.parquet.warmup=1" `
  "-Dsysds.test.parquet.reps=3" `
  "-Dsysds.test.parquet.profiles=mixed_schema" `
  test
```

### macOS: For mixed schema only
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=100000 \
  -Dsysds.test.parquet.cols=50 \
  -Dsysds.test.parquet.warmup=1 \
  -Dsysds.test.parquet.reps=3 \
  -Dsysds.test.parquet.profiles=mixed_schema \
  test
```

### Windows PowerShell: For sparse-like only
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=100000" `
  "-Dsysds.test.parquet.cols=50" `
  "-Dsysds.test.parquet.warmup=1" `
  "-Dsysds.test.parquet.reps=3" `
  "-Dsysds.test.parquet.profiles=sparse_like_double" `
  test
```

### macOS: For sparse-like only
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=100000 \
  -Dsysds.test.parquet.cols=50 \
  -Dsysds.test.parquet.warmup=1 \
  -Dsysds.test.parquet.reps=3 \
  -Dsysds.test.parquet.profiles=sparse_like_double \
  test
```

### Expected result
Expected number of result rows:
```text
1 profile x 8 operations x (1 warmup + 3 measured reps) = 32 rows
```

---

## Scenario 6: Change benchmark size
### What it does
Runs the same benchmark with a custom matrix/frame size.

Example size:
```text
50000 x 100
```

### Why this scenario?
Use this to check to see if results scale roughly with the number of cells and file size.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite" `
  "-Dsysds.test.parquet.benchmark=true" `
  "-Dsysds.test.parquet.rows=50000" `
  "-Dsysds.test.parquet.cols=100" `
  "-Dsysds.test.parquet.warmup=1" `
  "-Dsysds.test.parquet.reps=3" `
  test
```

### macOS
```bash
mvn \
  -Dtest=ParquetBenchmarkTest#benchmarkParquetReadWrite \
  -Dsysds.test.parquet.benchmark=true \
  -Dsysds.test.parquet.rows=50000 \
  -Dsysds.test.parquet.cols=100 \
  -Dsysds.test.parquet.warmup=1 \
  -Dsysds.test.parquet.reps=3 \
  test
```