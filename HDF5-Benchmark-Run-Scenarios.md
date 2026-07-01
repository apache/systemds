# HDF5 Benchmark Run Scenarios

This guide explains simple ways to run the test below:
```text
HDF5BenchmarkTest#benchmarkHDF5ReadWrite
```

You can find the test file under this path:
```text
src/test/java/org/apache/sysds/test/functions/io/hdf5/HDF5BenchmarkTest.java
```

The benchmark is disabled by default. Every scenario must include:
```text
-Dsysds.test.hdf5.benchmark=true
```

I've provisioned 2 data profiles for which you can run the test:
```text
dense_double_only
sparse_like_double
```

For each of data profile above, the test checks below actions:
```text
raw_write
raw_read
hdf5_write
seq hdf5_read
par hdf5_read
```

After running the tests successfully, you can find the results under the below path:
```text
target/hdf5-benchmark-results.csv
target/hdf5-benchmark-results.json
```

For better analysis of performance, on each row of the results file, you can check rows with:
```json
"is_warmup": false
```

---

## Scenario 1: Quick test

### What it does

Runs the benchmark with the default small matrix size.

Default size:
```text
10000 x 100
```

### Why this Scenario?

I chose this scenario just to make sure that the test compiles, HDF5 files can be written/read, and CSV/JSON output is created.

### Windows PowerShell

```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.keep.files=true" `
  test
```

### macOS

```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.keep.files=true \
  test
```

### Expected result
The test should pass and create below files:
```text
target/hdf5-benchmark-results.csv
target/hdf5-benchmark-results.json
```

---

## Scenario 2: Full baseline benchmark
### What it does
Runs the main benchmark size used for performance comparison.

Size:
```text
100000 x 100
```
The sparse-like data profile is created as below:
```text
1 nonzero per row = 1% logical density
```

### Why this scenario?
I defined this test as the standard benchmark run for comparing raw I/O, HDF5 write, sequential HDF5 read, and parallel HDF5 read.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.rows=100000" `
  "-Dsysds.test.hdf5.cols=100" `
  "-Dsysds.test.hdf5.block.size=1024" `
  "-Dsysds.test.hdf5.sparse.nnz.per.row=1" `
  "-Dsysds.test.hdf5.warmup.reps=1" `
  "-Dsysds.test.hdf5.measure.reps=3" `
  "-Dsysds.hdf5.read.parallel.threads=4" `
  "-Dsysds.hdf5.read.parallel.min.bytes=67108864" `
  "-Dsysds.test.hdf5.keep.files=true" `
  test
```

### macOS
```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.rows=100000 \
  -Dsysds.test.hdf5.cols=100 \
  -Dsysds.test.hdf5.block.size=1024 \
  -Dsysds.test.hdf5.sparse.nnz.per.row=1 \
  -Dsysds.test.hdf5.warmup.reps=1 \
  -Dsysds.test.hdf5.measure.reps=3 \
  -Dsysds.hdf5.read.parallel.threads=4 \
  -Dsysds.hdf5.read.parallel.min.bytes=67108864 \
  -Dsysds.test.hdf5.keep.files=true \
  test
```

### Expected result
The output should have 40 result rows:
```text
2 profiles x 5 operations x (1 warmup + 3 measured reps)
```

---

## Scenario 3: Full baseline without keeping temporary files

### What it does
Runs the baseline benchmark and deletes temporary raw/HDF5 work files after success.

### Why this scenario?
When you only need the CSV/JSON output and do not want to keep the temporary HDF5 files.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.rows=100000" `
  "-Dsysds.test.hdf5.cols=100" `
  "-Dsysds.test.hdf5.keep.files=false" `
  test
```

### macOS
```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.rows=100000 \
  -Dsysds.test.hdf5.cols=100 \
  -Dsysds.test.hdf5.keep.files=false \
  test
```

### Expected result
CSV/JSON remain in:
```text
target/
```

Temporary files under:
```text
target/hdf5-benchmark-work-...
```
are deleted.

Note for Windows: if cleanup fails because an HDF5 file is still locked, run with:
```text
-Dsysds.test.hdf5.keep.files=true
```

---

## Scenario 4: Skip nonzero counting during read
### What it does
Runs the same benchmark but sets:
```text
-Dsysds.hdf5.read.skip.nnz=true
```

### Why this scenario?
Use this to check how much read time is affected by nonzero counting.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.rows=100000" `
  "-Dsysds.test.hdf5.cols=100" `
  "-Dsysds.test.hdf5.sparse.nnz.per.row=1" `
  "-Dsysds.hdf5.read.parallel.threads=4" `
  "-Dsysds.hdf5.read.parallel.min.bytes=67108864" `
  "-Dsysds.hdf5.read.skip.nnz=true" `
  "-Dsysds.test.hdf5.keep.files=true" `
  test
```

### macOS
```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.rows=100000 \
  -Dsysds.test.hdf5.cols=100 \
  -Dsysds.test.hdf5.sparse.nnz.per.row=1 \
  -Dsysds.hdf5.read.parallel.threads=4 \
  -Dsysds.hdf5.read.parallel.min.bytes=67108864 \
  -Dsysds.hdf5.read.skip.nnz=true \
  -Dsysds.test.hdf5.keep.files=true \
  test
```

---

## Scenario 5: Disable mmap for diagnostic comparison
### What it does
Runs the benchmark with:
```text
-Dsysds.hdf5.read.mmap=false
```

### Why this scenario?
This is only for diagnostic purposes.. It helps check whether memory-mapped reading affects parallel read performance or Windows file locking.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.rows=100000" `
  "-Dsysds.test.hdf5.cols=100" `
  "-Dsysds.hdf5.read.parallel.threads=4" `
  "-Dsysds.hdf5.read.parallel.min.bytes=67108864" `
  "-Dsysds.hdf5.read.mmap=false" `
  "-Dsysds.test.hdf5.keep.files=true" `
  test
```

### macOS
```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.rows=100000 \
  -Dsysds.test.hdf5.cols=100 \
  -Dsysds.hdf5.read.parallel.threads=4 \
  -Dsysds.hdf5.read.parallel.min.bytes=67108864 \
  -Dsysds.hdf5.read.mmap=false \
  -Dsysds.test.hdf5.keep.files=true \
  test
```

---

## Scenario 6: Multi-thread Run
### What it does
Runs the same test several times with different parallel read thread counts:
```text
1, 2, 4, 8
```

### Why this scenario?
Use this to see where parallel HDF5 read stops improving.
It is done by calling Maven multiple times.

### Windows PowerShell
```powershell
foreach ($threads in 1,2,4,8) {
  Write-Host "Running HDF5 benchmark with threads=$threads"

  mvn `
    "-Dhadoop.home.dir=$hadoopHome" `
    "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
    "-Dsysds.test.hdf5.benchmark=true" `
    "-Dsysds.test.hdf5.rows=100000" `
    "-Dsysds.test.hdf5.cols=100" `
    "-Dsysds.test.hdf5.sparse.nnz.per.row=1" `
    "-Dsysds.test.hdf5.warmup.reps=1" `
    "-Dsysds.test.hdf5.measure.reps=3" `
    "-Dsysds.hdf5.read.parallel.threads=$threads" `
    "-Dsysds.hdf5.read.parallel.min.bytes=67108864" `
    "-Dsysds.test.hdf5.keep.files=true" `
    test

  Copy-Item ".\target\hdf5-benchmark-results.json" ".\target\hdf5-benchmark-results-threads-$threads.json" -Force
  Copy-Item ".\target\hdf5-benchmark-results.csv" ".\target\hdf5-benchmark-results-threads-$threads.csv" -Force
}
```

### macOS / Linux
```bash
for threads in 1 2 4 8; do
  echo "Running HDF5 benchmark with threads=$threads"

  mvn \
    -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
    -Dsysds.test.hdf5.benchmark=true \
    -Dsysds.test.hdf5.rows=100000 \
    -Dsysds.test.hdf5.cols=100 \
    -Dsysds.test.hdf5.sparse.nnz.per.row=1 \
    -Dsysds.test.hdf5.warmup.reps=1 \
    -Dsysds.test.hdf5.measure.reps=3 \
    -Dsysds.hdf5.read.parallel.threads=$threads \
    -Dsysds.hdf5.read.parallel.min.bytes=67108864 \
    -Dsysds.test.hdf5.keep.files=true \
    test

  cp target/hdf5-benchmark-results.json target/hdf5-benchmark-results-threads-$threads.json
  cp target/hdf5-benchmark-results.csv target/hdf5-benchmark-results-threads-$threads.csv
done
```

### Expected result
The copied outputs are stored as:
```text
target/hdf5-benchmark-results-threads-1.json
target/hdf5-benchmark-results-threads-2.json
target/hdf5-benchmark-results-threads-4.json
target/hdf5-benchmark-results-threads-8.json
```
and matching `.csv` files.

---

## Scenario 7: Change sparse density
### What it does
Runs the sparse-like data profile with more nonzeros per row.

Example:
```text
5 nonzeros per row / 100 columns = 5% logical density
```

### Why this scenario?
It's there to check how logical sparsity affects HDF5 file size, write time, and sparse read time.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.rows=100000" `
  "-Dsysds.test.hdf5.cols=100" `
  "-Dsysds.test.hdf5.sparse.nnz.per.row=5" `
  "-Dsysds.hdf5.read.parallel.threads=4" `
  "-Dsysds.test.hdf5.keep.files=true" `
  test
```

### macOS / Linux
```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.rows=100000 \
  -Dsysds.test.hdf5.cols=100 \
  -Dsysds.test.hdf5.sparse.nnz.per.row=5 \
  -Dsysds.hdf5.read.parallel.threads=4 \
  -Dsysds.test.hdf5.keep.files=true \
  test
```

---

## Scenario 8: Change raw I/O buffer size
### What it does
Changes the Java stream buffer used by the raw byte baseline.

Example:
```text
-Dsysds.test.hdf5.raw.buffer.bytes=1048576
```

### Why run it
Only to check the raw I/O baseline. It does not change HDF5 writer/reader logic.

### Windows PowerShell
```powershell
mvn `
  "-Dhadoop.home.dir=$hadoopHome" `
  "-Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite" `
  "-Dsysds.test.hdf5.benchmark=true" `
  "-Dsysds.test.hdf5.rows=100000" `
  "-Dsysds.test.hdf5.cols=100" `
  "-Dsysds.test.hdf5.raw.buffer.bytes=1048576" `
  "-Dsysds.test.hdf5.keep.files=true" `
  test
```

### macOS / Linux
```bash
mvn \
  -Dtest=HDF5BenchmarkTest#benchmarkHDF5ReadWrite \
  -Dsysds.test.hdf5.benchmark=true \
  -Dsysds.test.hdf5.rows=100000 \
  -Dsysds.test.hdf5.cols=100 \
  -Dsysds.test.hdf5.raw.buffer.bytes=1048576 \
  -Dsysds.test.hdf5.keep.files=true \
  test
```

### Expected result
Only raw baseline timings should be directly affected.
