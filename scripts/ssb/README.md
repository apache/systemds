# Star Schema Benchmark (SSB) for SystemDS

This README documents the SSB DML queries under `scripts/ssb/queries/` and the runner scripts under `scripts/ssb/shell/` that execute and benchmark them. It is focused on what is implemented today, how to run it, and how to interpret the outputs for performance analysis.

---

## Table of Contents

1. Project Layout
2. Quick Start
3. Data Location (`--input-dir` and DML `input_dir`)
4. Single-Engine Runner (`scripts/ssb/shell/run_ssb.sh`)
5. Multi-Engine Performance Runner (`scripts/ssb/shell/run_all_perf.sh`)
6. Outputs and Examples
7. Adding/Editing Queries
8. Troubleshooting

---

## 1) Project Layout

Paths are relative to the repo root:

```
systemds/
├── scripts/ssb/
│   ├── README.md                              # This guide
│   ├── queries/                               # DML queries (q1_1.dml ... q4_3.dml)
│   │   ├── q1_1.dml - q1_3.dml                # Flight 1
│   │   ├── q2_1.dml - q2_3.dml                # Flight 2
│   │   ├── q3_1.dml - q3_4.dml                # Flight 3
│   │   └── q4_1.dml - q4_3.dml                # Flight 4
│   ├── shell/
│   │   ├── run_ssb.sh                         # Single-engine (SystemDS) runner
│   │   ├── run_all_perf.sh                    # Multi-engine performance benchmark
│   │   └── ssbOutputData/                     # Results (created on first run)
│   │       ├── QueryData/                     # Per-query outputs from run_ssb.sh
│   │       └── PerformanceData/               # Multi-engine outputs from run_all_perf.sh
│   └── sql/                                   # SQL versions + `ssb.duckdb` for DuckDB
```

Note: The SSB raw data directory is not committed. You must point the runners to your generated data with `--input-dir`.

---

## 2) Quick Start

Set up SystemDS and run the SSB queries.

1) Build SystemDS (from repo root):

```bash
mvn -DskipTests package
```

2) Make sure the SystemDS binary exists (repo-local `bin/systemds` or on `PATH`).

3) Make runner scripts executable:

```bash
chmod +x scripts/ssb/shell/run_ssb.sh scripts/ssb/shell/run_all_perf.sh
```

4) Provide SSB data (from dbgen) in a directory, e.g. `/path/to/ssb-data`.

5) Run a single SSB query on SystemDS (from repo root):

```bash
scripts/ssb/shell/run_ssb.sh q1.1 --input-dir=/path/to/ssb-data --stats
```

6) Run the multi-engine performance benchmark across all queries (from repo root):

```bash
scripts/ssb/shell/run_all_perf.sh --input-dir=/path/to/ssb-data --stats --repeats=5
```

If `--input-dir` is omitted, the scripts default to `./data/` under the repo root.

---

## 3) Data Location (`--input-dir` and DML `input_dir`)

Both runners pass a named argument `input_dir` into DML as:

```
-nvargs input_dir=/absolute/path/to/ssb-data
```

Your DML scripts should construct paths from `input_dir`. Example:

```dml
dates     = read(paste(input_dir, "/date.tbl", sep=""), data_type="frame", format="csv", sep="|", header=FALSE)
lineorder = read(paste(input_dir, "/lineorder.tbl", sep=""), data_type="frame", format="csv", sep="|", header=FALSE)
```

Expected base files in `input_dir`: `customer.tbl`, `supplier.tbl`, `part.tbl`, `date.tbl` and `lineorder*.tbl` (fact table name can vary by scale). The runners validate that `--input-dir` exists before executing.

---

## 4) Single-Engine Runner (`scripts/ssb/shell/run_ssb.sh`)

Runs SSB DML queries with SystemDS and saves results per query.

- Usage:
  - `scripts/ssb/shell/run_ssb.sh` — run all SSB queries
  - `scripts/ssb/shell/run_ssb.sh q1.1 q2.3` — run specific queries
  - `scripts/ssb/shell/run_ssb.sh --stats` — include SystemDS internal statistics
  - `scripts/ssb/shell/run_ssb.sh --input-dir=/path/to/data` — set data dir
  - `scripts/ssb/shell/run_ssb.sh --output-dir=/tmp/out` — set output dir

- Query names: You can use dotted form (`q1.1`); the runner maps to `q1_1.dml` internally.

- Functionality:
  - Single-threaded execution via auto-generated `conf/single_thread.xml`.
  - DML `input_dir` forwarding with `-nvargs`.
  - Pre-check for data directory; clear errors if missing.
  - Runtime error detection by scanning for “An Error Occurred : …”.
  - Optional `--stats` to capture SystemDS internal statistics in JSON.
  - Per-query outputs in TXT, CSV, and JSON.
  - `run.json` with run-level metadata and per-query status/results.
  - Clear end-of-run summary and, for table results, a “DETAILED QUERY RESULTS” section.
  - Exit code is non-zero if any query failed (handy for CI).

- Output layout:
  - Base directory: `--output-dir` (default: `scripts/ssb/shell/ssbOutputData/QueryData`)
  - Each run: `ssb_run_<YYYYMMDD_HHMMSS>/`
    - `txt/<query>.txt` — human-readable result
    - `csv/<query>.csv` — scalar or table as CSV
    - `json/<query>.json` — per-query JSON
    - `run.json` — full metadata and results for the run

- Example console output (abridged):

```
[1/13] Running: q1_1.dml
...
=========================================
SSB benchmark completed!
Total queries executed: 13
Failed queries: 0
Statistics: enabled

=========================================
RUN METADATA SUMMARY
=========================================
Timestamp:       2025-09-05 12:34:56 UTC
Hostname:        myhost
Seed:            123456
Software Versions:
  SystemDS:      3.4.0-SNAPSHOT
  JDK:           21.0.2
System Resources:
  CPU:           Apple M2
  RAM:           16GB
Data Build Info:
  SSB Data:      customer:300000 part:200000 supplier:2000 lineorder:6001215
=========================================

===================================================
QUERIES SUMMARY
===================================================
No.  Query           Result                         Status
---------------------------------------------------
1    q1.1            12 rows (see below)            ✓ Success
2    q1.2            1                              ✓ Success
...
===================================================

=========================================
DETAILED QUERY RESULTS
=========================================
[1] Results for q1.1:
----------------------------------------
1992|ASIA|12345.67
1993|ASIA|23456.78
...
----------------------------------------
```

---

## 5) Multi-Engine Performance Runner (`scripts/ssb/shell/run_all_perf.sh`)

Runs SSB queries across SystemDS, PostgreSQL, and DuckDB with repeated timings and statistical analysis.

- Usage:
  - `scripts/ssb/shell/run_all_perf.sh` — run all queries on available engines
  - `scripts/ssb/shell/run_all_perf.sh q1.1 q2.3` — run specific queries
  - `scripts/ssb/shell/run_all_perf.sh --warmup=2 --repeats=10` — control sampling
  - `scripts/ssb/shell/run_all_perf.sh --stats` — include core/internal engine timings
  - `scripts/ssb/shell/run_all_perf.sh --layout=wide|stacked` — control terminal layout
  - `scripts/ssb/shell/run_all_perf.sh --input-dir=... --output-dir=...` — set paths

- Query names: dotted form (`q1.1`) is accepted; mapped internally to `q1_1.dml`.

- Engine prerequisites:
  - PostgreSQL:
    - Install `psql` CLI and ensure a PostgreSQL server is running.
    - Default connection in the script: `POSTGRES_DB=ssb`, `POSTGRES_USER=$(whoami)`, `POSTGRES_HOST=localhost`.
    - Create the `ssb` database and load the standard SSB tables and data (schema not included in this repo). The SQL queries under `scripts/ssb/sql/` expect the canonical SSB schema and data.
    - The runner verifies connectivity; if it cannot connect or tables are missing, PostgreSQL results are skipped.
  - DuckDB:
    - Install the DuckDB CLI (`duckdb`).
    - The runner looks for the database at `scripts/ssb/sql/ssb.duckdb`. Ensure it contains SSB tables and data.
    - If the CLI is missing or the DB file cannot be opened, DuckDB results are skipped.
  - SystemDS is required; the other engines are optional. Missing engines are reported and skipped gracefully.

- Functionality:
  - Single-threaded execution for fairness (SystemDS config; SQL engines via settings).
  - Pre-flight data-dir check and SystemDS test-run with runtime-error detection.
  - Warmups and repeated measurements using `/usr/bin/time -p` (ms resolution).
  - Statistics per engine: mean, population stdev, p95, and CV%.
  - “Shell” vs “Core” time: SystemDS core from `-stats`, PostgreSQL core via EXPLAIN ANALYZE, DuckDB core via JSON profiling.
  - Environment verification: gracefully skips PostgreSQL or DuckDB if not available.
  - Terminal-aware output: wide table with grid or stacked multi-line layout.
  - Results to CSV and JSON with rich metadata (system info, versions, run config).

- Layouts (display formats):
  - Auto selection: `--layout=auto` (default). Chooses `wide` if terminal is wide enough, else `stacked`.
  - Wide layout: `--layout=wide`. Prints a grid with columns for each engine and a `Fastest` column. Three header rows show labels for `mean`, `±/CV`, and `p95`.
  - Stacked layout: `--layout=stacked` or `--stacked`. Prints a compact, multi-line block per query (best for narrow terminals).
  - Dynamic scaling: The wide layout scales column widths to fit the terminal; if still too narrow, it falls back to stacked.
  - Row semantics: Row 1 = mean (ms); Row 2 = `±stdev/CV%`; Row 3 = `p95 (ms)`.
  - Fastest: The runner highlights the engine with the lowest mean per query.

- Output layout:
  - Base directory: `--output-dir` (default: `scripts/ssb/shell/ssbOutputData/PerformanceData`)
  - Files per run (timestamped basename):
    - `ssb_results_<UTC_ISO_TIMESTAMP>.csv`
    - `ssb_results_<UTC_ISO_TIMESTAMP>.json`

- Example console output (abridged, wide layout):

```
==================================================================================
                      MULTI-ENGINE PERFORMANCE BENCHMARK METADATA
==================================================================================
Timestamp:       2025-09-05 12:34:56 UTC
Hostname:        myhost
Seed:            123456
Software Versions:
  SystemDS:      3.4.0-SNAPSHOT
  JDK:           21.0.2
  PostgreSQL:    psql (PostgreSQL) 14.11
  DuckDB:        v0.10.3
System Resources:
  CPU:           Apple M2
  RAM:           16GB
Data Build Info:
  SSB Data:      customer:300000 part:200000 supplier:2000 lineorder:6001215
Run Configuration:
  Statistics:    enabled
  Queries:       13 selected
  Warmup Runs:   1
  Repeat Runs:   5

+--------+--------------+--------------+--------------+----------------+--------------+----------------+----------+
| Query  | SysDS Shell  | SysDS Core   | PostgreSQL   | PostgreSQL Core| DuckDB       | DuckDB Core    | Fastest  |
|        | mean         | mean         | mean         | mean           | mean         | mean           |          |
|        | ±/CV         | ±/CV         | ±/CV         | ±/CV           | ±/CV         | ±/CV           |          |
|        | p95          | p95          | p95          | p95            | p95          | p95            |          |
+--------+--------------+--------------+--------------+----------------+--------------+----------------+----------+
| q1_1   | 1824.0       | 1210.0       | 2410.0       | 2250.0         | 980.0        | 910.0          | DuckDB   |
|        | ±10.2/0.6%   | ±8.6/0.7%    | ±15.1/0.6%   | ±14.0/0.6%     | ±5.4/0.6%    | ±5.0/0.5%      |          |
|        | p95:1840.0   | p95:1225.0   | p95:2435.0   | p95:2274.0     | p95:989.0    | p95:919.0      |          |
+--------+--------------+--------------+--------------+----------------+--------------+----------------+----------+
```

- Example console output (abridged, stacked layout):

```
Query  : q1_1    Fastest: DuckDB
  SystemDS Shell: 1824.0
                   ±10.2ms/0.6%
                   p95:1840.0ms
  SystemDS Core:  1210.0
                   ±8.6ms/0.7%
                   p95:1225.0ms
  PostgreSQL:     2410.0
                   ±15.1ms/0.6%
                   p95:2435.0ms
  PostgreSQL Core:2250.0
                   ±14.0ms/0.6%
                   p95:2274.0ms
  DuckDB:         980.0
                   ±5.4ms/0.6%
                   p95:989.0ms
  DuckDB Core:    910.0
                   ±5.0ms/0.5%
                   p95:919.0ms
--------------------------------------------------------------------------------
```

---

## 6) Outputs and Examples

Where to find results and how to read them.

- SystemDS-only runner (`scripts/ssb/shell/run_ssb.sh`):
  - Directory: `scripts/ssb/shell/ssbOutputData/QueryData/ssb_run_<YYYYMMDD_HHMMSS>/`
  - Files: `txt/<query>.txt`, `csv/<query>.csv`, `json/<query>.json`, and `run.json`
  - `run.json` example (stats enabled, single query):

```json
{
  "benchmark_type": "ssb_systemds",
  "timestamp": "2025-09-07 19:45:11 UTC",
  "hostname": "eduroam-141-23-175-117.wlan.tu-berlin.de",
  "seed": 849958376,
  "software_versions": {
    "systemds": "3.4.0-SNAPSHOT",
    "jdk": "17.0.15"
  },
  "system_resources": {
    "cpu": "Apple M1 Pro",
    "ram": "16GB"
  },
  "data_build_info": {
    "customer": "30000",
    "part": "200000",
    "supplier": "2000",
    "date": "2557",
    "lineorder": "8217"
  },
  "run_configuration": {
    "statistics_enabled": true,
    "queries_selected": 1,
    "queries_executed": 1,
    "queries_failed": 0
  },
  "results": [
    {
      "query": "q1_1",
      "result": "687752409 ",
      "stats": [
        "SystemDS Statistics:",
        "Total elapsed time:        1.557 sec.",
        "Total compilation time:        0.410 sec.",
        "Total execution time:        1.147 sec.",
        "Cache hits (Mem/Li/WB/FS/HDFS):    11054/0/0/0/2.",
        "Cache writes (Li/WB/FS/HDFS):    0/26/3/0.",
        "Cache times (ACQr/m, RLS, EXP):    0.166/0.001/0.060/0.000 sec.",
        "HOP DAGs recompiled (PRED, SB):    0/175.",
        "HOP DAGs recompile time:    0.063 sec.",
        "Functions recompiled:        2.",
        "Functions recompile time:    0.016 sec.",
        "Total JIT compile time:        1.385 sec.",
        "Total JVM GC count:        1.",
        "Total JVM GC time:        0.026 sec.",
        "Heavy hitter instructions:",
        "  #  Instruction           Time(s)  Count",
        "  1  m_raJoin                0.940      1",
        "  2  ucumk+                  0.363      3",
        "  3  -                       0.219   1345",
        "  4  nrow                    0.166      7",
        "  5  ctable                  0.086      2",
        "  6  *                       0.078      1",
        "  7  parallelBinarySearch    0.069      1",
        "  8  ba+*                    0.049      5",
        "  9  rightIndex              0.016   8611",
        " 10  leftIndex               0.015   1680"
      ],
      "status": "success"
    }
  ]
}
```

  Notes:
  - The `result` field contains the query’s output (scalar or tabular content collapsed). When `--stats` is used, `stats` contains the full SystemDS statistics block line-by-line.
  - For failed queries, an `error_message` string is included and `status` is set to `"error"`.

- Multi-engine runner (`scripts/ssb/shell/run_all_perf.sh`):
  - Directory: `scripts/ssb/shell/ssbOutputData/PerformanceData/`
  - Files per run: `ssb_results_<UTC_ISO_TIMESTAMP>.csv` and `.json`
  - CSV contains display strings and raw numeric stats (mean/stdev/p95) for each engine; JSON contains the same plus metadata and fastest-engine per query.
  - `ssb_results_*.json` example (stats enabled, single query):

```json
{
  "benchmark_metadata": {
    "benchmark_type": "multi_engine_performance",
    "timestamp": "2025-09-07 20:11:16 UTC",
    "hostname": "eduroam-141-23-175-117.wlan.tu-berlin.de",
    "seed": 578860764,
    "software_versions": {
      "systemds": "3.4.0-SNAPSHOT",
      "jdk": "17.0.15",
      "postgresql": "psql (PostgreSQL) 17.5",
      "duckdb": "v1.3.2 (Ossivalis) 0b83e5d2f6"
    },
    "system_resources": {
      "cpu": "Apple M1 Pro",
      "ram": "16GB"
    },
    "data_build_info": {
      "customer": "30000",
      "part": "200000",
      "supplier": "2000",
      "date": "2557",
      "lineorder": "8217"
    },
    "run_configuration": {
      "statistics_enabled": true,
      "queries_selected": 1,
      "warmup_runs": 1,
      "repeat_runs": 5
    }
  },
  "results": [
    {
      "query": "q1_1",
      "systemds": {
        "shell": {
          "display": "2186.0 (±95.6ms/4.4%, p95:2250.0ms)",
          "mean_ms": 2186.0,
          "stdev_ms": 95.6,
          "p95_ms": 2250.0
        },
        "core": {
          "display": "1151.2 (±115.3ms/10.0%, p95:1334.0ms)",
          "mean_ms": 1151.2,
          "stdev_ms": 115.3,
          "p95_ms": 1334.0
        },
        "status": "success",
        "error_message": null
      },
      "postgresql": {
        "display": "26.0 (±4.9ms/18.8%, p95:30.0ms)",
        "mean_ms": 26.0,
        "stdev_ms": 4.9,
        "p95_ms": 30.0
      },
      "postgresql_core": {
        "display": "3.8 (±1.4ms/36.8%, p95:5.7ms)",
        "mean_ms": 3.8,
        "stdev_ms": 1.4,
        "p95_ms": 5.7
      },
      "duckdb": {
        "display": "30.0 (±0.0ms/0.0%, p95:30.0ms)",
        "mean_ms": 30.0,
        "stdev_ms": 0.0,
        "p95_ms": 30.0
      },
      "duckdb_core": {
        "display": "1.1 (±0.1ms/9.1%, p95:1.3ms)",
        "mean_ms": 1.1,
        "stdev_ms": 0.1,
        "p95_ms": 1.3
      },
      "fastest_engine": "PostgreSQL"
    }
  ]
}
```

  Differences at a glance:
  - Single-engine `run.json` focuses on query output (`result`) and, when enabled, the SystemDS `stats` array. Status and error handling are per-query.
  - Multi-engine results JSON focuses on timing statistics for each engine (`shell` vs `core` for SystemDS; `postgresql`/`postgresql_core`; `duckdb`/`duckdb_core`) along with a `fastest_engine` field. It does not include the query’s actual result values.

---

## 7) Adding/Editing Queries

Guidelines for DML in `scripts/ssb/queries/`:

- Name files as `qX_Y.dml` (e.g., `q1_1.dml`). The runners accept `q1.1` on the CLI and map it for you.
- Always derive paths from `input_dir` named argument (see Section 3).
- Keep I/O separate from compute where possible (helps early error detection).
- Add a short header comment with original SQL and intent.

Example header:

```dml
/*
  SQL: SELECT ...
  Description: Revenue per month by supplier region
*/
```

---

## 8) Troubleshooting

- Missing data directory: pass `--input-dir=/path/to/ssb-data` and ensure `*.tbl` files exist.
- SystemDS not found: build (`mvn -DskipTests package`) and use `./bin/systemds` or ensure `systemds` is on PATH.
- Query fails with runtime error: the runners mark `status: "error"` and include a short `error_message` in JSON outputs. See console snippet for context.
- macOS cache dropping: OS caches cannot be dropped like Linux; the multi-engine runner mitigates with warmups + repeated averages and reports p95/CV.

If something looks off, attach the relevant `run.json` or `ssb_results_*.json` when filing issues.

- To debug DML runtime errors, run the DML directly:

```bash
./bin/systemds -f scripts/ssb/queries/q1_1.dml -nvargs input_dir=/path/to/data
```

- When `--stats` is enabled, SystemDS internal "core" timing is extracted and reported separately (useful to separate JVM / startup overhead from core computation).

All these metrics appear in the generated CSVs and JSON entries.
- Permission errors: `chmod +x scripts/ssb/shell/*.sh`.
