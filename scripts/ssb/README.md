# Star Schema Benchmark (SSB) for SystemDS

This directory contains the complete implementation of the Star Schema Benchmark (SSB) for SystemDS, including DML query implementations, performance comparison scripts, and supporting SQL implementations for benchmarking against PostgreSQL and DuckDB.

## Overview

The Star Schema Benchmark (SSB) is a data warehousing benchmark derived from TPC-H, designed to evaluate the performance of analytical database systems using a star schema design. This implementation provides:

- **13 DML query implementations** for SystemDS
- **Multi-engine performance comparison** (SystemDS vs PostgreSQL vs DuckDB)
- **Statistical analysis** with warmup runs and multiple repetitions
- **Comprehensive result output** in multiple formats (TXT, CSV, JSON)

## Directory Structure

SystemDS DML queries live here:

```
scripts/ssb/
├── README.md
└── queries/
    ├── q1_1.dml
    ├── q1_2.dml
    ├── q1_3.dml
    ├── q2_1.dml
    ├── q2_2.dml
    ├── q2_3.dml
    ├── q3_1.dml
    ├── q3_2.dml
    ├── q3_3.dml
    ├── q3_4.dml
    ├── q4_1.dml
    ├── q4_2.dml
    └── q4_3.dml
```

Benchmarking scripts are kept under `shell/` at the repository root (wrappers also exist at the root for convenience):

```
./
├── run_ssb.sh                  # Wrapper that invokes shell/run_ssb.sh
├── run_all_perf.sh             # Wrapper that invokes shell/run_all_perf.sh
├── shell/
│   ├── run_ssb.sh              # SystemDS-only SSB runner (results extraction)
│   ├── run_all_perf.sh         # Multi-engine performance comparison (SystemDS/PostgreSQL/DuckDB)
│   └── Output data/            # Default results output directory (space in name)
└── sql/
    ├── postgres/               # SQL counterparts (files named like q1.1.sql)
    └── duckdb/                 # SQL counterparts + DuckDB DB (ssb.duckdb)
```

## SSB Query Categories

### Flight 1 (Q1.*) - Basic Aggregation
- **Focus**: Simple aggregations with selective predicates
- **Tables**: `lineorder`, `date`
- **Metrics**: Revenue calculations with various filters

### Flight 2 (Q2.*) - Product Analysis
- **Focus**: Product dimension analysis
- **Tables**: `lineorder`, `dates`, `part`, `supplier`
- **Metrics**: Revenue by product categories and supplier regions

### Flight 3 (Q3.*) - Customer Analysis
- **Focus**: Customer and supplier geographic analysis
- **Tables**: `lineorder`, `dates`, `customer`, `supplier`
- **Metrics**: Revenue by customer and supplier regions/cities

### Flight 4 (Q4.*) - Profitability Analysis
- **Focus**: Advanced metrics with multiple dimensions
- **Tables**: `lineorder`, `dates`, `customer`, `supplier`, `part`
- **Metrics**: Revenue and profit analysis

## Prerequisites

### Data Requirements
The benchmark requires SSB data files in the `data/` directory:
- `date.tbl` - Date dimension table
- `lineorder3.tbl` - Fact table (lineorder)
- `customer.tbl` - Customer dimension table
- `supplier.tbl` - Supplier dimension table
- `part.tbl` - Part dimension table

Notes:
- Files are `|`-delimited, without headers.
- Queries in this repo expect the fact table as `lineorder3.tbl`. Some helper scripts may refer to `lineorder.tbl` for simple row-count summaries; using `lineorder3.tbl` for the DML queries is correct.

Optional: Generate data with ssb-dbgen (example for SF=1):
```
dbgen -s 1
# Copy the generated .tbl files into data/
```

### Software Requirements
- **SystemDS**: 3.4.0-SNAPSHOT or later
- **Java**: JDK 8 or later
- **PostgreSQL**: (optional, for performance comparison)
- **DuckDB**: (optional, for performance comparison)

The shell runners create a single-thread SystemDS config at `conf/single_thread.xml` and pass it automatically. On macOS, ensure GNU coreutils `timeout` is available if running the SystemDS-only runner (`brew install coreutils`).

## Usage

### Quickstart
1) Place SSB `.tbl` files into `data/` (see Data Requirements).
2) Run one query directly:
   ```bash
   ./bin/systemds scripts/ssb/queries/q1_1.dml
   ```
3) Run all queries and export results (default output under `shell/Output data/`):
   ```bash
   ./run_ssb.sh --out-dir output/ssb
   ```

### Running Individual Queries

Execute a single DML query:
```bash
# From project root directory
./bin/systemds scripts/ssb/queries/q1_1.dml
```

### SystemDS Result Extraction

Run all queries and extract results in multiple formats (per-query files):
```bash
# From repo root (wrapper)
./run_ssb.sh

# or run the script directly from the shell directory
cd shell && ./run_ssb.sh

# Default results directory: shell/Output data/ssb_run_YYYYMMDD_HHMMSS/
#   - txt/<query>.txt   (human-readable)
#   - csv/<query>.csv   (data analysis)
#   - json/<query>.json (structured data)
#   - run.json          (run metadata)

# Custom output directory
./run_ssb.sh --out-dir output/ssb
```

#### run_ssb.sh Options
- `--stats`: Enable SystemDS internal timing (adds `-stats` when invoking SystemDS)
- `--seed=XXXX`: Set random seed recorded in outputs
- `--out-dir=PATH`: Custom output directory (default: `shell/Output data`)
- `query_names`: Specific queries to run (e.g., `q1_1 q2_1` or `q1.1 q2.1`)

### Multi-Engine Performance Comparison

Compare performance across SystemDS, PostgreSQL, and DuckDB:
```bash
# From repo root (wrapper)
./run_all_perf.sh

# With custom parameters
./run_all_perf.sh --warmup 3 --repeats 10 --stats

# Specific queries only
./run_all_perf.sh q1_1 q2_1 q3_1

# Or from shell directory
cd shell && ./run_all_perf.sh --stats q1_1
```

### Command Line Options

#### run_all_perf.sh Options
- `--stats`: Enable SystemDS internal timing statistics
- `--warmup N`: Number of warmup runs (default: 1)
- `--repeats N`: Number of timed repetitions (default: 5)
- `--seed=XXXX`: Set random seed for reproducibility
- `query_names`: Specific queries to run (e.g., q1_1 q2_1)

Internals and setup:
- SystemDS: executes DML from `scripts/ssb/queries/`
- PostgreSQL: looks for SQL in `sql/postgres/` (files named like `q1.1.sql`)
- DuckDB: looks for SQL in `sql/duckdb/` and a database at `sql/duckdb/ssb.duckdb`
- Single-threading is enforced for fairness:
  - SystemDS: `conf/single_thread.xml` with `sysds.num.threads=1` (auto-generated)
  - PostgreSQL: session `SET` commands disable parallel workers
  - DuckDB: `PRAGMA threads=1`

Outputs:
- Console table shows mean (and stdev/p95 when `--repeats>1`).
- CSV: `shell/Output data/results_YYYYMMDD_HHMMSS.csv`
- JSON metadata: `shell/Output data/results_YYYYMMDD_HHMMSS_metadata.json`

#### Statistical Output Format
When using multiple repetitions, results show:
```
1824 (±10, p95:1840)
```
- **1824**: Mean execution time (ms)
- **±10**: Standard deviation (ms)
- **p95:1840**: 95th percentile (ms)

## Implementation Details

### DML Query Structure
Each DML query follows a consistent pattern:
1. **Header Comment**: Original SQL query for reference
2. **Source Imports**: Required built-in functions
3. **Data Loading**: CSV table reads with frame data type
4. **Query Logic**: Relational algebra operations
5. **Output**: Results written to console and files

### Performance Optimization
- **Single-threaded execution**: Fair comparison across engines
- **Warmup runs**: JVM stabilization
- **Multiple repetitions**: Statistical reliability
- **Memory management**: Optimized JVM settings

### Built-in Functions Used
- `raSelection.dml`: Selection operations with predicates
- `raJoin.dml`: Join operations between tables
- `raGroupBy.dml`: Aggregation with grouping
- Various utility functions for data manipulation

## Output Formats

### SystemDS Runner (Per-Query Files)

TXT example (`txt/q1_1.txt`):
```
=========================================
SSB Query: q1_1
=========================================
Timestamp: 2025-08-27 20:30:45 UTC
Seed: 12345

Result:
---------
REVENUE: 2324000000
=========================================
```

CSV example (`csv/q1_1.csv`):
```csv
query,result
q1_1,2324000000
```

JSON example (`json/q1_1.json`):
```json
{
  "query": "q1_1",
  "timestamp": "2025-08-27 20:30:45 UTC",
  "seed": 12345,
  "result": "REVENUE: 2324000000",
  "metadata": {
    "systemds_version": "3.4.0-SNAPSHOT",
    "hostname": "host.local",
    "git_commit": "abc1234"
  }
}
```

Run metadata (`run.json`) example:
```json
{
  "benchmark_type": "ssb_systemds",
  "timestamp": "2025-08-27 20:30:45 UTC",
  "hostname": "host.local",
  "git_commit": "abc1234",
  "seed": 12345,
  "software_versions": { "systemds": "3.4.0-SNAPSHOT", "jdk": "17.0.8" },
  "system_resources": { "cpu": "Apple M2", "ram": "16GB" },
  "data_build_info": "customer:150000 supplier:10000 part:200000 date:2556 lineorder:6001215",
  "run_configuration": { "statistics_enabled": true, "queries_selected": 13, "queries_executed": 13, "queries_failed": 0 },
  "results": [
    { "query": "q1_1", "result": "REVENUE: 2324000000", "status": "success" }
  ]
}
```

## Performance Analysis

### Timing Measurements
- **SystemDS Shell**: Total execution time including JVM startup and I/O
- **SystemDS Core**: Pure computation time (visible when `--stats` is enabled)
- **PostgreSQL**: Single-threaded SQL execution (parallel disabled per session)
- **DuckDB**: Single-threaded analytical processing (`PRAGMA threads=1`)

### Statistical Analysis
Multiple repetitions provide:
- **Mean**: Average performance
- **Standard Deviation**: Consistency measure
- **95th Percentile**: Worst-case bounds

## Architecture Notes

### Relational Algebra Approach
The DML implementations use SystemDS's relational algebra built-in functions rather than direct matrix operations, providing:
- **SQL-like semantics**: Familiar query patterns
- **Optimization opportunities**: Built-in function optimizations
- **Maintainability**: Clear query structure

### Data Flow Pattern
1. **Load**: CSV files → SystemDS frames
2. **Filter**: Selection operations on individual tables
3. **Join**: Relational joins between dimensions and facts
4. **Aggregate**: Group-by operations with sum/count
5. **Output**: Results to multiple formats

## Troubleshooting

### Common Issues
1. **Data files missing**: Ensure SSB data files are in `data/` directory
2. **Memory errors**: Increase JVM heap size in scripts
3. **Permission errors**: Make scripts executable with `chmod +x`
4. **Path issues**: Run scripts from correct directory (project root wrappers, or `cd shell` for direct scripts)
5. **macOS timeout**: Install GNU coreutils to provide `timeout` (used by `shell/run_ssb.sh`)

### Performance Issues
1. **Slow execution**: Check data file sizes and system resources
2. **Inconsistent timing**: Increase warmup runs
3. **High standard deviation**: System load affecting measurements

### Optional: Multi‑Engine Setup
- **PostgreSQL**
  - Create database (default expected name: `ssb`), create tables matching SSB schema, and `COPY` data from `data/*.tbl` with `DELIMITER '|'`.
  - Place SQL queries under `sql/postgres/` named like `q1.1.sql`.
  - The perf script disables parallel workers for fairness.
- **DuckDB**
  - Create or reuse `sql/duckdb/ssb.duckdb`, load data from `data/*.tbl` with `DELIM '|'`.
  - Place SQL queries under `sql/duckdb/` named like `q1.1.sql`.
  - The perf script enforces `PRAGMA threads=1`.

## Contributing

### Adding New Queries
1. Create new `.dml` file in `queries/` directory
2. Follow existing naming convention (qX_Y.dml)
3. Include original SQL as header comment
4. Test with `run_ssb.sh` script
5. Add corresponding SQL files for comparison engines

### Script Modifications
1. Test changes with small query subset
2. Verify output format compatibility
3. Update documentation as needed
4. Validate statistical calculations

## References

- [Star Schema Benchmark Specification](http://www.cs.umb.edu/~poneil/StarSchemaB.PDF)
- [SystemDS Documentation](https://systemds.apache.org/docs)
- [SSB Query Definitions](https://github.com/electrum/ssb-dbgen)

## License

This implementation is part of the Apache SystemDS project and follows the same Apache 2.0 license.
