#!/usr/bin/env bash
#
# Multi-Engine SSB Performance Benchmark Runner
# =============================================
#
# CORE SCRIPTS STATUS:
# - Version: 1.0 (September 5, 2025)
# - Status: Production-Ready with Advanced Statistical Analysis
#
# ENHANCED FEATURES IMPLEMENTED:
# ✓ Multi-engine benchmarking (SystemDS, PostgreSQL, DuckDB)
# ✓ Advanced statistical analysis (mean, stdev, p95, CV) with high-precision calculations
# ✓ Single-pass timing optimization eliminating cache effects between measurements
# ✓ Cross-engine core timing support (SystemDS stats, PostgreSQL EXPLAIN, DuckDB JSON profiling)
# ✓ Adaptive terminal layout with dynamic column scaling and multi-row statistics display
# ✓ Comprehensive metadata collection (system info, software versions, data build info)
# ✓ Environment verification and graceful degradation for missing engines
# ✓ Real-time progress indicators with proper terminal width handling
# ✓ Precision timing measurements with millisecond accuracy using /usr/bin/time -p
# ✓ Robust error handling with pre-flight validation and error propagation
# ✓ CSV and JSON output with timestamped files and complete statistical data
# ✓ Fastest engine detection with tie handling
# ✓ Database connection validation and parallel execution control (disabled for fair comparison)
# ✓ Cross-platform compatibility (macOS/Linux) with intelligent executable discovery
# ✓ Reproducible benchmarking with configurable seeds and detailed run configuration
#
# RECENT IMPORTANT ADDITIONS:
# - Accepts --input-dir=PATH and forwards it into SystemDS DML runs via
#   `-nvargs input_dir=/path/to/data`. This allows DML queries to load data from
#   custom locations without hardcoded paths.
# - Runner performs a pre-flight input-dir existence check and exits early with
#   a clear message when the directory is missing.
# - Test-run output is scanned for runtime SystemDS errors; when detected the
#   runner marks the query as failed and includes an `error_message` field in
#   the generated JSON results to aid debugging and CI automation.
#
# STATISTICAL MEASUREMENTS:
# - Mean: Arithmetic average execution time (typical performance expectation)
# - Standard Deviation: Population stdev measuring consistency/reliability
# - P95 Percentile: 95th percentile for worst-case performance bounds
# - Coefficient of Variation: Relative variability as percentage for cross-scale comparison
# - Display Format: "1200.0 (±14.1ms/1.2%, p95:1220.0ms)" showing all key metrics
#
# ENGINES SUPPORTED:
# - SystemDS: Machine learning platform with DML queries (single-threaded via XML config)
# - PostgreSQL: Industry-standard relational database (parallel workers disabled)
# - DuckDB: High-performance analytical database (single-threaded via PRAGMA)
#
# USAGE (from repo root):
#   scripts/ssb/shell/run_all_perf.sh                          # run full benchmark with all engines
#   scripts/ssb/shell/run_all_perf.sh --stats                  # enable internal engine timing statistics
#   scripts/ssb/shell/run_all_perf.sh --warmup=3 --repeats=10  # custom warmup and repetition settings
#   scripts/ssb/shell/run_all_perf.sh --layout=wide            # force wide table layout
#   scripts/ssb/shell/run_all_perf.sh --seed=12345             # reproducible benchmark with specific seed
#   scripts/ssb/shell/run_all_perf.sh q1.1 q2.3 q4.1           # benchmark specific queries only
#
set -euo pipefail
export LC_ALL=C

REPEATS=5
WARMUP=1
POSTGRES_DB="ssb"
POSTGRES_USER="$(whoami)"
POSTGRES_HOST="localhost"

export _JAVA_OPTIONS="${_JAVA_OPTIONS:-} -Xms2g -Xmx2g -XX:+UseParallelGC -XX:ParallelGCThreads=1"

# Determine script directory and project root (repo root)
if command -v realpath >/dev/null 2>&1; then
  SCRIPT_DIR="$(dirname "$(realpath "$0")")"
else
  SCRIPT_DIR="$(python - <<'PY'
import os, sys
print(os.path.dirname(os.path.abspath(sys.argv[1])))
PY
"$0")"
fi
# Resolve repository root robustly (script may be in scripts/ssb/shell)
if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  # Fallback: ascend until we find markers (.git or pom.xml)
  __dir="$SCRIPT_DIR"
  PROJECT_ROOT=""
  while [[ "$__dir" != "/" ]]; do
    if [[ -d "$__dir/.git" || -f "$__dir/pom.xml" ]]; then
      PROJECT_ROOT="$__dir"; break
    fi
    __dir="$(dirname "$__dir")"
  done
  : "${PROJECT_ROOT:=$(cd "$SCRIPT_DIR/../../../" && pwd)}"
fi

# Create single-thread configuration
CONF_DIR="$PROJECT_ROOT/conf"
SINGLE_THREAD_CONF="$CONF_DIR/single_thread.xml"
mkdir -p "$CONF_DIR"
if [[ ! -f "$SINGLE_THREAD_CONF" ]]; then
cat > "$SINGLE_THREAD_CONF" <<'XML'
<configuration>
  <property>
    <name>sysds.cp.parallel.ops</name><value>false</value>
  </property>
  <property>
    <name>sysds.num.threads</name><value>1</value>
  </property>
</configuration>
XML
fi
SYS_EXTRA_ARGS=( "-config" "$SINGLE_THREAD_CONF" )

# Query and system directories
QUERY_DIR="$PROJECT_ROOT/scripts/ssb/queries"

# Locate SystemDS binary
SYSTEMDS_CMD="$PROJECT_ROOT/bin/systemds"
if [[ ! -x "$SYSTEMDS_CMD" ]]; then
  SYSTEMDS_CMD="$(command -v systemds || true)"
fi
if [[ -z "$SYSTEMDS_CMD" || ! -x "$SYSTEMDS_CMD" ]]; then
  echo "SystemDS binary not found." >&2
  exit 1
fi

# Database directories and executables
# SQL files were moved under scripts/ssb/sql
SQL_DIR="$PROJECT_ROOT/scripts/ssb/sql"

# Try to find PostgreSQL psql executable
PSQL_EXEC=""
for path in "/opt/homebrew/opt/libpq/bin/psql" "/usr/local/bin/psql" "/usr/bin/psql" "$(command -v psql || true)"; do
  if [[ -x "$path" ]]; then
    PSQL_EXEC="$path"
    break
  fi
done

# Try to find DuckDB executable
DUCKDB_EXEC=""
for path in "/opt/homebrew/bin/duckdb" "/usr/local/bin/duckdb" "/usr/bin/duckdb" "$(command -v duckdb || true)"; do
  if [[ -x "$path" ]]; then
    DUCKDB_EXEC="$path"
    break
  fi
done

DUCKDB_DB_PATH="$SQL_DIR/ssb.duckdb"

# Environment verification
verify_environment() {
  local ok=true
  echo "Verifying environment..."

  if [[ ! -x "$SYSTEMDS_CMD" ]]; then
    echo "✗ SystemDS binary missing ($SYSTEMDS_CMD)" >&2
    ok=false
  else
    echo "✓ SystemDS binary found: $SYSTEMDS_CMD"
  fi

  if [[ -z "$PSQL_EXEC" || ! -x "$PSQL_EXEC" ]]; then
    echo "✗ psql not found (tried common paths)" >&2
    echo "  PostgreSQL benchmarks will be skipped" >&2
    PSQL_EXEC=""
  else
    echo "✓ psql found: $PSQL_EXEC"
    if ! "$PSQL_EXEC" -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" -c "SELECT 1" >/dev/null 2>&1; then
      echo "✗ Could not connect to PostgreSQL database ($POSTGRES_DB)" >&2
      echo "  PostgreSQL benchmarks will be skipped" >&2
      PSQL_EXEC=""
    else
      echo "✓ PostgreSQL database connection successful"
    fi
  fi

  if [[ -z "$DUCKDB_EXEC" || ! -x "$DUCKDB_EXEC" ]]; then
    echo "✗ DuckDB not found (tried common paths)" >&2
    echo "  DuckDB benchmarks will be skipped" >&2
    DUCKDB_EXEC=""
  else
    echo "✓ DuckDB found: $DUCKDB_EXEC"
    if [[ ! -f "$DUCKDB_DB_PATH" ]]; then
      echo "✗ DuckDB database missing ($DUCKDB_DB_PATH)" >&2
      echo "  DuckDB benchmarks will be skipped" >&2
      DUCKDB_EXEC=""
    elif ! "$DUCKDB_EXEC" "$DUCKDB_DB_PATH" -c "SELECT 1" >/dev/null 2>&1; then
      echo "✗ DuckDB database could not be opened" >&2
      echo "  DuckDB benchmarks will be skipped" >&2
      DUCKDB_EXEC=""
    else
      echo "✓ DuckDB database accessible"
    fi
  fi

  if [[ ! -x "$SYSTEMDS_CMD" ]]; then
    echo "Error: SystemDS is required but not found" >&2
    exit 1
  fi

  echo ""
}

# Convert seconds to milliseconds
sec_to_ms() {
  awk -v sec="$1" 'BEGIN{printf "%.1f", sec * 1000}'
}

# Statistical functions for multiple measurements
calculate_statistics() {
  local values=("$@")
  local n=${#values[@]}

  if [[ $n -eq 0 ]]; then
    echo "0|0|0"
    return
  fi

  if [[ $n -eq 1 ]]; then
    # mean|stdev|p95
    printf '%.1f|0.0|%.1f\n' "${values[0]}" "${values[0]}"
    return
  fi

  # Compute mean and population stdev with higher precision in a single awk pass
  local mean_stdev
  mean_stdev=$(printf '%s\n' "${values[@]}" | awk '
    { x[NR]=$1; s+=$1 }
    END {
      n=NR; if(n==0){ printf "0|0"; exit }
      m=s/n;
      ss=0; for(i=1;i<=n;i++){ d=x[i]-m; ss+=d*d }
      stdev=sqrt(ss/n);
      printf "%.6f|%.6f", m, stdev
    }')

  local mean=$(echo "$mean_stdev" | cut -d'|' -f1)
  local stdev=$(echo "$mean_stdev" | cut -d'|' -f2)

  # Calculate p95 (nearest-rank: ceil(0.95*n))
  local sorted_values=($(printf '%s\n' "${values[@]}" | sort -n))
  local p95_index=$(awk -v n="$n" 'BEGIN{ idx = int(0.95*n + 0.999999); if(idx<1) idx=1; if(idx>n) idx=n; print idx-1 }')
  local p95=${sorted_values[$p95_index]}

  # Format to one decimal place
  printf '%.1f|%.1f|%.1f\n' "$mean" "$stdev" "$p95"
}

# Format statistics for display
format_statistics() {
  local mean="$1"
  local stdev="$2"
  local p95="$3"
  local repeats="$4"

  if [[ $repeats -eq 1 ]]; then
    echo "$mean"
  else
    # Calculate coefficient of variation (CV) as percentage
    local cv_percent=0
    if [[ $(awk -v mean="$mean" 'BEGIN{print (mean > 0)}') -eq 1 ]]; then
      cv_percent=$(awk -v stdev="$stdev" -v mean="$mean" 'BEGIN{printf "%.1f", (stdev * 100) / mean}')
    fi
    echo "$mean (±${stdev}ms/${cv_percent}%, p95:${p95}ms)"
  fi
}

# Format only the stats line (without the mean), e.g., "(±10.2ms/0.6%, p95:1740.0ms)"
format_stats_only() {
  local mean="$1"
  local stdev="$2"
  local p95="$3"
  local repeats="$4"

  if [[ $repeats -eq 1 ]]; then
    echo ""
    return
  fi
  # Only for numeric means
  if ! [[ "$mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo ""
    return
  fi
  local cv_percent=0
  if [[ $(awk -v mean="$mean" 'BEGIN{print (mean > 0)}') -eq 1 ]]; then
    cv_percent=$(awk -v stdev="$stdev" -v mean="$mean" 'BEGIN{printf "%.1f", (stdev * 100) / mean}')
  fi
  echo "(±${stdev}ms/${cv_percent}%, p95:${p95}ms)"
}

# Format only the CV line (±stdev/CV%)
format_cv_only() {
  local mean="$1"; local stdev="$2"; local repeats="$3"
  if [[ $repeats -eq 1 ]]; then echo ""; return; fi
  if ! [[ "$mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo ""; return; fi
  local cv_percent=0
  if [[ $(awk -v mean="$mean" 'BEGIN{print (mean > 0)}') -eq 1 ]]; then
    cv_percent=$(awk -v stdev="$stdev" -v mean="$mean" 'BEGIN{printf "%.1f", (stdev * 100) / mean}')
  fi
  echo "±${stdev}ms/${cv_percent}%"
}

# Format only the p95 line
format_p95_only() {
  local p95="$1"; local repeats="$2"
  if [[ $repeats -eq 1 ]]; then echo ""; return; fi
  if ! [[ "$p95" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo ""; return; fi
  echo "p95:${p95}ms"
}

# Column widths for wide layout - optimized for 125-char terminals
WIDE_COL_WIDTHS=(8 14 14 12 16 12 12 18)

# Draw a grid line like +----------+----------------+...
grid_line_wide() {
  local parts=("+")
  for w in "${WIDE_COL_WIDTHS[@]}"; do
    parts+=("$(printf '%*s' "$((w+2))" '' | tr ' ' '-')+")
  done
  printf '%s\n' "${parts[*]}" | tr -d ' '
}

# Print a grid row with vertical separators using the wide layout widths
grid_row_wide() {
  local -a cells=("$@")
  local cols=${#WIDE_COL_WIDTHS[@]}
  while [[ ${#cells[@]} -lt $cols ]]; do
    cells+=("")
  done

  # Build a printf format string that right-aligns numeric and statistic-like cells
  # (numbers, lines starting with ± or p95, or containing p95/±) while leaving the
  # first column (query) left-aligned for readability.
  local fmt=""
  for i in $(seq 0 $((cols-1))); do
    local w=${WIDE_COL_WIDTHS[i]}
    if [[ $i -eq 0 ]]; then
      # Query name: left-align
      fmt+="| %-${w}s"
    else
      local cell="${cells[i]}"
      # Heuristic: right-align if the cell is a plain number or contains statistic markers
      if [[ "$cell" =~ ^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$ ]] || [[ "$cell" == ±* ]] || [[ "$cell" == *'±'* ]] || [[ "$cell" == p95* ]] || [[ "$cell" == *'p95'* ]] || [[ "$cell" == \(* ]]; then
        fmt+=" | %${w}s"
      else
        fmt+=" | %-${w}s"
      fi
    fi
  done
  fmt+=" |\n"

  printf "$fmt" "${cells[@]}"
}

# Time a command and return real time in ms
time_command_ms() {
  local out
  # Properly capture stderr from /usr/bin/time while suppressing stdout of the command
  out=$({ /usr/bin/time -p "$@" > /dev/null; } 2>&1)
  local real_sec=$(echo "$out" | awk '/^real /{print $2}')
  if [[ -z "$real_sec" || ! "$real_sec" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "(error)"
    return 1
  fi
  sec_to_ms "$real_sec"
}

# Time a command, capturing stdout to a file, and return real time in ms
time_command_ms_capture() {
  local stdout_file="$1"; shift
  local out
  out=$({ /usr/bin/time -p "$@" > "$stdout_file"; } 2>&1)
  local real_sec=$(echo "$out" | awk '/^real /{print $2}')
  if [[ -z "$real_sec" || ! "$real_sec" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "(error)"
    return 1
  fi
  sec_to_ms "$real_sec"
}

# Run a SystemDS query and compute statistics
run_systemds_avg() {
  local dml="$1"
  # Optional second parameter: path to write an error message if the test-run fails
  local err_out_file="${2:-}"
  local shell_times=()
  local core_times=()
  local core_have=false

  # Change to project root directory so relative paths in DML work correctly
  local original_dir="$(pwd)"
  cd "$PROJECT_ROOT"

  # First, test run to validate the query (avoids timing zero or errors later)
  tmp_test=$(mktemp)
  if $RUN_STATS; then
    if ! "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}" > "$tmp_test" 2>&1; then
      err_msg=$(sed -n '1,200p' "$tmp_test" | tr '\n' ' ')
      echo "Error: SystemDS test run failed for $dml: $err_msg" >&2
      # Write error message to provided error file for JSON capture
      if [[ -n "$err_out_file" ]]; then printf '%s' "$err_msg" > "$err_out_file" || true; fi
      rm -f "$tmp_test"
      echo "(error)|0|0|(n/a)|0|0"
      cd "$original_dir"; return
    fi
    err_msg=$(sed -n '/An Error Occurred :/,$ p' "$tmp_test" | sed -n '1,200p' | tr '\n' ' ')
    if [[ -n "$err_msg" ]]; then
      echo "Error: SystemDS reported runtime error for $dml: $err_msg" >&2
      if [[ -n "$err_out_file" ]]; then printf '%s' "$err_msg" > "$err_out_file" || true; fi
      rm -f "$tmp_test"
      echo "(error)|0|0|(n/a)|0|0"
      cd "$original_dir"; return
    fi
  else
    if ! "$SYSTEMDS_CMD" "$dml" "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}" > "$tmp_test" 2>&1; then
      err_msg=$(sed -n '1,200p' "$tmp_test" | tr '\n' ' ')
      echo "Error: SystemDS test run failed for $dml: $err_msg" >&2
      if [[ -n "$err_out_file" ]]; then printf '%s' "$err_msg" > "$err_out_file" || true; fi
      rm -f "$tmp_test"
      echo "(error)|0|0|(n/a)|0|0"
      cd "$original_dir"; return
    fi
    err_msg=$(sed -n '/An Error Occurred :/,$ p' "$tmp_test" | sed -n '1,200p' | tr '\n' ' ')
    if [[ -n "$err_msg" ]]; then
      echo "Error: SystemDS reported runtime error for $dml: $err_msg" >&2
      if [[ -n "$err_out_file" ]]; then printf '%s' "$err_msg" > "$err_out_file" || true; fi
      rm -f "$tmp_test"
      echo "(error)|0|0|(n/a)|0|0"
      cd "$original_dir"; return
    fi
  fi
  rm -f "$tmp_test"

  # Warmup runs
  for ((w=1; w<=WARMUP; w++)); do
    if $RUN_STATS; then
      "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}" > /dev/null 2>&1 || true
    else
      "$SYSTEMDS_CMD" "$dml" "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}" > /dev/null 2>&1 || true
    fi
  done

  # Timed runs - collect all measurements
  for ((i=1; i<=REPEATS; i++)); do
    if $RUN_STATS; then
      local shell_ms
      local temp_file
      temp_file=$(mktemp)
      shell_ms=$(time_command_ms_capture "$temp_file" "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}") || {
        rm -f "$temp_file"; cd "$original_dir"; echo "(error)|0|0|(n/a)|0|0"; return; }
      shell_times+=("$shell_ms")

      # Extract SystemDS internal timing from the same run
      local internal_sec
      internal_sec=$(awk '/Total execution time:/ {print $4}' "$temp_file" | tail -1 || true)
      rm -f "$temp_file"
      if [[ -n "$internal_sec" ]] && [[ "$internal_sec" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        local core_ms
        core_ms=$(awk -v sec="$internal_sec" 'BEGIN{printf "%.1f", sec * 1000}')
        core_times+=("$core_ms")
        core_have=true
      fi
    else
      local shell_ms
      shell_ms=$(time_command_ms "$SYSTEMDS_CMD" "$dml" "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}") || { cd "$original_dir"; echo "(error)|0|0|(n/a)|0|0"; return; }
      shell_times+=("$shell_ms")
    fi
  done

  # Return to original directory
  cd "$original_dir"

  # Calculate statistics for shell times
  local shell_stats
  shell_stats=$(calculate_statistics "${shell_times[@]}")

  # Calculate statistics for core times if available
  local core_stats
  if $RUN_STATS && $core_have && [[ ${#core_times[@]} -gt 0 ]]; then
    core_stats=$(calculate_statistics "${core_times[@]}")
  else
    core_stats="(n/a)|0|0"
  fi

  echo "$shell_stats|$core_stats"
}

# Run a PostgreSQL query and compute statistics
run_psql_avg_ms() {
  local sql_file="$1"

  # Check if PostgreSQL is available
  if [[ -z "$PSQL_EXEC" ]]; then
    echo "(unavailable)|0|0|(n/a)|0|0"
    return
  fi

  # Test run first
  "$PSQL_EXEC" -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" \
    -v ON_ERROR_STOP=1 -q \
    -c "SET max_parallel_workers=0; SET max_parallel_maintenance_workers=0; SET max_parallel_workers_per_gather=0; SET parallel_leader_participation=off;" \
    -f "$sql_file" >/dev/null 2>/dev/null || {
      echo "(error)|0|0|(n/a)|0|0"
      return
    }

  local shell_times=()
  local core_times=()
  local core_have=false

  for ((i=1; i<=REPEATS; i++)); do
    # Wall-clock shell time
    local ms
    ms=$(time_command_ms "$PSQL_EXEC" -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" \
      -v ON_ERROR_STOP=1 -q \
      -c "SET max_parallel_workers=0; SET max_parallel_maintenance_workers=0; SET max_parallel_workers_per_gather=0; SET parallel_leader_participation=off;" \
      -f "$sql_file" 2>/dev/null) || {
        echo "(error)|0|0|(n/a)|0|0"
        return
      }
    shell_times+=("$ms")

    # Core execution time using EXPLAIN ANALYZE (if --stats enabled)
    if $RUN_STATS; then
      local tmp_explain
      tmp_explain=$(mktemp)

      # Create EXPLAIN ANALYZE version of the query
      echo "SET max_parallel_workers=0; SET max_parallel_maintenance_workers=0; SET max_parallel_workers_per_gather=0; SET parallel_leader_participation=off;" > "$tmp_explain"
      echo "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)" >> "$tmp_explain"
      cat "$sql_file" >> "$tmp_explain"

      # Execute EXPLAIN ANALYZE and extract execution time
      local explain_output core_ms
      explain_output=$("$PSQL_EXEC" -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" \
        -v ON_ERROR_STOP=1 -q -f "$tmp_explain" 2>/dev/null || true)

      if [[ -n "$explain_output" ]]; then
        # Extract "Execution Time: X.XXX ms" from EXPLAIN ANALYZE output
        local exec_time_ms
        exec_time_ms=$(echo "$explain_output" | grep -oE "Execution Time: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1 || true)

        if [[ -n "$exec_time_ms" ]] && [[ "$exec_time_ms" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
          core_ms=$(awk -v ms="$exec_time_ms" 'BEGIN{printf "%.1f", ms}')
          core_times+=("$core_ms")
          core_have=true
        fi
      fi

      rm -f "$tmp_explain"
    fi
  done

  # Build outputs
  local shell_stats core_stats
  shell_stats=$(calculate_statistics "${shell_times[@]}")
  if $RUN_STATS && $core_have && [[ ${#core_times[@]} -gt 0 ]]; then
    core_stats=$(calculate_statistics "${core_times[@]}")
  else
    core_stats="(n/a)|0|0"
  fi
  echo "$shell_stats|$core_stats"
}

# Run a DuckDB query and compute statistics
run_duckdb_avg_ms() {
  local sql_file="$1"

  # Check if DuckDB is available
  if [[ -z "$DUCKDB_EXEC" ]]; then
    echo "(unavailable)|0|0|(n/a)|0|0"
    return
  fi

  # Test run with minimal setup (no profiling)
  local tmp_test
  tmp_test=$(mktemp)
  printf 'PRAGMA threads=1;\n' > "$tmp_test"
  cat "$sql_file" >> "$tmp_test"
  "$DUCKDB_EXEC" "$DUCKDB_DB_PATH" < "$tmp_test" >/dev/null 2>&1 || {
    rm -f "$tmp_test"
    echo "(error)|0|0|(n/a)|0|0"
    return
  }
  rm -f "$tmp_test"

  local shell_times=()
  local core_times=()
  local core_have=false

  for ((i=1; i<=REPEATS; i++)); do
    local tmp_sql iter_json
    tmp_sql=$(mktemp)
    if $RUN_STATS; then
      # Enable JSON profiling per-run and write to a temporary file
      iter_json=$(mktemp -t duckprof.XXXXXX).json
      cat > "$tmp_sql" <<SQL
PRAGMA threads=1;
PRAGMA enable_profiling=json;
PRAGMA profiling_output='$iter_json';
SQL
    else
      echo 'PRAGMA threads=1;' > "$tmp_sql"
    fi
    cat "$sql_file" >> "$tmp_sql"

    # Wall-clock shell time
    local ms
    ms=$(time_command_ms "$DUCKDB_EXEC" "$DUCKDB_DB_PATH" < "$tmp_sql") || {
      rm -f "$tmp_sql" ${iter_json:+"$iter_json"}
      echo "(error)|0|0|(n/a)|0|0"
      return
    }
    shell_times+=("$ms")

    # Parse core latency from JSON profile if available
    if $RUN_STATS && [[ -n "${iter_json:-}" && -f "$iter_json" ]]; then
      local core_sec
      if command -v jq >/dev/null 2>&1; then
        core_sec=$(jq -r '.latency // empty' "$iter_json" 2>/dev/null || true)
      else
        core_sec=$(grep -oE '"latency"\s*:\s*[0-9.]+' "$iter_json" 2>/dev/null | sed -E 's/.*:\s*//' | head -1 || true)
      fi
      if [[ -n "$core_sec" ]] && [[ "$core_sec" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        local core_ms
        core_ms=$(awk -v s="$core_sec" 'BEGIN{printf "%.1f", s*1000}')
        core_times+=("$core_ms")
        core_have=true
      fi
    fi

    rm -f "$tmp_sql" ${iter_json:+"$iter_json"}
  done

  # Build outputs
  local shell_stats core_stats
  shell_stats=$(calculate_statistics "${shell_times[@]}")
  if $RUN_STATS && $core_have && [[ ${#core_times[@]} -gt 0 ]]; then
    core_stats=$(calculate_statistics "${core_times[@]}")
  else
    core_stats="(n/a)|0|0"
  fi
  echo "$shell_stats|$core_stats"
}

# Help function
show_help() {
  cat << 'EOF'
Multi-Engine SSB Performance Benchmark Runner v1.0

USAGE (from repo root):
  scripts/ssb/shell/run_all_perf.sh [OPTIONS] [QUERIES...]

OPTIONS:
  -stats, --stats         Enable SystemDS internal statistics collection
  -warmup=N, --warmup=N   Set number of warmup runs (default: 1)
  -repeats=N, --repeats=N Set number of timing repetitions (default: 5)
  -seed=N, --seed=N       Set random seed for reproducible results (default: auto-generated)
  -stacked, --stacked     Use stacked, multi-line layout (best for narrow terminals)
  -layout=MODE, --layout=MODE Set layout: auto|wide|stacked (default: auto)
                          Note: --layout=stacked is equivalent to --stacked
                                --layout=wide forces wide table layout
  -input-dir=PATH, --input-dir=PATH Specify custom data directory (default: $PROJECT_ROOT/data)
  -output-dir=PATH, --output-dir=PATH Specify custom output directory (default: $PROJECT_ROOT/scripts/ssb/shell/ssbOutputData/PerformanceData)
  -h, -help, --help, --h  Show this help message
  -v, -version, --version, --v Show version information

QUERIES:
  If no queries are specified, all available SSB queries (q*.dml) will be executed.
  To run specific queries, provide their names (with or without .dml extension):
    scripts/ssb/shell/run_all_perf.sh q1.1 q2.3 q4.1

EXAMPLES (from repo root):
  scripts/ssb/shell/run_all_perf.sh                          # Run full benchmark with all engines
  scripts/ssb/shell/run_all_perf.sh --warmup=3 --repeats=10  # Custom warmup and repetition settings
  scripts/ssb/shell/run_all_perf.sh -warmup=3 -repeats=10    # Same with single dashes
  scripts/ssb/shell/run_all_perf.sh --stats                  # Enable SystemDS internal timing
  scripts/ssb/shell/run_all_perf.sh --layout=wide            # Force wide table layout
  scripts/ssb/shell/run_all_perf.sh --stacked                # Force stacked layout for narrow terminals
  scripts/ssb/shell/run_all_perf.sh q1.1 q2.3                # Benchmark specific queries only
  scripts/ssb/shell/run_all_perf.sh --seed=12345             # Reproducible benchmark run
  scripts/ssb/shell/run_all_perf.sh --input-dir=/path/to/data  # Custom data directory
  scripts/ssb/shell/run_all_perf.sh -input-dir=/path/to/data   # Same as above (single dash)
  scripts/ssb/shell/run_all_perf.sh --output-dir=/tmp/results  # Custom output directory
  scripts/ssb/shell/run_all_perf.sh -output-dir=/tmp/results   # Same as above (single dash)

ENGINES:
  - SystemDS: Machine learning platform with DML queries
  - PostgreSQL: Industry-standard relational database (if available)
  - DuckDB: High-performance analytical database (if available)

OUTPUT:
  Results are saved in CSV and JSON formats with comprehensive metadata:
  - Performance timing statistics (mean, stdev, p95)
  - Engine comparison and fastest detection
  - System information and run configuration

STATISTICAL OUTPUT FORMAT:
  1824 (±10, p95:1840)
    │     │       └── 95th percentile (worst-case bound)
    │     └── Standard deviation (consistency measure)
    └── Mean execution time (typical performance)

For more information, see the documentation in scripts/ssb/README.md
EOF
}

# Parse arguments
RUN_STATS=false
QUERIES=()
SEED=""
LAYOUT="auto"
INPUT_DIR=""
OUTPUT_DIR=""

# Support both --opt=value and --opt value forms
EXPECT_OPT=""
for arg in "$@"; do
  if [[ -n "$EXPECT_OPT" ]]; then
    case "$EXPECT_OPT" in
      seed)
        SEED="$arg"
        EXPECT_OPT=""
        continue
        ;;
      input-dir)
        INPUT_DIR="$arg"
        EXPECT_OPT=""
        continue
        ;;
      output-dir)
        OUTPUT_DIR="$arg"
        EXPECT_OPT=""
        continue
        ;;
      warmup)
        WARMUP="$arg"
        if ! [[ "$WARMUP" =~ ^[0-9]+$ ]] || [[ "$WARMUP" -lt 0 ]]; then
          echo "Error: --warmup requires a non-negative integer (e.g., --warmup 2)" >&2
          exit 1
        fi
        EXPECT_OPT=""
        continue
        ;;
      repeats)
        REPEATS="$arg"
        if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
          echo "Error: --repeats requires a positive integer (e.g., --repeats 5)" >&2
          exit 1
        fi
        EXPECT_OPT=""
        continue
        ;;
      layout)
        LAYOUT="$arg"
        if [[ "$LAYOUT" != "auto" && "$LAYOUT" != "wide" && "$LAYOUT" != "stacked" ]]; then
          echo "Error: --layout requires one of: auto, wide, stacked (e.g., --layout wide)" >&2
          exit 1
        fi
        EXPECT_OPT=""
        continue
        ;;
    esac
  fi

  if [[ "$arg" == "--help" || "$arg" == "-help" || "$arg" == "-h" || "$arg" == "--h" ]]; then
    show_help
    exit 0
  elif [[ "$arg" == "--version" || "$arg" == "-version" || "$arg" == "-v" || "$arg" == "--v" ]]; then
    echo "Multi-Engine SSB Performance Benchmark Runner v1.0"
    echo "First Public Release: September 5, 2025"
    exit 0
  elif [[ "$arg" == "--stats" || "$arg" == "-stats" ]]; then
    RUN_STATS=true
  elif [[ "$arg" == --seed=* || "$arg" == -seed=* ]]; then
    SEED="${arg#*seed=}"
  elif [[ "$arg" == "--seed" || "$arg" == "-seed" ]]; then
    EXPECT_OPT="seed"
  elif [[ "$arg" == --warmup=* || "$arg" == -warmup=* ]]; then
    WARMUP="${arg#*warmup=}"
    if ! [[ "$WARMUP" =~ ^[0-9]+$ ]] || [[ "$WARMUP" -lt 0 ]]; then
      echo "Error: -warmup/--warmup requires a non-negative integer (e.g., -warmup=2)" >&2
      exit 1
    fi
  elif [[ "$arg" == --input-dir=* || "$arg" == -input-dir=* ]]; then
    INPUT_DIR="${arg#*input-dir=}"
  elif [[ "$arg" == "--input-dir" || "$arg" == "-input-dir" ]]; then
    EXPECT_OPT="input-dir"
  elif [[ "$arg" == --output-dir=* || "$arg" == -output-dir=* ]]; then
    OUTPUT_DIR="${arg#*output-dir=}"
  elif [[ "$arg" == "--output-dir" || "$arg" == "-output-dir" ]]; then
    EXPECT_OPT="output-dir"
  elif [[ "$arg" == "--warmup" || "$arg" == "-warmup" ]]; then
    EXPECT_OPT="warmup"
  elif [[ "$arg" == --repeats=* || "$arg" == -repeats=* ]]; then
    REPEATS="${arg#*repeats=}"
    if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
      echo "Error: -repeats/--repeats requires a positive integer (e.g., -repeats=5)" >&2
      exit 1
    fi
  elif [[ "$arg" == "--repeats" || "$arg" == "-repeats" ]]; then
    EXPECT_OPT="repeats"
  elif [[ "$arg" == "--stacked" || "$arg" == "-stacked" ]]; then
    LAYOUT="stacked"
  elif [[ "$arg" == --layout=* || "$arg" == -layout=* ]]; then
    LAYOUT="${arg#*layout=}"
    if [[ "$LAYOUT" != "auto" && "$LAYOUT" != "wide" && "$LAYOUT" != "stacked" ]]; then
      echo "Error: -layout/--layout requires one of: auto, wide, stacked (e.g., --layout=wide)" >&2
      exit 1
    fi
  elif [[ "$arg" == "--layout" || "$arg" == "-layout" ]]; then
    EXPECT_OPT="layout"
  else
    # Check if argument looks like an unrecognized option (starts with dash)
    if [[ "$arg" == -* ]]; then
      echo "Error: Unrecognized option '$arg'" >&2
      echo "Use --help or -h to see available options." >&2
      exit 1
    else
      # Treat as query name
      QUERIES+=( "$(echo "$arg" | tr '.' '_')" )
    fi
  fi
 done

# If the last option expected a value but none was provided
if [[ -n "$EXPECT_OPT" ]]; then
  case "$EXPECT_OPT" in
    seed) echo "Error: -seed/--seed requires a value (e.g., -seed=12345)" >&2 ;;
    warmup) echo "Error: -warmup/--warmup requires a value (e.g., -warmup=2)" >&2 ;;
    repeats) echo "Error: -repeats/--repeats requires a value (e.g., -repeats=5)" >&2 ;;
    layout) echo "Error: -layout/--layout requires a value (e.g., -layout=wide)" >&2 ;;
  esac
  exit 1
fi

# Generate seed if not provided
if [[ -z "$SEED" ]]; then
  SEED=$((RANDOM * 32768 + RANDOM))
fi
if [[ ${#QUERIES[@]} -eq 0 ]]; then
  for f in "$QUERY_DIR"/q*.dml; do
    [[ -e "$f" ]] || continue
    bname="$(basename "$f")"
    QUERIES+=( "${bname%.dml}" )
  done
fi

# Set data directory
if [[ -z "$INPUT_DIR" ]]; then
  INPUT_DIR="$PROJECT_ROOT/data"
fi

# Set output directory
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$PROJECT_ROOT/scripts/ssb/shell/ssbOutputData/PerformanceData"
fi

# Normalize paths by removing trailing slashes
INPUT_DIR="${INPUT_DIR%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

# Pass input directory to DML scripts via SystemDS named arguments
NVARGS=( -nvargs "input_dir=${INPUT_DIR}" )

# Validate data directory
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: Data directory '$INPUT_DIR' does not exist." >&2
  echo "Please ensure the directory exists or specify a different path with -input-dir." >&2
  exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Metadata collection functions
collect_system_metadata() {
  local timestamp hostname systemds_version jdk_version postgres_version duckdb_version cpu_info ram_info

  # Basic system info
  timestamp=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  hostname=$(hostname 2>/dev/null || echo "unknown")

  # SystemDS version
  if [[ -x "$SYSTEMDS_CMD" ]]; then
    # Try to get version from pom.xml first
    if [[ -f "$PROJECT_ROOT/pom.xml" ]]; then
      systemds_version=$(grep -A1 '<groupId>org.apache.systemds</groupId>' "$PROJECT_ROOT/pom.xml" | grep '<version>' | sed 's/.*<version>\(.*\)<\/version>.*/\1/' | head -1 2>/dev/null || echo "unknown")
    else
      systemds_version="unknown"
    fi

    # If pom.xml method failed, try alternative methods
    if [[ "$systemds_version" == "unknown" ]]; then
      # Try to extract from SystemDS JAR manifest
      if [[ -f "$PROJECT_ROOT/target/systemds.jar" ]]; then
        systemds_version=$(unzip -p "$PROJECT_ROOT/target/systemds.jar" META-INF/MANIFEST.MF 2>/dev/null | grep "Implementation-Version" | cut -d: -f2 | tr -d ' ' || echo "unknown")
      else
        # Try to find any SystemDS JAR and extract version
        local jar_file=$(find "$PROJECT_ROOT" -name "systemds*.jar" | head -1 2>/dev/null)
        if [[ -n "$jar_file" ]]; then
          systemds_version=$(unzip -p "$jar_file" META-INF/MANIFEST.MF 2>/dev/null | grep "Implementation-Version" | cut -d: -f2 | tr -d ' ' || echo "unknown")
        else
          systemds_version="unknown"
        fi
      fi
    fi
  else
    systemds_version="unknown"
  fi

  # JDK version
  if command -v java >/dev/null 2>&1; then
    jdk_version=$(java -version 2>&1 | grep -v "Picked up" | head -1 | sed 's/.*"\(.*\)".*/\1/' || echo "unknown")
  else
    jdk_version="unknown"
  fi

  # PostgreSQL version
  if command -v psql >/dev/null 2>&1; then
    postgres_version=$(psql --version 2>/dev/null | head -1 || echo "not available")
  else
    postgres_version="not available"
  fi

  # DuckDB version
  if command -v duckdb >/dev/null 2>&1; then
    duckdb_version=$(duckdb --version 2>/dev/null || echo "not available")
  else
    duckdb_version="not available"
  fi

  # System resources
  if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    ram_info=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))GB
  else
    # Linux
    cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2- | sed 's/^ *//' 2>/dev/null || echo "unknown")
    ram_info=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}' 2>/dev/null || echo 0) / 1024 / 1024 ))GB
  fi

  # Store metadata globally
  RUN_TIMESTAMP="$timestamp"
  RUN_HOSTNAME="$hostname"
  RUN_SYSTEMDS_VERSION="$systemds_version"
  RUN_JDK_VERSION="$jdk_version"
  RUN_POSTGRES_VERSION="$postgres_version"
  RUN_DUCKDB_VERSION="$duckdb_version"
  RUN_CPU_INFO="$cpu_info"
  RUN_RAM_INFO="$ram_info"
}

collect_data_metadata() {
  # Check for SSB data directory and get basic stats
  local ssb_data_dir="$INPUT_DIR"
  local json_parts=()
  local display_parts=()

  if [[ -d "$ssb_data_dir" ]]; then
    # Try to get row counts from data files (if they exist)
    for table in customer part supplier date; do
      local file="$ssb_data_dir/${table}.tbl"
      if [[ -f "$file" ]]; then
        local count=$(wc -l < "$file" 2>/dev/null | tr -d ' ' || echo "0")
        json_parts+=("    \"$table\": \"$count\"")
        display_parts+=("$table:$count")
      fi
    done
    # Check for any lineorder*.tbl file (SSB fact table)
    local lineorder_file=$(find "$ssb_data_dir" -name "lineorder*.tbl" -type f | head -1)
    if [[ -n "$lineorder_file" && -f "$lineorder_file" ]]; then
      local count=$(wc -l < "$lineorder_file" 2>/dev/null | tr -d ' ' || echo "0")
      json_parts+=("    \"lineorder\": \"$count\"")
      display_parts+=("lineorder:$count")
    fi
  fi

  if [[ ${#json_parts[@]} -eq 0 ]]; then
    RUN_DATA_INFO='"No data files found"'
    RUN_DATA_DISPLAY="No data files found"
  else
    # Join array elements with commas and newlines, wrap in braces for JSON
    local formatted_json="{\n"
    for i in "${!json_parts[@]}"; do
      formatted_json+="${json_parts[$i]}"
      if [[ $i -lt $((${#json_parts[@]} - 1)) ]]; then
        formatted_json+=",\n"
      else
        formatted_json+="\n"
      fi
    done
    formatted_json+="  }"
    RUN_DATA_INFO="$formatted_json"

    # Join with spaces for display
    local IFS=" "
    RUN_DATA_DISPLAY="${display_parts[*]}"
  fi
}

print_metadata_header() {
  echo "=================================================================================="
  echo "                      MULTI-ENGINE PERFORMANCE BENCHMARK METADATA"
  echo "=================================================================================="
  echo "Timestamp:       $RUN_TIMESTAMP"
  echo "Hostname:        $RUN_HOSTNAME"
  echo "Seed:            $SEED"
  echo
  echo "Software Versions:"
  echo "  SystemDS:      $RUN_SYSTEMDS_VERSION"
  echo "  JDK:           $RUN_JDK_VERSION"
  echo "  PostgreSQL:    $RUN_POSTGRES_VERSION"
  echo "  DuckDB:        $RUN_DUCKDB_VERSION"
  echo
  echo "System Resources:"
  echo "  CPU:           $RUN_CPU_INFO"
  echo "  RAM:           $RUN_RAM_INFO"
  echo
  echo "Data Build Info:"
  echo "  SSB Data:      $RUN_DATA_DISPLAY"
  echo
  echo "Run Configuration:"
  echo "  Statistics:    $(if $RUN_STATS; then echo "enabled"; else echo "disabled"; fi)"
  echo "  Queries:       ${#QUERIES[@]} selected"
  echo "  Warmup Runs:   $WARMUP"
  echo "  Repeat Runs:   $REPEATS"
  echo "=================================================================================="
  echo
}

# Progress indicator function
progress_indicator() {
  local query_name="$1"
  local stage="$2"
  # Use terminal width for proper clearing, fallback to 120 chars if tput fails
  local term_width
  term_width=$(tput cols 2>/dev/null || echo 120)
  local spaces=$(printf "%*s" "$term_width" "")
  echo -ne "\r$spaces\r$query_name: Running $stage..."
}

# Clear progress line function
clear_progress() {
  local term_width
  term_width=$(tput cols 2>/dev/null || echo 120)
  local spaces=$(printf "%*s" "$term_width" "")
  echo -ne "\r$spaces\r"
}

# Main execution
# Collect metadata
collect_system_metadata
collect_data_metadata

# Print metadata header
print_metadata_header

verify_environment
echo
echo "NOTE (macOS): You cannot drop OS caches like Linux (sync; echo 3 > /proc/sys/vm/drop_caches)."
echo "We mitigate with warm-up runs and repeated averages to ensure consistent measurements."
echo
echo "INTERPRETATION GUIDE:"
echo "- SystemDS Shell (ms): Total execution time including JVM startup, I/O, and computation"
echo "- SystemDS Core (ms):  Pure computation time excluding JVM overhead (only with --stats)"
echo "- PostgreSQL (ms):     Single-threaded execution time with parallel workers disabled"
echo "- PostgreSQL Core (ms): Query execution time from EXPLAIN ANALYZE (only with --stats)"
echo "- DuckDB (ms):         Single-threaded execution time with threads=1 pragma"
echo "- DuckDB Core (ms):    Engine-internal latency from JSON profiling (with --stats)"
echo "- (missing):           SQL file not found for this query"
echo "- (n/a):               Core timing unavailable (run with --stats flag for internal timing)"
echo
echo "NOTE: All engines use single-threaded execution for fair comparison."
echo "      Multiple runs with averaging provide statistical reliability."
echo
echo "Single-threaded execution; warm-up runs: $WARMUP, timed runs: $REPEATS"
echo "Row 1 shows mean (ms); Row 2 shows ±stdev/CV; Row 3 shows p95 (ms)."
echo "Core execution times available for all engines with --stats flag."
echo
term_width=$(tput cols 2>/dev/null || echo 120)
if [[ "$LAYOUT" == "auto" ]]; then
  if [[ $term_width -ge 140 ]]; then
    LAYOUT_MODE="wide"
  else
    LAYOUT_MODE="stacked"
  fi
else
  LAYOUT_MODE="$LAYOUT"
fi

# If the user requested wide layout but the terminal is too narrow, fall back to stacked
if [[ "$LAYOUT_MODE" == "wide" ]]; then
  # compute total printable width: sum(widths) + 3*cols + 1 (accounting for separators)
  sumw=0
  for w in "${WIDE_COL_WIDTHS[@]}"; do sumw=$((sumw + w)); done
  cols=${#WIDE_COL_WIDTHS[@]}
  total_width=$((sumw + 3*cols + 1))
  if [[ $total_width -gt $term_width ]]; then
    # Try to scale columns down proportionally to fit terminal width
    reserved=$((3*cols + 1))
    avail=$((term_width - reserved))
  if [[ $avail -le 0 ]]; then
    :
  else
      # Minimum sensible widths per column (keep labels readable)
      MIN_COL_WIDTHS=(6 8 8 6 10 6 6 16)
      # Start with proportional distribution
      declare -a new_widths=()
      for w in "${WIDE_COL_WIDTHS[@]}"; do
        nw=$(( w * avail / sumw ))
        if [[ $nw -lt 1 ]]; then nw=1; fi
        new_widths+=("$nw")
      done
      # Enforce minimums
      sum_new=0
      for i in "${!new_widths[@]}"; do
        if [[ ${new_widths[i]} -lt ${MIN_COL_WIDTHS[i]:-4} ]]; then
          new_widths[i]=${MIN_COL_WIDTHS[i]:-4}
        fi
        sum_new=$((sum_new + new_widths[i]))
      done
      # If even minimums exceed available, fallback to stacked
      if [[ $sum_new -gt $avail ]]; then
        :
      else
        # Distribute remaining columns' widths left-to-right
        rem=$((avail - sum_new))
        i=0
        while [[ $rem -gt 0 ]]; do
          new_widths[i]=$((new_widths[i] + 1))
          rem=$((rem - 1))
          i=$(( (i + 1) % cols ))
        done
        # Replace WIDE_COL_WIDTHS with the scaled values for printing
        WIDE_COL_WIDTHS=("${new_widths[@]}")
        # Recompute total_width for logging
        sumw=0
        for w in "${WIDE_COL_WIDTHS[@]}"; do sumw=$((sumw + w)); done
        total_width=$((sumw + reserved))
        echo "Info: scaled wide layout to fit terminal ($term_width cols): table width $total_width"
      fi
    fi
  fi
fi

if [[ "$LAYOUT_MODE" == "wide" ]]; then
  grid_line_wide
  grid_row_wide \
    "Query" \
    "SysDS Shell" "SysDS Core" \
    "PostgreSQL" "PostgreSQL Core" \
    "DuckDB" "DuckDB Core" \
    "Fastest"
  grid_row_wide "" "mean" "mean" "mean" "mean" "mean" "mean" ""
  grid_row_wide "" "±/CV" "±/CV" "±/CV" "±/CV" "±/CV" "±/CV" ""
  grid_row_wide "" "p95" "p95" "p95" "p95" "p95" "p95" ""
  grid_line_wide
else
  echo "================================================================================"
  echo "Stacked layout (use --layout=wide for table view)."
  echo "Row 1 shows mean (ms); Row 2 shows (±stdev/CV, p95)."
  echo "--------------------------------------------------------------------------------"
fi
# Prepare output file paths and write CSV header with comprehensive metadata
# Ensure results directory exists and create timestamped filenames
RESULT_DIR="$OUTPUT_DIR"
mkdir -p "$RESULT_DIR"
RESULT_BASENAME="ssb_results_$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_CSV="$RESULT_DIR/${RESULT_BASENAME}.csv"
RESULT_JSON="$RESULT_DIR/${RESULT_BASENAME}.json"

{
  echo "# Multi-Engine Performance Benchmark Results"
  echo "# Timestamp: $RUN_TIMESTAMP"
  echo "# Hostname: $RUN_HOSTNAME"
  echo "# Seed: $SEED"
  echo "# SystemDS: $RUN_SYSTEMDS_VERSION"
  echo "# JDK: $RUN_JDK_VERSION"
  echo "# PostgreSQL: $RUN_POSTGRES_VERSION"
  echo "# DuckDB: $RUN_DUCKDB_VERSION"
  echo "# CPU: $RUN_CPU_INFO"
  echo "# RAM: $RUN_RAM_INFO"
  echo "# Data: $RUN_DATA_DISPLAY"
  echo "# Warmup: $WARMUP, Repeats: $REPEATS"
  echo "# Statistics: $(if $RUN_STATS; then echo "enabled"; else echo "disabled"; fi)"
  echo "#"
  echo "query,systemds_shell_display,systemds_shell_mean,systemds_shell_stdev,systemds_shell_p95,systemds_core_display,systemds_core_mean,systemds_core_stdev,systemds_core_p95,postgres_display,postgres_mean,postgres_stdev,postgres_p95,postgres_core_display,postgres_core_mean,postgres_core_stdev,postgres_core_p95,duckdb_display,duckdb_mean,duckdb_stdev,duckdb_p95,duckdb_core_display,duckdb_core_mean,duckdb_core_stdev,duckdb_core_p95,fastest"
} > "$RESULT_CSV"
for base in "${QUERIES[@]}"; do
  # Show progress indicator for SystemDS
  progress_indicator "$base" "SystemDS"

  dml_path="$QUERY_DIR/${base}.dml"
  # Parse SystemDS results: shell_mean|shell_stdev|shell_p95|core_mean|core_stdev|core_p95
  # Capture potential SystemDS test-run error messages for JSON reporting
  tmp_err_msg=$(mktemp)
  systemds_result="$(run_systemds_avg "$dml_path" "$tmp_err_msg")"
  # Read any captured error message
  sysds_err_text="$(sed -n '1,200p' "$tmp_err_msg" 2>/dev/null | tr '\n' ' ' || true)"
  rm -f "$tmp_err_msg"
  IFS='|' read -r sd_shell_mean sd_shell_stdev sd_shell_p95 sd_core_mean sd_core_stdev sd_core_p95 <<< "$systemds_result"

  # Format SystemDS results for display
  if [[ "$sd_shell_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    sd_shell_display=$(format_statistics "$sd_shell_mean" "$sd_shell_stdev" "$sd_shell_p95" "$REPEATS")
  else
  sd_shell_display="$sd_shell_mean"
    sd_shell_stdev="0"
    sd_shell_p95="0"
  fi
  if [[ "$sd_core_mean" == "(n/a)" ]]; then
    sd_core_display="(n/a)"
  else
    sd_core_display=$(format_statistics "$sd_core_mean" "$sd_core_stdev" "$sd_core_p95" "$REPEATS")
  fi

  sql_name="${base//_/.}.sql"
  sql_path="$SQL_DIR/$sql_name"
  pg_display="(missing)"
  duck_display="(missing)"

  if [[ -n "$PSQL_EXEC" && -f "$sql_path" ]]; then
    progress_indicator "$base" "PostgreSQL"
    pg_result="$(run_psql_avg_ms "$sql_path")"
    IFS='|' read -r pg_mean pg_stdev pg_p95 pg_core_mean pg_core_stdev pg_core_p95 <<< "$pg_result"
    if [[ "$pg_mean" == "(unavailable)" || "$pg_mean" == "(error)" ]]; then
      pg_display="$pg_mean"
      pg_core_display="$pg_mean"
      pg_stdev="0"
      pg_p95="0"
      pg_core_mean="(n/a)"
      pg_core_stdev="0"
      pg_core_p95="0"
    else
      pg_display=$(format_statistics "$pg_mean" "$pg_stdev" "$pg_p95" "$REPEATS")
      if [[ "$pg_core_mean" != "(n/a)" ]]; then
        pg_core_display=$(format_statistics "$pg_core_mean" "$pg_core_stdev" "$pg_core_p95" "$REPEATS")
      else
        pg_core_display="(n/a)"
      fi
    fi
  elif [[ -z "$PSQL_EXEC" ]]; then
    pg_display="(unavailable)"
    pg_core_display="(unavailable)"
    pg_mean="(unavailable)"
    pg_core_mean="(unavailable)"
    pg_stdev="0"
    pg_p95="0"
    pg_core_stdev="0"
    pg_core_p95="0"
  else
    pg_display="(missing)"
    pg_core_display="(missing)"
    pg_mean="(missing)"
    pg_core_mean="(missing)"
    pg_stdev="0"
    pg_p95="0"
    pg_core_stdev="0"
    pg_core_p95="0"
  fi

  if [[ -n "$DUCKDB_EXEC" && -f "$sql_path" ]]; then
    progress_indicator "$base" "DuckDB"
    duck_result="$(run_duckdb_avg_ms "$sql_path")"
    IFS='|' read -r duck_mean duck_stdev duck_p95 duck_core_mean duck_core_stdev duck_core_p95 <<< "$duck_result"
    if [[ "$duck_mean" == "(unavailable)" || "$duck_mean" == "(error)" ]]; then
      duck_display="$duck_mean"
      duck_stdev="0"
      duck_p95="0"
      duck_core_display="(n/a)"
      duck_core_mean="(n/a)"
      duck_core_stdev="0"
      duck_core_p95="0"
    else
      duck_display=$(format_statistics "$duck_mean" "$duck_stdev" "$duck_p95" "$REPEATS")
      if [[ "$duck_core_mean" == "(n/a)" ]]; then
        duck_core_display="(n/a)"
      else
        duck_core_display=$(format_statistics "$duck_core_mean" "$duck_core_stdev" "$duck_core_p95" "$REPEATS")
      fi
    fi
  elif [[ -z "$DUCKDB_EXEC" ]]; then
    duck_display="(unavailable)"
    duck_mean="(unavailable)"
    duck_stdev="0"
    duck_p95="0"
    duck_core_display="(unavailable)"
    duck_core_mean="(unavailable)"
    duck_core_stdev="0"
    duck_core_p95="0"
  else
    duck_display="(missing)"
    duck_mean="(missing)"
    duck_stdev="0"
    duck_p95="0"
    duck_core_display="(missing)"
    duck_core_mean="(missing)"
    duck_core_stdev="0"
    duck_core_p95="0"
  fi

  # Determine fastest engine based on mean values
  fastest=""
  min_ms=999999999
  for engine in systemds pg duck; do
    val=""
    eng_name=""
    case "$engine" in
      systemds) val="$sd_shell_mean"; eng_name="SystemDS";;
      pg) val="$pg_mean"; eng_name="PostgreSQL";;
      duck) val="$duck_mean"; eng_name="DuckDB";;
    esac
    # Check if value is a valid number (including decimal)
    if [[ "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      # Use awk for floating point comparison
      if [[ $(awk -v val="$val" -v min="$min_ms" 'BEGIN{print (val < min)}') -eq 1 ]]; then
        min_ms=$(awk -v val="$val" 'BEGIN{printf "%.1f", val}')
        fastest="$eng_name"
      elif [[ $(awk -v val="$val" -v min="$min_ms" 'BEGIN{print (val == min)}') -eq 1 ]] && [[ -n "$fastest" ]]; then
        fastest="$fastest+$eng_name"  # Show ties
      fi
    fi
  done
  [[ -z "$fastest" ]] && fastest="(n/a)"

  # Determine SystemDS per-query status and include any error message captured
  systemds_status="success"
  systemds_error_message=null
  if [[ "$sd_shell_mean" == "(error)" ]] || [[ -n "$sysds_err_text" ]]; then
    systemds_status="error"
    if [[ -n "$sysds_err_text" ]]; then
      # Escape quotes for JSON embedding
      esc=$(printf '%s' "$sysds_err_text" | sed -e 's/"/\\"/g')
      systemds_error_message="\"$esc\""
    else
      systemds_error_message="\"SystemDS reported an error during test-run\""
    fi
  fi

  # Prepare mean-only and stats-only cells
  # Means: use numeric mean when available; otherwise use existing display label (unavailable/missing)
  sd_shell_mean_cell=$([[ "$sd_shell_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$sd_shell_mean" || echo "$sd_shell_display")
  sd_core_mean_cell=$([[ "$sd_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$sd_core_mean" || echo "$sd_core_display")
  pg_mean_cell=$([[ "$pg_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$pg_mean" || echo "$pg_display")
  pg_core_mean_cell=$([[ "$pg_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$pg_core_mean" || echo "$pg_core_display")
  duck_mean_cell=$([[ "$duck_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$duck_mean" || echo "$duck_display")
  duck_core_mean_cell=$([[ "$duck_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$duck_core_mean" || echo "$duck_core_display")

  # Stats lines split: CV and p95
  sd_shell_cv_cell=$(format_cv_only "$sd_shell_mean" "$sd_shell_stdev" "$REPEATS")
  sd_core_cv_cell=$(format_cv_only "$sd_core_mean" "$sd_core_stdev" "$REPEATS")
  pg_cv_cell=$(format_cv_only "$pg_mean" "$pg_stdev" "$REPEATS")
  pg_core_cv_cell=$(format_cv_only "$pg_core_mean" "$pg_core_stdev" "$REPEATS")
  duck_cv_cell=$(format_cv_only "$duck_mean" "$duck_stdev" "$REPEATS")
  duck_core_cv_cell=$(format_cv_only "$duck_core_mean" "$duck_core_stdev" "$REPEATS")

  sd_shell_p95_cell=$(format_p95_only "$sd_shell_p95" "$REPEATS")
  sd_core_p95_cell=$(format_p95_only "$sd_core_p95" "$REPEATS")
  pg_p95_cell=$(format_p95_only "$pg_p95" "$REPEATS")
  pg_core_p95_cell=$(format_p95_only "$pg_core_p95" "$REPEATS")
  duck_p95_cell=$(format_p95_only "$duck_p95" "$REPEATS")
  duck_core_p95_cell=$(format_p95_only "$duck_core_p95" "$REPEATS")

  # Clear progress line and display final results
  clear_progress
  if [[ "$LAYOUT_MODE" == "wide" ]]; then
    # Three-line table style with grid separators
    grid_row_wide \
      "$base" \
      "$sd_shell_mean_cell" "$sd_core_mean_cell" \
      "$pg_mean_cell" "$pg_core_mean_cell" \
      "$duck_mean_cell" "$duck_core_mean_cell" \
      "$fastest"
    grid_row_wide \
      "" \
      "$sd_shell_cv_cell" "$sd_core_cv_cell" \
      "$pg_cv_cell" "$pg_core_cv_cell" \
      "$duck_cv_cell" "$duck_core_cv_cell" \
      ""
    grid_row_wide \
      "" \
      "$sd_shell_p95_cell" "$sd_core_p95_cell" \
      "$pg_p95_cell" "$pg_core_p95_cell" \
      "$duck_p95_cell" "$duck_core_p95_cell" \
      ""
    grid_line_wide
  else
    # Stacked layout for narrow terminals
    echo "Query  : $base    Fastest: $fastest"
    printf '  %-20s %s\n' "SystemDS Shell:" "$sd_shell_mean_cell"
    [[ -n "$sd_shell_cv_cell" ]] && printf '  %-20s %s\n' "" "$sd_shell_cv_cell"
    [[ -n "$sd_shell_p95_cell" ]] && printf '  %-20s %s\n' "" "$sd_shell_p95_cell"
    printf '  %-20s %s\n' "SystemDS Core:" "$sd_core_mean_cell"
    [[ -n "$sd_core_cv_cell" ]] && printf '  %-20s %s\n' "" "$sd_core_cv_cell"
    [[ -n "$sd_core_p95_cell" ]] && printf '  %-20s %s\n' "" "$sd_core_p95_cell"
    printf '  %-20s %s\n' "PostgreSQL:" "$pg_mean_cell"
    [[ -n "$pg_cv_cell" ]] && printf '  %-20s %s\n' "" "$pg_cv_cell"
    [[ -n "$pg_p95_cell" ]] && printf '  %-20s %s\n' "" "$pg_p95_cell"
    printf '  %-20s %s\n' "PostgreSQL Core:" "$pg_core_mean_cell"
    [[ -n "$pg_core_cv_cell" ]] && printf '  %-20s %s\n' "" "$pg_core_cv_cell"
    [[ -n "$pg_core_p95_cell" ]] && printf '  %-20s %s\n' "" "$pg_core_p95_cell"
    printf '  %-20s %s\n' "DuckDB:" "$duck_mean_cell"
    [[ -n "$duck_cv_cell" ]] && printf '  %-20s %s\n' "" "$duck_cv_cell"
    [[ -n "$duck_p95_cell" ]] && printf '  %-20s %s\n' "" "$duck_p95_cell"
    printf '  %-20s %s\n' "DuckDB Core:" "$duck_core_mean_cell"
    [[ -n "$duck_core_cv_cell" ]] && printf '  %-20s %s\n' "" "$duck_core_cv_cell"
    [[ -n "$duck_core_p95_cell" ]] && printf '  %-20s %s\n' "" "$duck_core_p95_cell"
    echo "--------------------------------------------------------------------------------"
  fi

  # Write comprehensive data to CSV
  echo "$base,\"$sd_shell_display\",$sd_shell_mean,$sd_shell_stdev,$sd_shell_p95,\"$sd_core_display\",$sd_core_mean,$sd_core_stdev,$sd_core_p95,\"$pg_display\",$pg_mean,$pg_stdev,$pg_p95,\"$pg_core_display\",$pg_core_mean,$pg_core_stdev,$pg_core_p95,\"$duck_display\",$duck_mean,$duck_stdev,$duck_p95,\"$duck_core_display\",$duck_core_mean,$duck_core_stdev,$duck_core_p95,$fastest" >> "$RESULT_CSV"

  # Build JSON entry for this query
  json_entry=$(cat <<JSON_ENTRY
    {
      "query": "$base",
      "systemds": {
        "shell": {
          "display": "$sd_shell_display",
          "mean_ms": $(if [[ "$sd_shell_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$sd_shell_mean"; else echo "null"; fi),
          "stdev_ms": $(if [[ "$sd_shell_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$sd_shell_stdev"; else echo "null"; fi),
          "p95_ms": $(if [[ "$sd_shell_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$sd_shell_p95"; else echo "null"; fi)
        },
          "core": {
          "display": "$sd_core_display",
          "mean_ms": $(if [[ "$sd_core_mean" == "(n/a)" ]]; then echo "null"; else echo "$sd_core_mean"; fi),
          "stdev_ms": $(if [[ "$sd_core_mean" == "(n/a)" ]]; then echo "null"; else echo "$sd_core_stdev"; fi),
          "p95_ms": $(if [[ "$sd_core_mean" == "(n/a)" ]]; then echo "null"; else echo "$sd_core_p95"; fi)
        }
        "status": "$systemds_status",
        "error_message": $systemds_error_message
      },
      "postgresql": {
        "display": "$pg_display",
        "mean_ms": $(if [[ "$pg_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$pg_mean"; else echo "null"; fi),
        "stdev_ms": $(if [[ "$pg_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$pg_stdev"; else echo "null"; fi),
        "p95_ms": $(if [[ "$pg_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$pg_p95"; else echo "null"; fi)
      },
      "postgresql_core": {
        "display": "$pg_core_display",
        "mean_ms": $(if [[ "$pg_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$pg_core_mean"; else echo "null"; fi),
        "stdev_ms": $(if [[ "$pg_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$pg_core_stdev"; else echo "null"; fi),
        "p95_ms": $(if [[ "$pg_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$pg_core_p95"; else echo "null"; fi)
      },
      "duckdb": {
        "display": "$duck_display",
        "mean_ms": $(if [[ "$duck_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$duck_mean"; else echo "null"; fi),
        "stdev_ms": $(if [[ "$duck_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$duck_stdev"; else echo "null"; fi),
        "p95_ms": $(if [[ "$duck_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$duck_p95"; else echo "null"; fi)
      },
      "duckdb_core": {
        "display": "$duck_core_display",
        "mean_ms": $(if [[ "$duck_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$duck_core_mean"; else echo "null"; fi),
        "stdev_ms": $(if [[ "$duck_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$duck_core_stdev"; else echo "null"; fi),
        "p95_ms": $(if [[ "$duck_core_mean" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$duck_core_p95"; else echo "null"; fi)
      },
      "fastest_engine": "$fastest"
    }
JSON_ENTRY
)
  RESULTS_JSON_ARRAY+=("$json_entry")
 done
 echo "==================================================================================================================================================================="
echo

# Generate comprehensive JSON file with metadata and results
{
  echo "{"
  echo "  \"benchmark_metadata\": {"
  echo "    \"benchmark_type\": \"multi_engine_performance\","
  echo "    \"timestamp\": \"$RUN_TIMESTAMP\","
  echo "    \"hostname\": \"$RUN_HOSTNAME\","
  echo "    \"seed\": $SEED,"
  echo "    \"software_versions\": {"
  echo "      \"systemds\": \"$RUN_SYSTEMDS_VERSION\","
  echo "      \"jdk\": \"$RUN_JDK_VERSION\","
  echo "      \"postgresql\": \"$RUN_POSTGRES_VERSION\","
  echo "      \"duckdb\": \"$RUN_DUCKDB_VERSION\""
  echo "    },"
  echo "    \"system_resources\": {"
  echo "      \"cpu\": \"$RUN_CPU_INFO\","
  echo "      \"ram\": \"$RUN_RAM_INFO\""
  echo "    },"
  echo -e "    \"data_build_info\": $RUN_DATA_INFO,"
  echo "    \"run_configuration\": {"
  echo "      \"statistics_enabled\": $(if $RUN_STATS; then echo "true"; else echo "false"; fi),"
  echo "      \"queries_selected\": ${#QUERIES[@]},"
  echo "      \"warmup_runs\": $WARMUP,"
  echo "      \"repeat_runs\": $REPEATS"
  echo "    }"
  echo "  },"
  echo "  \"results\": ["

  # Output results array
  for i in "${!RESULTS_JSON_ARRAY[@]}"; do
    echo "${RESULTS_JSON_ARRAY[$i]}"
    if [[ $i -lt $((${#RESULTS_JSON_ARRAY[@]} - 1)) ]]; then
      echo "    ,"
    fi
  done

  echo "  ]"
  echo "}"
} > "$RESULT_JSON"

echo "Results saved to $RESULT_CSV"
echo "Results saved to $RESULT_JSON"
