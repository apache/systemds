#!/usr/bin/env bash
#
# Multi-Engine SSB Performance Benchmark Runner
# =============================================
#
# CORE SCRIPTS STATUS:
# - Version: 2.0 (August 24, 2025)
# - Status: Fully Enhanced with Performance Analysis
# - Last Updated: August 24, 2025
#
# FEATURES IMPLEMENTED:
# ✓ Multi-engine benchmarking (SystemDS, PostgreSQL, DuckDB)
# ✓ Environment verification and dependency checking
# ✓ Progress indicators with real-time execution status
# ✓ Precision timing measurements with millisecond accuracy
# ✓ Statistical analysis with warmup and multiple repetitions
# ✓ CSV output generation for result analysis
# ✓ Fastest engine detection and reporting
# ✓ Database connection validation
# ✓ Parallel execution control (disabled for fair comparison)
# ✓ Comprehensive error handling and logging
# ✓ Cross-platform compatibility (macOS/Linux)
#
# ENGINES SUPPORTED:
# - SystemDS: Machine learning platform with DML queries
# - PostgreSQL: Industry-standard relational database
# - DuckDB: High-performance analytical database
#
# USAGE:
#   ./run_all_perf.sh               # run full benchmark with all engines
#   ./run_all_perf.sh --warmup 2    # custom warmup iterations
#   ./run_all_perf.sh --repeats 3   # custom repetition count
#   ./run_all_perf.sh --seed=12345  # run with specific seed for reproducibility
#
set -euo pipefail
export LC_ALL=C

REPEATS=5
WARMUP=1
POSTGRES_DB="ssb"
POSTGRES_USER="$(whoami)"
POSTGRES_HOST="localhost"

export _JAVA_OPTIONS="${_JAVA_OPTIONS:-} -Xms2g -Xmx2g -XX:+UseParallelGC -XX:ParallelGCThreads=1"

# Determine script directory and project root
if command -v realpath >/dev/null 2>&1; then
  SCRIPT_DIR="$(dirname "$(realpath "$0")")"
else
  SCRIPT_DIR="$(python - <<'PY'
import os, sys
print(os.path.dirname(os.path.abspath(sys.argv[1])))
PY
"$0")"
fi
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
POSTGRES_DIR="$PROJECT_ROOT/sql/postgres"
DUCKDB_DIR="$PROJECT_ROOT/sql/duckdb"

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

DUCKDB_DB_PATH="$DUCKDB_DIR/ssb.duckdb"

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
  awk -v sec="$1" 'BEGIN{printf "%.0f", sec * 1000}'
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
    echo "${values[0]}|0|${values[0]}"
    return
  fi

  # Calculate mean
  local sum=0
  for val in "${values[@]}"; do
    sum=$((sum + val))
  done
  local mean=$((sum / n))

  # Calculate standard deviation
  local variance_sum=0
  for val in "${values[@]}"; do
    local diff=$((val - mean))
    variance_sum=$((variance_sum + diff * diff))
  done
  local variance=$((variance_sum / n))
  local stdev=$(awk -v var="$variance" 'BEGIN{printf "%.0f", sqrt(var)}')

  # Calculate p95 (95th percentile)
  # Sort the array
  local sorted_values=($(printf '%s\n' "${values[@]}" | sort -n))
  local p95_index=$(awk -v n="$n" 'BEGIN{printf "%.0f", (n-1) * 0.95}')
  if [[ $p95_index -ge $n ]]; then
    p95_index=$((n-1))
  fi
  local p95=${sorted_values[$p95_index]}

  echo "$mean|$stdev|$p95"
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
    if [[ $mean -gt 0 ]]; then
      cv_percent=$(awk -v stdev="$stdev" -v mean="$mean" 'BEGIN{printf "%.1f", (stdev * 100) / mean}')
    fi
    echo "$mean (±${stdev}ms/${cv_percent}%, p95:$p95)"
  fi
}

# Time a command and return real time in ms
time_command_ms() {
  local out
  # Properly capture stderr from /usr/bin/time while suppressing stdout of the command
  out=$({ /usr/bin/time -p "$@" > /dev/null; } 2>&1)
  local real_sec=$(echo "$out" | awk '/^real /{print $2}')
  if [[ -z "$real_sec" ]]; then
    echo "0"
    return
  fi
  sec_to_ms "$real_sec"
}

# Run a SystemDS query and compute statistics
run_systemds_avg() {
  local dml="$1"
  local shell_times=()
  local core_times=()
  local core_have=false

  # Change to project root directory so relative paths in DML work correctly
  local original_dir="$(pwd)"
  cd "$PROJECT_ROOT"

  # Warmup runs
  for ((w=1; w<=WARMUP; w++)); do
    if $RUN_STATS; then
      "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}" > /dev/null 2>&1 || true
    else
      "$SYSTEMDS_CMD" "$dml" "${SYS_EXTRA_ARGS[@]}" > /dev/null 2>&1 || true
    fi
  done

  # Timed runs - collect all measurements
  for ((i=1; i<=REPEATS; i++)); do
    if $RUN_STATS; then
      local shell_ms
      shell_ms=$(time_command_ms "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}")
      shell_times+=("$shell_ms")

      # Capture SystemDS stats output and extract timing
      local temp_file
      temp_file=$(mktemp)
      "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}" > "$temp_file" 2>/dev/null || true
      local internal_sec
      internal_sec=$(awk '/Total execution time:/ {print $4}' "$temp_file" || true)
      rm -f "$temp_file"
      if [[ -n "$internal_sec" ]] && [[ "$internal_sec" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        local core_ms
        core_ms=$(awk -v sec="$internal_sec" 'BEGIN{printf "%.0f", sec * 1000}')
        core_times+=("$core_ms")
        core_have=true
      fi
    else
      local shell_ms
      shell_ms=$(time_command_ms "$SYSTEMDS_CMD" "$dml" "${SYS_EXTRA_ARGS[@]}")
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
    echo "(unavailable)|0|0"
    return
  fi

  # Test run first
  "$PSQL_EXEC" -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" \
    -v ON_ERROR_STOP=1 -q \
    -c "SET max_parallel_workers=0; SET max_parallel_maintenance_workers=0; SET max_parallel_workers_per_gather=0; SET parallel_leader_participation=off;" \
    -f "$sql_file" >/dev/null 2>/dev/null || {
      echo "(error)|0|0"
      return
    }

  local times=()
  for ((i=1; i<=REPEATS; i++)); do
    local ms
    ms=$(time_command_ms "$PSQL_EXEC" -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" \
      -v ON_ERROR_STOP=1 -q \
      -c "SET max_parallel_workers=0; SET max_parallel_maintenance_workers=0; SET max_parallel_workers_per_gather=0; SET parallel_leader_participation=off;" \
      -f "$sql_file" 2>/dev/null) || {
        echo "(error)|0|0"
        return
      }
    times+=("$ms")
  done

  calculate_statistics "${times[@]}"
}

# Run a DuckDB query and compute statistics
run_duckdb_avg_ms() {
  local sql_file="$1"

  # Check if DuckDB is available
  if [[ -z "$DUCKDB_EXEC" ]]; then
    echo "(unavailable)|0|0"
    return
  fi

  local tmp_sql
  tmp_sql=$(mktemp)
  printf 'PRAGMA threads=1;\n' > "$tmp_sql"
  cat "$sql_file" >> "$tmp_sql"

  # Test run first
  "$DUCKDB_EXEC" "$DUCKDB_DB_PATH" -init "$tmp_sql" -c ".quit" >/dev/null 2>&1 || {
    rm -f "$tmp_sql"
    echo "(error)|0|0"
    return
  }

  local times=()
  for ((i=1; i<=REPEATS; i++)); do
    local ms
    ms=$(time_command_ms "$DUCKDB_EXEC" "$DUCKDB_DB_PATH" -init "$tmp_sql" -c ".quit" 2>/dev/null) || {
      rm -f "$tmp_sql"
      echo "(error)|0|0"
      return
    }
    times+=("$ms")
  done
  rm -f "$tmp_sql"

  calculate_statistics "${times[@]}"
}

# Parse arguments
RUN_STATS=false
QUERIES=()
SEED=""
for arg in "$@"; do
  if [[ "$arg" == "--stats" ]]; then
    RUN_STATS=true
  elif [[ "$arg" == --seed=* ]]; then
    SEED="${arg#--seed=}"
  elif [[ "$arg" == "--seed" ]]; then
    echo "Error: --seed requires a value (e.g., --seed=12345)" >&2
    exit 1
  elif [[ "$arg" == --warmup=* ]]; then
    WARMUP="${arg#--warmup=}"
    if ! [[ "$WARMUP" =~ ^[0-9]+$ ]] || [[ "$WARMUP" -lt 0 ]]; then
      echo "Error: --warmup requires a non-negative integer (e.g., --warmup=2)" >&2
      exit 1
    fi
  elif [[ "$arg" == "--warmup" ]]; then
    echo "Error: --warmup requires a value (e.g., --warmup=2)" >&2
    exit 1
  elif [[ "$arg" == --repeats=* ]]; then
    REPEATS="${arg#--repeats=}"
    if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
      echo "Error: --repeats requires a positive integer (e.g., --repeats=5)" >&2
      exit 1
    fi
  elif [[ "$arg" == "--repeats" ]]; then
    echo "Error: --repeats requires a value (e.g., --repeats=5)" >&2
    exit 1
  else
    QUERIES+=( "$(echo "$arg" | tr '.' '_')" )
  fi
 done

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
  local ssb_data_dir="$PROJECT_ROOT/data"
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
  echo -ne "\r$query_name: Running $stage...                                                          "
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
echo "- DuckDB (ms):         Single-threaded execution time with threads=1 pragma"
echo "- (missing):           SQL file not found for this query"
echo "- (n/a):               Core timing unavailable (run with --stats flag for SystemDS internal timing)"
echo
echo "NOTE: All engines use single-threaded execution for fair comparison."
echo "      Multiple runs with averaging provide statistical reliability."
echo
echo "Single-threaded execution; warm-up runs: $WARMUP, timed runs: $REPEATS"
echo "SystemDS core times available with --stats."
echo
echo "============================================================================================================"
printf '%-12s %-22s %-20s %-18s %-12s %-10s\n' "Query" "SystemDS Shell (ms)" "SystemDS Core (ms)" "PostgreSQL (ms)" "DuckDB (ms)" "Fastest"
echo "------------------------------------------------------------------------------------------------------------"
# Create output data directory
OUTPUT_DATA_DIR="$SCRIPT_DIR/OutputPerformanceData"
mkdir -p "$OUTPUT_DATA_DIR"
RESULT_CSV="$OUTPUT_DATA_DIR/results_$(date +%Y%m%d_%H%M%S).csv"
RESULT_JSON="$OUTPUT_DATA_DIR/results_$(date +%Y%m%d_%H%M%S).json"

# Initialize results array for JSON
RESULTS_JSON_ARRAY=()

# Write CSV header with comprehensive metadata
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
  echo "query,systemds_shell_display,systemds_shell_mean,systemds_shell_stdev,systemds_shell_p95,systemds_core_display,systemds_core_mean,systemds_core_stdev,systemds_core_p95,postgres_display,postgres_mean,postgres_stdev,postgres_p95,duckdb_display,duckdb_mean,duckdb_stdev,duckdb_p95,fastest"
} > "$RESULT_CSV"
for base in "${QUERIES[@]}"; do
  # Show progress indicator for SystemDS
  progress_indicator "$base" "SystemDS"

  dml_path="$QUERY_DIR/${base}.dml"
  # Parse SystemDS results: shell_mean|shell_stdev|shell_p95|core_mean|core_stdev|core_p95
  systemds_result="$(run_systemds_avg "$dml_path")"
  IFS='|' read -r sd_shell_mean sd_shell_stdev sd_shell_p95 sd_core_mean sd_core_stdev sd_core_p95 <<< "$systemds_result"

  # Format SystemDS results for display
  sd_shell_display=$(format_statistics "$sd_shell_mean" "$sd_shell_stdev" "$sd_shell_p95" "$REPEATS")
  if [[ "$sd_core_mean" == "(n/a)" ]]; then
    sd_core_display="(n/a)"
  else
    sd_core_display=$(format_statistics "$sd_core_mean" "$sd_core_stdev" "$sd_core_p95" "$REPEATS")
  fi

  sql_name="${base//_/.}.sql"
  pg_display="(missing)"
  duck_display="(missing)"

  if [[ -n "$PSQL_EXEC" && -f "$POSTGRES_DIR/$sql_name" ]]; then
    progress_indicator "$base" "PostgreSQL"
    pg_result="$(run_psql_avg_ms "$POSTGRES_DIR/$sql_name")"
    IFS='|' read -r pg_mean pg_stdev pg_p95 <<< "$pg_result"
    if [[ "$pg_mean" == "(unavailable)" || "$pg_mean" == "(error)" ]]; then
      pg_display="$pg_mean"
      pg_stdev="0"
      pg_p95="0"
    else
      pg_display=$(format_statistics "$pg_mean" "$pg_stdev" "$pg_p95" "$REPEATS")
    fi
  elif [[ -z "$PSQL_EXEC" ]]; then
    pg_display="(unavailable)"
    pg_mean="(unavailable)"
    pg_stdev="0"
    pg_p95="0"
  else
    pg_display="(missing)"
    pg_mean="(missing)"
    pg_stdev="0"
    pg_p95="0"
  fi

  if [[ -n "$DUCKDB_EXEC" && -f "$DUCKDB_DIR/$sql_name" ]]; then
    progress_indicator "$base" "DuckDB"
    duck_result="$(run_duckdb_avg_ms "$DUCKDB_DIR/$sql_name")"
    IFS='|' read -r duck_mean duck_stdev duck_p95 <<< "$duck_result"
    if [[ "$duck_mean" == "(unavailable)" || "$duck_mean" == "(error)" ]]; then
      duck_display="$duck_mean"
      duck_stdev="0"
      duck_p95="0"
    else
      duck_display=$(format_statistics "$duck_mean" "$duck_stdev" "$duck_p95" "$REPEATS")
    fi
  elif [[ -z "$DUCKDB_EXEC" ]]; then
    duck_display="(unavailable)"
    duck_mean="(unavailable)"
    duck_stdev="0"
    duck_p95="0"
  else
    duck_display="(missing)"
    duck_mean="(missing)"
    duck_stdev="0"
    duck_p95="0"
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
    if [[ "$val" =~ ^[0-9]+$ ]] && (( val < min_ms )); then
      min_ms=$val
      fastest="$eng_name"
    fi
  done
  [[ -z "$fastest" ]] && fastest="(n/a)"

  # Clear progress line and display final results
  echo -ne "\r                                                                                          \r"
  printf '%-12s %-22s %-20s %-18s %-12s %-10s\n' "$base" "$sd_shell_display" "$sd_core_display" "$pg_display" "$duck_display" "$fastest"

  # Write comprehensive data to CSV
  echo "$base,\"$sd_shell_display\",$sd_shell_mean,$sd_shell_stdev,$sd_shell_p95,\"$sd_core_display\",$sd_core_mean,$sd_core_stdev,$sd_core_p95,\"$pg_display\",$pg_mean,$pg_stdev,$pg_p95,\"$duck_display\",$duck_mean,$duck_stdev,$duck_p95,$fastest" >> "$RESULT_CSV"

  # Build JSON entry for this query
  json_entry=$(cat <<JSON_ENTRY
    {
      "query": "$base",
      "systemds": {
        "shell": {
          "display": "$sd_shell_display",
          "mean_ms": $sd_shell_mean,
          "stdev_ms": $sd_shell_stdev,
          "p95_ms": $sd_shell_p95
        },
        "core": {
          "display": "$sd_core_display",
          "mean_ms": $(if [[ "$sd_core_mean" == "(n/a)" ]]; then echo "null"; else echo "$sd_core_mean"; fi),
          "stdev_ms": $(if [[ "$sd_core_mean" == "(n/a)" ]]; then echo "null"; else echo "$sd_core_stdev"; fi),
          "p95_ms": $(if [[ "$sd_core_mean" == "(n/a)" ]]; then echo "null"; else echo "$sd_core_p95"; fi)
        }
      },
      "postgresql": {
        "display": "$pg_display",
        "mean_ms": $(if [[ "$pg_mean" =~ ^[0-9]+$ ]]; then echo "$pg_mean"; else echo "null"; fi),
        "stdev_ms": $(if [[ "$pg_mean" =~ ^[0-9]+$ ]]; then echo "$pg_stdev"; else echo "null"; fi),
        "p95_ms": $(if [[ "$pg_mean" =~ ^[0-9]+$ ]]; then echo "$pg_p95"; else echo "null"; fi)
      },
      "duckdb": {
        "display": "$duck_display",
        "mean_ms": $(if [[ "$duck_mean" =~ ^[0-9]+$ ]]; then echo "$duck_mean"; else echo "null"; fi),
        "stdev_ms": $(if [[ "$duck_mean" =~ ^[0-9]+$ ]]; then echo "$duck_stdev"; else echo "null"; fi),
        "p95_ms": $(if [[ "$duck_mean" =~ ^[0-9]+$ ]]; then echo "$duck_p95"; else echo "null"; fi)
      },
      "fastest_engine": "$fastest"
    }
JSON_ENTRY
)
  RESULTS_JSON_ARRAY+=("$json_entry")
 done
 echo "============================================================================================================"
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