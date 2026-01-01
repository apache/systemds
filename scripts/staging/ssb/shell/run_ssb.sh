#!/usr/bin/env bash
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

# SystemDS Star Schema Benchmark (SSB) Runner
# ===========================================
#
# CORE SCRIPTS STATUS:
# - Version: 1.0 (September 5, 2025)
# - Status: Production-Ready with Advanced User Experience
# - First Public Release: September 5, 2025
#
# FEATURES IMPLEMENTED:
# ✓ Basic SSB query execution with SystemDS 3.4.0-SNAPSHOT
# ✓ Single-threaded configuration for consistent benchmarking
# ✓ Progress indicators with real-time updates
# ✓ Comprehensive timing measurements using /usr/bin/time
# ✓ Query result extraction (scalar and table formats)
# ✓ Success/failure tracking with detailed reporting
# ✓ Query summary table with execution status
# ✓ "See below" notation with result reprinting (NEW)
# ✓ Long table outputs displayed after summary (NEW)
# ✓ Error handling with timeout protection
# ✓ Cross-platform compatibility (macOS/Linux)
#
# RECENT IMPORTANT ADDITIONS:
# - Accepts --input-dir=PATH and forwards it into DML runs as a SystemDS named
#   argument: -nvargs input_dir=/path/to/data (DML can use sys.vinput_dir or
#   the named argument to locate data files instead of hardcoded `data/`).
# - Fast-fail on missing input directory: the runner verifies the provided
#   input path exists and exits with a clear error message if not.
# - Runtime SystemDS error detection: test-run output is scanned for runtime
#   error blocks (e.g., "An Error Occurred : ..."). Queries with runtime
#   failures are reported as `status: "error"` and include `error_message`
#   in generated JSON metadata for easier debugging and CI integration.
#
# MAJOR FEATURES IN v1.0 (First Public Release):
# - Complete SSB query execution with SystemDS 3.4.0-SNAPSHOT
# - Enhanced "see below" notation with result reprinting
# - Long table outputs displayed after summary for better UX
# - Eliminated need to scroll back through terminal output
# - Maintained array alignment for consistent result tracking
# - JSON metadata contains complete query results, not "see below"
# - Added --out-dir option for custom output directory
# - Multi-format output: TXT, CSV, JSON for each query result
# - Structured output directory with comprehensive run.json metadata file
#
# DEPENDENCIES:
# - SystemDS binary (3.4.0-SNAPSHOT or later)
# - Single-threaded configuration file (auto-generated)
# - SSB query files in scripts/ssb/queries/
# - Bash 4.0+ with timeout support
#
# USAGE (from repo root):
#   scripts/ssb/shell/run_ssb.sh                    # run all SSB queries
#   scripts/ssb/shell/run_ssb.sh q1.1 q2.3          # run specific queries
#   scripts/ssb/shell/run_ssb.sh --stats            # enable internal statistics
#   scripts/ssb/shell/run_ssb.sh q3.1 --stats       # run specific query with stats
#   scripts/ssb/shell/run_ssb.sh --seed=12345       # run with specific seed for reproducibility
#   scripts/ssb/shell/run_ssb.sh --out-dir=/path    # specify output directory for results
#
set -euo pipefail
export LC_ALL=C

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
if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
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

# Locate SystemDS executable
SYSTEMDS_CMD="$PROJECT_ROOT/bin/systemds"
if [[ ! -x "$SYSTEMDS_CMD" ]]; then
  SYSTEMDS_CMD="$(command -v systemds || true)"
fi
if [[ -z "$SYSTEMDS_CMD" || ! -x "$SYSTEMDS_CMD" ]]; then
  echo "Error: could not find SystemDS executable." >&2
  echo "       Tried: $PROJECT_ROOT/bin/systemds and PATH" >&2
  exit 1
fi

# Ensure single-threaded configuration file exists
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

# Query directory
QUERY_DIR="$PROJECT_ROOT/scripts/ssb/queries"

# Verify query directory exists
if [[ ! -d "$QUERY_DIR" ]]; then
  echo "Error: Query directory not found: $QUERY_DIR" >&2
  exit 1
fi

# Help function
show_help() {
  cat << 'EOF'
SystemDS Star Schema Benchmark (SSB) Runner v1.0

USAGE (from repo root):
  scripts/ssb/shell/run_ssb.sh [OPTIONS] [QUERIES...]

OPTIONS:
  --stats, -stats         Enable SystemDS internal statistics collection
  --seed=N, -seed=N      Set random seed for reproducible results (default: auto-generated)
  --output-dir=PATH, -output-dir=PATH  Specify custom output directory (default: $PROJECT_ROOT/scripts/ssb/shell/ssbOutputData/QueryData)
  --input-dir=PATH, -input-dir=PATH  Specify custom data directory (default: $PROJECT_ROOT/data)
  --help, -help, -h, --h Show this help message
  --version, -version, -v, --v  Show version information

QUERIES:
  If no queries are specified, all available SSB queries (q*.dml) will be executed.
  To run specific queries, provide their names (with or without .dml extension):
    ./run_ssb.sh q1.1 q2.3 q4.1

EXAMPLES (from repo root):
  scripts/ssb/shell/run_ssb.sh                          # Run all SSB queries
  scripts/ssb/shell/run_ssb.sh --stats                  # Run all queries with statistics
  scripts/ssb/shell/run_ssb.sh -stats                   # Same as above (single dash)
  scripts/ssb/shell/run_ssb.sh q1.1 q2.3                # Run specific queries only
  scripts/ssb/shell/run_ssb.sh --seed=12345 --stats     # Reproducible run with statistics
  scripts/ssb/shell/run_ssb.sh -seed=12345 -stats       # Same as above (single dash)
  scripts/ssb/shell/run_ssb.sh --output-dir=/tmp/results   # Custom output directory
  scripts/ssb/shell/run_ssb.sh -output-dir=/tmp/results    # Same as above (single dash)
  scripts/ssb/shell/run_ssb.sh --input-dir=/path/to/data  # Custom data directory
  scripts/ssb/shell/run_ssb.sh -input-dir=/path/to/data   # Same as above (single dash)

OUTPUT:
  Results are saved in multiple formats:
  - TXT: Human-readable format
  - CSV: Machine-readable data format
  - JSON: Structured format with metadata
  - run.json: Complete run metadata and results

For more information, see the documentation in scripts/ssb/README.md
EOF
}

# Parse arguments
RUN_STATS=false
QUERIES=()
SEED=""
OUT_DIR=""
INPUT_DIR=""
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-help" || "$arg" == "-h" || "$arg" == "--h" ]]; then
    show_help
    exit 0
  elif [[ "$arg" == "--version" || "$arg" == "-version" || "$arg" == "-v" || "$arg" == "--v" ]]; then
    echo "SystemDS Star Schema Benchmark (SSB) Runner v1.0"
    echo "First Public Release: September 5, 2025"
    exit 0
  elif [[ "$arg" == "--stats" || "$arg" == "-stats" ]]; then
    RUN_STATS=true
  elif [[ "$arg" == --seed=* || "$arg" == -seed=* ]]; then
    if [[ "$arg" == --seed=* ]]; then
      SEED="${arg#--seed=}"
    else
      SEED="${arg#-seed=}"
    fi
  elif [[ "$arg" == "--seed" || "$arg" == "-seed" ]]; then
    echo "Error: --seed/-seed requires a value (e.g., --seed=12345 or -seed=12345)" >&2
    exit 1
  elif [[ "$arg" == --output-dir=* || "$arg" == -output-dir=* ]]; then
    if [[ "$arg" == --output-dir=* ]]; then
      OUT_DIR="${arg#--output-dir=}"
    else
      OUT_DIR="${arg#-output-dir=}"
    fi
  elif [[ "$arg" == "--output-dir" || "$arg" == "-output-dir" ]]; then
    echo "Error: --output-dir/-output-dir requires a value (e.g., --output-dir=/path/to/output or -output-dir=/path/to/output)" >&2
    exit 1
  elif [[ "$arg" == --input-dir=* || "$arg" == -input-dir=* ]]; then
    if [[ "$arg" == --input-dir=* ]]; then
      INPUT_DIR="${arg#--input-dir=}"
    else
      INPUT_DIR="${arg#-input-dir=}"
    fi
  elif [[ "$arg" == "--input-dir" || "$arg" == "-input-dir" ]]; then
    echo "Error: --input-dir/-input-dir requires a value (e.g., --input-dir=/path/to/data or -input-dir=/path/to/data)" >&2
    exit 1
  else
    # Check if argument looks like an unrecognized option (starts with dash)
    if [[ "$arg" == -* ]]; then
      echo "Error: Unrecognized option '$arg'" >&2
      echo "Use --help or -h to see available options." >&2
      exit 1
    else
      # Treat as query name
      name="$(echo "$arg" | tr '.' '_')"
      QUERIES+=( "$name.dml" )
    fi
  fi
done

# Set default output directory if not provided
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$PROJECT_ROOT/scripts/ssb/shell/ssbOutputData/QueryData"
fi

# Set default input data directory if not provided
if [[ -z "$INPUT_DIR" ]]; then
  INPUT_DIR="$PROJECT_ROOT/data"
fi

# Normalize paths by removing trailing slashes
INPUT_DIR="${INPUT_DIR%/}"
OUT_DIR="${OUT_DIR%/}"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Pass input directory to DML scripts via SystemDS named arguments
NVARGS=( -nvargs "input_dir=${INPUT_DIR}" )

# Validate input data directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: Input data directory '$INPUT_DIR' does not exist." >&2
  echo "Please create the directory or specify a valid path with --input-dir=PATH" >&2
  exit 1
fi

# Generate seed if not provided
if [[ -z "$SEED" ]]; then
  SEED=$((RANDOM * 32768 + RANDOM))
fi

# Discover queries if none provided
shopt -s nullglob
if [[ ${#QUERIES[@]} -eq 0 ]]; then
  for f in "$QUERY_DIR"/q*.dml; do
    if [[ -f "$f" ]]; then
      QUERIES+=("$(basename "$f")")
    fi
  done
  if [[ ${#QUERIES[@]} -eq 0 ]]; then
    echo "Error: No query files found in $QUERY_DIR" >&2
    exit 1
  fi
fi
shopt -u nullglob

# Metadata collection functions
collect_system_metadata() {
  local timestamp hostname systemds_version jdk_version cpu_info ram_info

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
    jdk_version=$(java -version 2>&1 | head -1 | sed 's/.*"\(.*\)".*/\1/' || echo "unknown")
  else
    jdk_version="unknown"
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

# Output format functions
create_output_structure() {
  local run_id="$1"
  local base_dir="$OUT_DIR/ssb_run_$run_id"

  # Create output directory structure
  mkdir -p "$base_dir"/{txt,csv,json}

  # Set global variables for output paths
  OUTPUT_BASE_DIR="$base_dir"
  OUTPUT_TXT_DIR="$base_dir/txt"
  OUTPUT_CSV_DIR="$base_dir/csv"
  OUTPUT_JSON_DIR="$base_dir/json"
  OUTPUT_METADATA_FILE="$base_dir/run.json"
}

save_query_result_txt() {
  local query_name="$1"
  local result_data="$2"
  local output_file="$OUTPUT_TXT_DIR/${query_name}.txt"

  {
    echo "========================================="
    echo "SSB Query: $query_name"
    echo "========================================="
    echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Seed: $SEED"
    echo ""
    echo "Result:"
    echo "---------"
    echo "$result_data"
    echo ""
    echo "========================================="
  } > "$output_file"
}

save_query_result_csv() {
  local query_name="$1"
  local result_data="$2"
  local output_file="$OUTPUT_CSV_DIR/${query_name}.csv"

  # Check if result is a single scalar value (including negative numbers and scientific notation)
  if [[ "$result_data" =~ ^-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
    # Scalar result
    {
      echo "query,result"
      echo "$query_name,$result_data"
    } > "$output_file"
  else
    # Table result - try to convert to CSV format
    {
      echo "# SSB Query: $query_name"
      echo "# Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
      echo "# Seed: $SEED"
      # Convert space-separated table data to CSV
      echo "$result_data" | sed 's/  */,/g' | sed 's/^,//g' | sed 's/,$//g'
    } > "$output_file"
  fi
}

save_query_result_json() {
  local query_name="$1"
  local result_data="$2"
  local output_file="$OUTPUT_JSON_DIR/${query_name}.json"

  # Escape quotes and special characters for JSON
  local escaped_result=$(echo "$result_data" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')

  {
    echo "{"
    echo "  \"query\": \"$query_name\","
    echo "  \"timestamp\": \"$(date -u '+%Y-%m-%d %H:%M:%S UTC')\","
    echo "  \"seed\": $SEED,"
    echo "  \"result\": \"$escaped_result\","
    echo "  \"metadata\": {"
    echo "    \"systemds_version\": \"$RUN_SYSTEMDS_VERSION\","
    echo "    \"hostname\": \"$RUN_HOSTNAME\""
    echo "  }"
    echo "}"
  } > "$output_file"
}

save_all_formats() {
  local query_name="$1"
  local result_data="$2"

  save_query_result_txt "$query_name" "$result_data"
  save_query_result_csv "$query_name" "$result_data"
  save_query_result_json "$query_name" "$result_data"
}

# Collect metadata
collect_system_metadata
collect_data_metadata

# Create output directory structure with timestamp-based run ID
RUN_ID="$(date +%Y%m%d_%H%M%S)"
create_output_structure "$RUN_ID"

# Execute queries
count=0
failed=0
SUCCESSFUL_QUERIES=()  # Array to track successfully executed queries
ALL_RUN_QUERIES=()     # Array to track all queries that were attempted (in order)
QUERY_STATUS=()        # Array to track status: "success" or "error"
QUERY_ERROR_MSG=()     # Array to store error messages for failed queries
QUERY_RESULTS=()       # Array to track query results for display
QUERY_FULL_RESULTS=()  # Array to track complete query results for JSON
QUERY_STATS=()         # Array to track SystemDS statistics for JSON
QUERY_TIMINGS=()       # Array to track execution timing statistics
LONG_OUTPUTS=()        # Array to store long table outputs for display after summary

# Progress indicator function
progress_indicator() {
  local query_name="$1"
  local current="$2"
  local total="$3"
  echo -ne "\r[$current/$total] Running: $query_name                                                   "
}

for q in "${QUERIES[@]}"; do
  dml="$QUERY_DIR/$q"
  if [[ ! -f "$dml" ]]; then
    echo "Warning: query file '$dml' not found; skipping." >&2
    continue
  fi

  # Show progress
  progress_indicator "$q" "$((count + failed + 1))" "${#QUERIES[@]}"

  # Change to project root directory so relative paths in DML work correctly
  cd "$PROJECT_ROOT"

  # Clear progress line before showing output
  echo -ne "\r                                                                                   \r"
  echo "[$((count + failed + 1))/${#QUERIES[@]}] Running: $q"

  # Record attempted query
  ALL_RUN_QUERIES+=("$q")

  if $RUN_STATS; then
    # Capture output to extract result
    temp_output=$(mktemp)
  if "$SYSTEMDS_CMD" "$dml" -stats "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}" | tee "$temp_output"; then
      # Even when SystemDS exits 0, the DML can emit runtime errors. Detect common error markers.
      error_msg=$(sed -n '/An Error Occurred :/,$ p' "$temp_output" | sed -n '1,200p' | tr '\n' ' ' | sed 's/^ *//;s/ *$//')
      if [[ -n "$error_msg" ]]; then
        echo "Error: Query $q reported runtime error" >&2
        echo "$error_msg" >&2
        failed=$((failed+1))
        QUERY_STATUS+=("error")
        QUERY_ERROR_MSG+=("$error_msg")
        # Maintain array alignment
        QUERY_STATS+=("")
        QUERY_RESULTS+=("N/A")
        QUERY_FULL_RESULTS+=("N/A")
        LONG_OUTPUTS+=("")
      else
        count=$((count+1))
        SUCCESSFUL_QUERIES+=("$q")  # Track successful query
        QUERY_STATUS+=("success")
      # Extract result - try multiple patterns with timeouts to prevent hanging:
      # 1. Simple scalar pattern like "REVENUE: 687752409"
      result=$(timeout 5s grep -E "^[A-Z_]+:\s*[0-9]+" "$temp_output" | tail -1 | awk '{print $2}' 2>/dev/null || true)
      full_result="$result"  # For scalar results, display and full results are the same

      # 2. If no scalar pattern, check for table output and get row count
      if [[ -z "$result" ]]; then
        # Look for frame info like "# FRAME: nrow = 53, ncol = 3"
        nrows=$(timeout 5s grep "# FRAME: nrow =" "$temp_output" | awk '{print $5}' | tr -d ',' 2>/dev/null || true)
        if [[ -n "$nrows" ]]; then
          result="${nrows} rows (see below)"
          # Extract and store the long output for later display (excluding statistics)
          long_output=$(grep -v "^#" "$temp_output" | grep -v "WARNING" | grep -v "WARN" | grep -v "^$" | sed '/^SystemDS Statistics:/,$ d')
          LONG_OUTPUTS+=("$long_output")
          # For JSON, store the actual table data
          full_result="$long_output"
        else
          # Count actual data rows (lines with numbers, excluding headers and comments) - limit to prevent hanging
          nrows=$(timeout 5s grep -E "^[0-9]" "$temp_output" | sed '/^SystemDS Statistics:/,$ d' | head -1000 | wc -l | tr -d ' ' 2>/dev/null || echo "0")
          if [[ "$nrows" -gt 0 ]]; then
            result="${nrows} rows (see below)"
            # Extract and store the long output for later display (excluding statistics)
            long_output=$(grep -E "^[0-9]" "$temp_output" | sed '/^SystemDS Statistics:/,$ d' | head -1000)
            LONG_OUTPUTS+=("$long_output")
            # For JSON, store the actual table data
            full_result="$long_output"
          else
            result="N/A"
            full_result="N/A"
            LONG_OUTPUTS+=("")  # Empty placeholder to maintain array alignment
          fi
        fi
      else
        LONG_OUTPUTS+=("")  # Empty placeholder for scalar results to maintain array alignment
      fi
      QUERY_RESULTS+=("$result")  # Track query result for display
      QUERY_FULL_RESULTS+=("$full_result")  # Track complete query result for JSON

      # Save result in all formats
      query_name_clean="${q%.dml}"

      # Extract and store statistics for JSON (preserving newlines)
      stats_output=$(sed -n '/^SystemDS Statistics:/,$ p' "$temp_output")
  QUERY_STATS+=("$stats_output")  # Track statistics for JSON

      save_all_formats "$query_name_clean" "$full_result"
      fi
    else
      echo "Error: Query $q failed" >&2
      failed=$((failed+1))
      QUERY_STATUS+=("error")
      QUERY_ERROR_MSG+=("Query execution failed (non-zero exit)")
      # Add empty stats entry for failed queries to maintain array alignment
      QUERY_STATS+=("")
    fi
    rm -f "$temp_output"
  else
    # Capture output to extract result
    temp_output=$(mktemp)
  if "$SYSTEMDS_CMD" "$dml" "${SYS_EXTRA_ARGS[@]}" "${NVARGS[@]}" | tee "$temp_output"; then
      # Detect runtime errors in output even if command returned 0
      error_msg=$(sed -n '/An Error Occurred :/,$ p' "$temp_output" | sed -n '1,200p' | tr '\n' ' ' | sed 's/^ *//;s/ *$//')
      if [[ -n "$error_msg" ]]; then
        echo "Error: Query $q reported runtime error" >&2
        echo "$error_msg" >&2
        failed=$((failed+1))
        QUERY_STATUS+=("error")
        QUERY_ERROR_MSG+=("$error_msg")
        QUERY_STATS+=("")
        QUERY_RESULTS+=("N/A")
        QUERY_FULL_RESULTS+=("N/A")
        LONG_OUTPUTS+=("")
      else
        count=$((count+1))
        SUCCESSFUL_QUERIES+=("$q")  # Track successful query
        QUERY_STATUS+=("success")
      # Extract result - try multiple patterns with timeouts to prevent hanging:
      # 1. Simple scalar pattern like "REVENUE: 687752409"
      result=$(timeout 5s grep -E "^[A-Z_]+:\s*[0-9]+" "$temp_output" | tail -1 | awk '{print $2}' 2>/dev/null || true)
      full_result="$result"  # For scalar results, display and full results are the same

      # 2. If no scalar pattern, check for table output and get row count
      if [[ -z "$result" ]]; then
        # Look for frame info like "# FRAME: nrow = 53, ncol = 3"
        nrows=$(timeout 5s grep "# FRAME: nrow =" "$temp_output" | awk '{print $5}' | tr -d ',' 2>/dev/null || true)
        if [[ -n "$nrows" ]]; then
          result="${nrows} rows (see below)"
          # Extract and store the long output for later display
          long_output=$(grep -v "^#" "$temp_output" | grep -v "WARNING" | grep -v "WARN" | grep -v "^$" | tail -n +1)
          LONG_OUTPUTS+=("$long_output")
          # For JSON, store the actual table data
          full_result="$long_output"
        else
          # Count actual data rows (lines with numbers, excluding headers and comments) - limit to prevent hanging
          nrows=$(timeout 5s grep -E "^[0-9]" "$temp_output" | head -1000 | wc -l | tr -d ' ' 2>/dev/null || echo "0")
          if [[ "$nrows" -gt 0 ]]; then
            result="${nrows} rows (see below)"
            # Extract and store the long output for later display
            long_output=$(grep -E "^[0-9]" "$temp_output" | head -1000)
            LONG_OUTPUTS+=("$long_output")
            # For JSON, store the actual table data
            full_result="$long_output"
          else
            result="N/A"
            full_result="N/A"
            LONG_OUTPUTS+=("")  # Empty placeholder to maintain array alignment
          fi
        fi
      else
        LONG_OUTPUTS+=("")  # Empty placeholder for scalar results to maintain array alignment
      fi
      QUERY_RESULTS+=("$result")  # Track query result for display
      QUERY_FULL_RESULTS+=("$full_result")  # Track complete query result for JSON

  # Add empty stats entry for non-stats runs to maintain array alignment
  QUERY_STATS+=("")

      # Save result in all formats
      query_name_clean="${q%.dml}"
      save_all_formats "$query_name_clean" "$full_result"
      fi
    else
      echo "Error: Query $q failed" >&2
      failed=$((failed+1))
      QUERY_STATUS+=("error")
      QUERY_ERROR_MSG+=("Query execution failed (non-zero exit)")
      # Add empty stats entry for failed queries to maintain array alignment
      QUERY_STATS+=("")
    fi
    rm -f "$temp_output"
  fi
done

# Summary
echo ""
echo "========================================="
echo "SSB benchmark completed!"
echo "Total queries executed: $count"
if [[ $failed -gt 0 ]]; then
  echo "Failed queries: $failed"
fi
if $RUN_STATS; then
  echo "Statistics: enabled"
else
  echo "Statistics: disabled"
fi

# Display run metadata summary
echo ""
echo "========================================="
echo "RUN METADATA SUMMARY"
echo "========================================="
echo "Timestamp:       $RUN_TIMESTAMP"
echo "Hostname:        $RUN_HOSTNAME"
echo "Seed:            $SEED"
echo ""
echo "Software Versions:"
echo "  SystemDS:      $RUN_SYSTEMDS_VERSION"
echo "  JDK:           $RUN_JDK_VERSION"
echo ""
echo "System Resources:"
echo "  CPU:           $RUN_CPU_INFO"
echo "  RAM:           $RUN_RAM_INFO"
echo ""
echo "Data Build Info:"
echo "  SSB Data:      $RUN_DATA_DISPLAY"
echo "========================================="

# Generate metadata JSON file (include all attempted queries with status and error messages)
{
  echo "{"
  echo "  \"benchmark_type\": \"ssb_systemds\","
  echo "  \"timestamp\": \"$RUN_TIMESTAMP\","
  echo "  \"hostname\": \"$RUN_HOSTNAME\","
  echo "  \"seed\": $SEED,"
  echo "  \"software_versions\": {"
  echo "    \"systemds\": \"$RUN_SYSTEMDS_VERSION\","
  echo "    \"jdk\": \"$RUN_JDK_VERSION\""
  echo "  },"
  echo "  \"system_resources\": {"
  echo "    \"cpu\": \"$RUN_CPU_INFO\","
  echo "    \"ram\": \"$RUN_RAM_INFO\""
  echo "  },"
  echo -e "  \"data_build_info\": $RUN_DATA_INFO,"
  echo "  \"run_configuration\": {"
  echo "    \"statistics_enabled\": $(if $RUN_STATS; then echo "true"; else echo "false"; fi),"
  echo "    \"queries_selected\": ${#QUERIES[@]},"
  echo "    \"queries_executed\": $count,"
  echo "    \"queries_failed\": $failed"
  echo "  },"
  echo "  \"results\": ["
  for i in "${!ALL_RUN_QUERIES[@]}"; do
    query="${ALL_RUN_QUERIES[$i]}"
    status="${QUERY_STATUS[$i]:-error}"
    error_msg="${QUERY_ERROR_MSG[$i]:-}"
    # Find matching full_result and stats by searching SUCCESSFUL_QUERIES index
    full_result=""
    stats_result=""
    if [[ "$status" == "success" ]]; then
      # Find index in SUCCESSFUL_QUERIES
      for j in "${!SUCCESSFUL_QUERIES[@]}"; do
        if [[ "${SUCCESSFUL_QUERIES[$j]}" == "$query" ]]; then
          full_result="${QUERY_FULL_RESULTS[$j]}"
          stats_result="${QUERY_STATS[$j]}"
          break
        fi
      done
    fi
    # Escape quotes and newlines for JSON
    escaped_result=$(echo "$full_result" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')
    escaped_error=$(echo "$error_msg" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')

    echo "    {"
    echo "      \"query\": \"${query%.dml}\","
    echo "      \"status\": \"$status\","
    echo "      \"error_message\": \"$escaped_error\","
    echo "      \"result\": \"$escaped_result\""
    if [[ -n "$stats_result" ]]; then
      echo "      ,\"stats\": ["
      echo "$stats_result" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | sed 's/\t/    /g' | awk '
        BEGIN { first = 1 }
        {
          if (!first) printf ",\n"
          printf "        \"%s\"", $0
          first = 0
        }
        END { if (!first) printf "\n" }
      '
      echo "      ]"
    fi
    if [[ $i -lt $((${#ALL_RUN_QUERIES[@]} - 1)) ]]; then
      echo "    },"
    else
      echo "    }"
    fi
  done
  echo "  ]"
  echo "}"
} > "$OUTPUT_METADATA_FILE"

echo ""
echo "Metadata saved to $OUTPUT_METADATA_FILE"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "  - TXT files: $OUTPUT_TXT_DIR"
echo "  - CSV files: $OUTPUT_CSV_DIR"
echo "  - JSON files: $OUTPUT_JSON_DIR"

# Detailed per-query summary (show status and error messages if any)
if [[ ${#ALL_RUN_QUERIES[@]} -gt 0 ]]; then
  echo ""
  echo "==================================================="
  echo "QUERIES SUMMARY"
  echo "==================================================="
  printf "%-4s %-15s %-30s %s\n" "No." "Query" "Result" "Status"
  echo "---------------------------------------------------"
  for i in "${!ALL_RUN_QUERIES[@]}"; do
    query="${ALL_RUN_QUERIES[$i]}"
    query_display="${query%.dml}"  # Remove .dml extension for display
    status="${QUERY_STATUS[$i]:-error}"
    if [[ "$status" == "success" ]]; then
      # Find index in SUCCESSFUL_QUERIES to fetch result
      result=""
      for j in "${!SUCCESSFUL_QUERIES[@]}"; do
        if [[ "${SUCCESSFUL_QUERIES[$j]}" == "$query" ]]; then
          result="${QUERY_RESULTS[$j]}"
          break
        fi
      done
      printf "%-4d %-15s %-30s %s\n" "$((i+1))" "$query_display" "$result" "✓ Success"
    else
      err="${QUERY_ERROR_MSG[$i]:-Unknown error}"
      printf "%-4d %-15s %-30s %s\n" "$((i+1))" "$query_display" "N/A" "ERROR: ${err}"
    fi
  done
echo "==================================================="
fi

# Display long outputs for queries that had table results
if [[ ${#SUCCESSFUL_QUERIES[@]} -gt 0 ]]; then
  # Check if we have any long outputs to display
  has_long_outputs=false
  for i in "${!LONG_OUTPUTS[@]}"; do
    if [[ -n "${LONG_OUTPUTS[$i]}" ]]; then
      has_long_outputs=true
      break
    fi
  done

  if $has_long_outputs; then
    echo ""
    echo "========================================="
    echo "DETAILED QUERY RESULTS"
    echo "========================================="
    for i in "${!SUCCESSFUL_QUERIES[@]}"; do
      if [[ -n "${LONG_OUTPUTS[$i]}" ]]; then
        query="${SUCCESSFUL_QUERIES[$i]}"
        query_display="${query%.dml}"  # Remove .dml extension for display
        echo ""
        echo "[$((i+1))] Results for $query_display:"
        echo "----------------------------------------"
        echo "${LONG_OUTPUTS[$i]}"
        echo "----------------------------------------"
      fi
    done
    echo "========================================="
  fi
fi

# Exit with appropriate code
if [[ $failed -gt 0 ]]; then
  exit 1
fi
