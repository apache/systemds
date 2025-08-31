#!/usr/bin/env bash
# Wrapper script to run SSB multi-engine performance benchmark from project root
#
# This script allows running the SSB performance benchmark from the main project directory
# by forwarding all arguments to the actual script in the shell/ directory.
#
# USAGE:
#   ./run_all_perf.sh                    # Run all SSB queries on all engines
#   ./run_all_perf.sh q1_1 q2_3          # Run specific queries only
#   ./run_all_perf.sh --stats            # Run with SystemDS internal statistics
#   ./run_all_perf.sh --stats q1_1       # Run specific query with stats
#
set -euo pipefail

# Get the directory where this wrapper script is located (project root)
if command -v realpath >/dev/null 2>&1; then
  PROJECT_ROOT="$(dirname "$(realpath "$0")")"
else
  PROJECT_ROOT="$(python - <<'PY'
import os, sys
print(os.path.dirname(os.path.abspath(sys.argv[1])))
PY
"$0")"
fi

# Path to the actual performance benchmark script in the shell directory
PERF_SCRIPT="$PROJECT_ROOT/shell/run_all_perf.sh"

# Check if the actual script exists
if [[ ! -f "$PERF_SCRIPT" ]]; then
  echo "Error: Performance benchmark script not found at $PERF_SCRIPT" >&2
  exit 1
fi

# Make sure the actual script is executable
if [[ ! -x "$PERF_SCRIPT" ]]; then
  echo "Error: Performance benchmark script is not executable: $PERF_SCRIPT" >&2
  echo "       Run: chmod +x $PERF_SCRIPT" >&2
  exit 1
fi

# Execute the actual performance benchmark script with all provided arguments
exec "$PERF_SCRIPT" "$@"
