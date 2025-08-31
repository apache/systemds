#!/usr/bin/env bash
# Wrapper script to run SSB benchmark from project root
#
# This script allows running the SSB benchmark from the main project directory
# by forwarding all arguments to the actual script in the shell/ directory.
#
# USAGE:
#   ./run_ssb.sh             # run all SSB queries
#   ./run_ssb.sh q1.1 q2.3   # run specific queries
#   ./run_ssb.sh --stats     # enable internal statistics
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

# Path to the actual SSB script in the shell directory
SSB_SCRIPT="$PROJECT_ROOT/shell/run_ssb.sh"

# Check if the actual script exists
if [[ ! -f "$SSB_SCRIPT" ]]; then
  echo "Error: SSB script not found at $SSB_SCRIPT" >&2
  exit 1
fi

# Make sure the actual script is executable
if [[ ! -x "$SSB_SCRIPT" ]]; then
  echo "Error: SSB script is not executable: $SSB_SCRIPT" >&2
  echo "       Run: chmod +x $SSB_SCRIPT" >&2
  exit 1
fi

# Execute the actual SSB script with all provided arguments
exec "$SSB_SCRIPT" "$@"
