#!/usr/bin/env bash
set -euo pipefail


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
if [[ -x "$PROJECT_ROOT/bin/systemds" ]]; then
  SYSTEMDS_CMD="$PROJECT_ROOT/bin/systemds"

elif command -v systemds >/dev/null 2>&1; then
  SYSTEMDS_CMD="$(command -v systemds)"

else
  echo "Error: could not find SystemDS executable." >&2
  echo "       Build ./bin/systemds or add it to \$PATH." >&2
  exit 1
fi

QUERY_DIR="$PROJECT_ROOT/scripts/ssb/queries"

RUN_STATS=false

QUERIES=()


for arg in "$@"; do
  if [[ "$arg" == "--stats" ]]; then
    RUN_STATS=true
  else
    query_file=$(echo "$arg" | tr '.' '_').dml
    QUERIES+=("$query_file")
  fi
done
if [ ${#QUERIES[@]} -eq 0 ]; then
  QUERIES=($(ls "$QUERY_DIR"/q*.dml | xargs -n1 basename))
fi

for query in "${QUERIES[@]}"; do
  query_path="$QUERY_DIR/$query"
  echo "Running: $query_path"

  if [ "$RUN_STATS" = true ]; then
    $SYSTEMDS_CMD "$query_path" -stats
  else
    $SYSTEMDS_CMD "$query_path"
  fi
done