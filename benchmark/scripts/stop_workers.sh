#!/usr/bin/env bash
LOG_DIR="$(cd "$(dirname "$0")/../../benchmark/results" && pwd)"
for i in 1 2 3 4; do
  PID_FILE="$LOG_DIR/worker${i}.pid"
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null && echo "Stopped worker $i (PID $PID)" || true
    rm -f "$PID_FILE"
  fi
done

