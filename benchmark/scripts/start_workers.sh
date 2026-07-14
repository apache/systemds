#!/usr/bin/env bash
# Start 4 local SystemDS federated workers on ports 8301-8304.
# Each worker is given the absolute path to its data shard directory.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JAR="$REPO_ROOT/target/SystemDS.jar"
DATA_DIR="$REPO_ROOT/benchmark/data"
LOG_DIR="$REPO_ROOT/benchmark/results"
mkdir -p "$LOG_DIR"

for i in 1 2 3 4; do
  PORT=$((8300 + i))
  echo "Starting worker $i on port $PORT …"
  java --add-modules=jdk.incubator.vector -jar "$JAR" \
      -w "$PORT" \
      > "$LOG_DIR/worker${i}.log" 2>&1 &
  echo $! > "$LOG_DIR/worker${i}.pid"
done

# Give workers time to bind their ports.
sleep 3
echo "Workers running. PIDs:"
for i in 1 2 3 4; do cat "$LOG_DIR/worker${i}.pid"; done

