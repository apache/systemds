#!/bin/bash
# Standalone-Launcher für SystemDS

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
JAR_FILE="$SCRIPT_DIR/../target/systemds-3.4.0-SNAPSHOT.jar"

if [ ! -f "$JAR_FILE" ]; then
  echo "ERROR: Standalone JAR nicht gefunden: $JAR_FILE"
  exit 1
fi

java -cp "$JAR_FILE" org.apache.sysds.api.DMLScript "$@"
