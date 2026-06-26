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

# Deploy + run SystemDS on a Databricks cluster.
#
# Tested against DBR 16.4 LTS (Spark 3.5.2 / Scala 2.12), where the SystemDS
# jar runs unchanged.
#
# Quick start:
#   1. Copy scripts/databricks/.env.example to .env and edit it (or place the
#      .env at the root of your workspace; this script searches parent dirs).
#   2. Authenticate the Databricks CLI once, interactively:
#        databricks auth login --profile <your-profile>
#   3. Build the SystemDS jar (mvn -q -DskipTests package) so target/SystemDS.jar
#      exists, or point JAR_LOCAL at an existing jar.
#   4. Run a step:
#        ./deploy.sh upload     # create UC volume + copy SystemDS.jar into it
#        ./deploy.sh cluster    # create single-user cluster + install SystemDS.jar
#        ./deploy.sh libs       # install Delta Kernel Maven libraries on cluster
#        ./deploy.sh import     # import the demo notebook(s)
#        ./deploy.sh all        # upload + cluster + libs + import
#
# All configuration is read from environment variables (see .env.example).
# Anything already exported in your shell overrides the .env file.
#
# Hard-won requirements baked in below:
#   - Cluster creation requires a compute policy (e.g. Personal Compute) that
#     restricts node types and autotermination.
#   - UC clusters only accept JAR libraries from a UC Volume (not DBFS, not
#     /Workspace).
#   - Must be SINGLE_USER (Assigned) mode; shared/USER_ISOLATION blocks JAR libs.
#   - SystemDS needs the Vector API module + a full --add-opens set at JVM
#     launch, and an absolute scratch dir (the notebook sets sysds.scratch).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#-------------------------------------------------------------
# Load configuration from a .env file.
# Resolution order:
#   1. $ENV_FILE if explicitly set.
#   2. The first .env found walking up from this script's directory
#      (script dir -> repo root -> workspace root -> ... -> /).
#-------------------------------------------------------------
find_env_file() {
  if [[ -n "${ENV_FILE:-}" ]]; then
    printf '%s\n' "$ENV_FILE"
    return 0
  fi
  local dir="$HERE"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/.env" ]]; then
      printf '%s\n' "$dir/.env"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

if ENV_PATH="$(find_env_file)"; then
  echo ">> loading config from $ENV_PATH"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_PATH"
  set +a
else
  echo ">> no .env found; relying on exported environment variables / defaults"
fi

#-------------------------------------------------------------
# Configuration (env var with sensible fallback).
#-------------------------------------------------------------
PROFILE="${PROFILE:-DEFAULT}"

# Repo root is used to locate the default jar (target/SystemDS.jar).
REPO_ROOT="${REPO_ROOT:-$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/../..")}"
JAR_LOCAL="${JAR_LOCAL:-$REPO_ROOT/target/SystemDS.jar}"

db() { databricks -p "$PROFILE" "$@"; }

# Resolve the current user lazily (only needed for the notebook import target).
resolve_user() {
  if [[ -z "${USER_NAME:-}" ]]; then
    USER_NAME="$(db current-user me -o json \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')"
  fi
}

CATALOG="${CATALOG:-main}"
SCHEMA="${SCHEMA:-default}"
VOLUME="${VOLUME:-systemds}"
VOL_DIR="/Volumes/$CATALOG/$SCHEMA/$VOLUME"
JAR_REMOTE="$VOL_DIR/SystemDS.jar"

POLICY_ID="${POLICY_ID:-}"
SPARK_VERSION="${SPARK_VERSION:-16.4.x-scala2.12}"
NODE_TYPE="${NODE_TYPE:-i3.xlarge}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AUTOTERMINATION_MINUTES="${AUTOTERMINATION_MINUTES:-30}"
CLUSTER_NAME="${CLUSTER_NAME:-systemds}"
CLUSTER_ID_FILE="${CLUSTER_ID_FILE:-$HERE/.cluster_id}"

# Delta Kernel is not on the DBR classpath and is not bundled in SystemDS.jar
# (the uber jar shades only wink + antlr). The native Delta read/write path needs
# delta-kernel-api + delta-kernel-defaults installed as cluster Maven libraries.
# Use >= 3.3.2: earlier releases (3.3.0/3.3.1) subclass parquet-mr's
# package-private InternalParquetRecordReader, which breaks across Databricks'
# library/app classloaders (IllegalAccessError, then NoSuchMethodError against
# DBR's parquet). 3.3.2 (delta PR #4494) switched to parquet's public
# ParquetReader API, so Kernel works with DBR's own parquet. 4.x targets Spark 4
# (wrong for DBR 16.4 = Spark 3.5). Must match the delta-kernel.version in pom.xml.
DELTA_KERNEL_VERSION="${DELTA_KERNEL_VERSION:-3.3.2}"

# SystemDS JVM flags (mirror of <jvm.addopens> in pom.xml).
JVMOPTS="${JVMOPTS:---add-modules=jdk.incubator.vector \
--add-opens=java.base/java.nio=ALL-UNNAMED \
--add-opens=java.base/java.io=ALL-UNNAMED \
--add-opens=java.base/java.util=ALL-UNNAMED \
--add-opens=java.base/java.lang=ALL-UNNAMED \
--add-opens=java.base/java.lang.ref=ALL-UNNAMED \
--add-opens=java.base/java.util.concurrent=ALL-UNNAMED \
--add-opens=java.base/sun.nio.ch=ALL-UNNAMED}"

# Notebooks to import (space-separated, relative to this dir). Language is
# detected from the extension (.scala -> SCALA, .py -> PYTHON).
NB_FILES="${NB_FILES:-SystemDS_MLContext_Demo.scala SystemDS_vs_SparkML_LinReg.scala SystemDS_Delta_E2E.scala SystemDS_Python_Demo.py}"

#-------------------------------------------------------------
# Steps.
#-------------------------------------------------------------
upload() {
  [[ -f "$JAR_LOCAL" ]] || { echo "!! jar not found: $JAR_LOCAL (build it or set JAR_LOCAL)"; exit 1; }
  echo ">> ensuring UC volume $CATALOG.$SCHEMA.$VOLUME"
  db volumes create "$CATALOG" "$SCHEMA" "$VOLUME" MANAGED 2>/dev/null || true
  echo ">> uploading $JAR_LOCAL -> $JAR_REMOTE"
  db fs cp --overwrite "$JAR_LOCAL" "dbfs:$JAR_REMOTE"
}

cluster() {
  resolve_user
  echo ">> creating cluster $CLUSTER_NAME ($SPARK_VERSION, $NODE_TYPE, single-user)"
  local policy_json=""
  if [[ -n "$POLICY_ID" ]]; then
    policy_json="\"policy_id\": \"$POLICY_ID\", \"apply_policy_default_values\": true,"
  fi
  CID=$(db clusters create --no-wait -o json --json "{
    \"cluster_name\": \"$CLUSTER_NAME\",
    $policy_json
    \"spark_version\": \"$SPARK_VERSION\",
    \"node_type_id\": \"$NODE_TYPE\",
    \"num_workers\": $NUM_WORKERS,
    \"autotermination_minutes\": $AUTOTERMINATION_MINUTES,
    \"data_security_mode\": \"SINGLE_USER\",
    \"single_user_name\": \"$USER_NAME\",
    \"spark_conf\": {
      \"spark.databricks.cluster.profile\": \"singleNode\",
      \"spark.master\": \"local[*]\",
      \"spark.driver.extraJavaOptions\": \"$JVMOPTS\",
      \"spark.executor.extraJavaOptions\": \"$JVMOPTS\"
    }
  }" | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster_id"])')
  echo "cluster_id=$CID"
  echo "$CID" > "$CLUSTER_ID_FILE"
  echo ">> installing library $JAR_REMOTE (queued; installs once RUNNING)"
  db libraries install --json "{\"cluster_id\":\"$CID\",\"libraries\":[{\"jar\":\"$JAR_REMOTE\"}]}"
}

# Resolve the target cluster id: prefer the one written by cluster(), else look
# it up by name.
resolve_cluster_id() {
  if [[ -n "${CLUSTER_ID:-}" ]]; then
    return 0
  fi
  if [[ -f "$CLUSTER_ID_FILE" ]]; then
    CLUSTER_ID="$(cat "$CLUSTER_ID_FILE")"
    return 0
  fi
  CLUSTER_ID="$(db clusters list -o json \
    | python3 -c 'import sys,json;
clusters=json.load(sys.stdin);
m=[c for c in clusters if c.get("cluster_name")=="'"$CLUSTER_NAME"'"];
print(m[0]["cluster_id"] if m else "")')"
  [[ -n "$CLUSTER_ID" ]] || { echo "!! could not resolve cluster id for $CLUSTER_NAME (run ./deploy.sh cluster first or set CLUSTER_ID)"; exit 1; }
}

# Install the Delta Kernel Maven libraries on the cluster. delta-kernel-defaults
# pulls delta-kernel-api transitively; both come from Maven Central. parquet,
# hadoop and jackson are excluded so Kernel uses DBR's own copies (avoids
# duplicate-class / split-package issues across the library/app classloaders).
libs() {
  resolve_cluster_id
  echo ">> installing Delta Kernel $DELTA_KERNEL_VERSION Maven libs on $CLUSTER_ID"
  db libraries install --json "{
    \"cluster_id\": \"$CLUSTER_ID\",
    \"libraries\": [
      {\"maven\": {
        \"coordinates\": \"io.delta:delta-kernel-defaults:$DELTA_KERNEL_VERSION\",
        \"exclusions\": [
          \"org.apache.parquet:parquet-hadoop\",
          \"org.apache.hadoop:hadoop-client-runtime\",
          \"org.apache.hadoop:hadoop-client-api\",
          \"com.fasterxml.jackson.core:jackson-databind\"
        ]
      }}
    ]
  }"
  echo ">> queued; check status with: db libraries cluster-status $CLUSTER_ID"
}

import() {
  resolve_user
  local dir="${NB_DIR:-/Users/$USER_NAME}"
  local nb lang base
  for nb in $NB_FILES; do
    case "$nb" in
      *.scala) lang="SCALA" ;;
      *.py)    lang="PYTHON" ;;
      *.sql)   lang="SQL" ;;
      *.r|*.R) lang="R" ;;
      *) echo "!! skipping $nb (unknown notebook language)"; continue ;;
    esac
    base="$(basename "${nb%.*}")"
    echo ">> importing $nb -> $dir/$base ($lang)"
    db workspace import --overwrite --language "$lang" --format SOURCE \
      --file "$HERE/$nb" "$dir/$base"
  done
}

case "${1:-all}" in
  upload)  upload ;;
  cluster) cluster ;;
  libs)    libs ;;
  import)  import ;;
  all)     upload; cluster; libs; import ;;
  *) echo "usage: $0 {upload|cluster|libs|import|all}"; exit 1 ;;
esac
