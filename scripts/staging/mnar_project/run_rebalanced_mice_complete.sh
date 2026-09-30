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

set -e

base_dir="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="$base_dir/mnar_project/data/processed"
dml_file="$base_dir/mnar_project/dml/rebalanced_mice_complete.dml"
systemds_root="$base_dir/systemds"

java_home_default="/opt/homebrew/Cellar/openjdk@17/17.0.19/libexec/openjdk.jdk/Contents/Home"

dataset_name="${1:-diabetes}"
iter="${2:-5}"

if [ -z "$JAVA_HOME" ]; then
  export JAVA_HOME="$java_home_default"
fi

export PATH="$JAVA_HOME/bin:$PATH"
systemds_jar_file="$systemds_root/target/systemds-3.4.0-SNAPSHOT.jar"

if [ -f "$systemds_jar_file" ]; then
  export SYSTEMDS_JAR_FILE="$systemds_jar_file"
fi

if [ "$dataset_name" = "diabetes" ]; then
  input_file="$data_dir/diabetes_mnar_systemds.csv"
  output_file="/private/tmp/rebalanced_mice_complete_diabetes_out.csv"
  tuning_file="/private/tmp/rebalanced_mice_complete_diabetes_tuning.csv"
elif [ "$dataset_name" = "diabetes_full" ]; then
  input_file="$data_dir/diabetes_full_mnar_systemds.csv"
  output_file="/private/tmp/rebalanced_mice_complete_diabetes_full_out.csv"
  tuning_file="/private/tmp/rebalanced_mice_complete_diabetes_full_tuning.csv"
elif [ "$dataset_name" = "AQ" ]; then
  input_file="$data_dir/AQ_mnar_systemds.csv"
  output_file="/private/tmp/rebalanced_mice_complete_AQ_out.csv"
  tuning_file="/private/tmp/rebalanced_mice_complete_AQ_tuning.csv"
else
  echo "unknown dataset: $dataset_name"
  echo "use: diabetes, diabetes_full or AQ"
  exit 1
fi

echo "dataset: $dataset_name"
echo "input: $input_file"
echo "output: $output_file"
echo "tuning: $tuning_file"
echo "iter: $iter"

SYSTEMDS_ROOT="$systemds_root" \
"$systemds_root/bin/systemds" \
-f "$dml_file" \
-args "$input_file" "$output_file" "$tuning_file" "$iter"
