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

# exit in case of error or unbound var
set -euo pipefail

# get file directory to allow finding the file with the utils
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source cluster.env
source "$SCRIPT_DIR/cluster_utils.sh"

# generate the step definition into STEP variable
generate_step_definition
if [ $STEP -z ]; then
    echo "Error: Empty state definition, probably due to empty SYSTEMDS_PROGRAM option."
    exit 1
fi

echo "Adding a step to run $SYSTEMDS_PROGRAM with SystemDS"
STEP_INFO=$(aws emr add-steps --cluster-id $CLUSTER_ID  --region $REGION --steps $STEP)

if [ "$AUTO_TERMINATION_TIME" = 0 ]; then
    STEP_ID=$(echo $STEP_INFO | jq .StepIds | tr -d '"' | tr -d ']' | tr -d '[' | tr -d '[:space:]' )
    echo "Waiting for the step to finish before termination (immediate automatic termination enabled)"
    aws emr wait step-complete --cluster-id $CLUSTER_ID --step-id $STEP_ID --region $REGION
    echo "The step has finished and now the cluster will before immediately terminated"
    aws emr terminate-clusters --cluster-ids $CLUSTER_ID
elif [ "$AUTO_TERMINATION_TIME" -gt 0 ]; then
    echo "Delayed automatic termination will apply only in case this option was set on cluster launch."
    echo "You should manually track the step completion"
else
    echo "Automatic termination was not enabled so you should manually track the step completion and terminate the cluster"
fi


