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

if [ -n "$TARGET_SUBNET" ]; then
    SUBNET=$TARGET_SUBNET
else
    #Get the first available subnet in the default VPC of the configured region
    SUBNET=$(aws ec2 describe-subnets --region $REGION \
      --filter "Name=defaultForAz,Values=true" --query "Subnets[0].SubnetId" --output text)
fi

# generate the step definition into STEP variable
generate_step_definition

echo -e "\nLaunching EMR cluster via AWS CLI and adding a step to run $SYSTEMDS_PROGRAM with SystemDS"
CLUSTER_INFO=$(aws emr create-cluster \
    --applications Name=AmazonCloudWatchAgent Name=Spark \
    --ec2-attributes '{
        "KeyName":"'${KEYPAIR_NAME}'",
        "InstanceProfile":"EMR_EC2_DefaultRole",
        '"$( [ -n "$SECURITY_GROUP_ID'" ] && echo '"AdditionalMasterSecurityGroups": ["'${SECURITY_GROUP_ID}'"],' )"'
        "SubnetId": "'${SUBNET}'"
    }'\
    --service-role EMR_DefaultRole \
    --enable-debugging \
    --release-label $EMR_VERSION \
    --log-uri $LOG_URI \
    --name "SystemDS cluster" \
    --instance-groups file://$INSTANCE_CONFIGS \
    --configurations file://$SPARK_CONFIGS \
    --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
    --no-termination-protected \
    $( [ -n "$STEP" ] && echo "--steps $STEP" ) \
    $( [ "$AUTO_TERMINATION_TIME" = 0 ] && echo "--auto-terminate" ) \
    $( [ "$AUTO_TERMINATION_TIME" -gt 0 ] && echo "--auto-termination-policy IdleTimeout=$AUTO_TERMINATION_TIME" ) \
    --region $REGION)

CLUSTER_ID=$(echo $CLUSTER_INFO | jq .ClusterId | tr -d '"')
echo "Cluster successfully initialized with cluster ID: "${CLUSTER_ID}
set_config "CLUSTER_ID" $CLUSTER_ID

# Wait for cluster to start
echo -e "\nWaiting for cluster to enter running state..."
aws emr wait cluster-running --cluster-id $CLUSTER_ID --region $REGION

CLUSTER_URL=$(aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION | jq .Cluster.MasterPublicDnsName | tr -d '"')
set_config "CLUSTER_URL" "$CLUSTER_URL"

echo "...launching process has finished and the cluster is not in state running."

if [ "$AUTO_TERMINATION_TIME" = 0 ]; then
    echo -e "\nImmediate automatic termination was enabled so the cluster will terminate directly after the step completion"
elif [ "$AUTO_TERMINATION_TIME" -gt 0 ]; then
    echo -e "\nDelayed automatic termination was enabled so the cluster will terminate $AUTO_TERMINATION_TIME
    seconds after entering idle state"
else
    echo -e "\nAutomatic termination was not enabled so you should manually terminate the cluster"
fi