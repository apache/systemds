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

source single_node.env
source "$SCRIPT_DIR/single_node_utils.sh"

read INSTANCE_TYPE EBS_OPTIMIZED ROOT_VOLUME_SIZE ROOT_VOLUME_TYPE <<< $(
    jq -r '.InstanceType, .EbsOptimized, .VolumeSize, .VolumeType' $CONFIGURATIONS |
    tr '\n' ' '
)
set_config "INSTANCE_TYPE" $INSTANCE_TYPE

get_image_details # declares and defines $UBUNTU_IMAGE_ID and $ROOT_DEVICE
generate_ebs_configs # declares and defines $EBS_CONFIGS and $EBS_OPTIMIZED
# generate them first here to ensure there declares valid values before instance launch
generate_jvm_configs # parse the JVM configs

# transform the provided security groups in a suitable format for the cli option
IFS=',' read -r -a security_group_array <<< "$SECURITY_GROUPS"
SECURITY_GROUP_NAMES="${security_group_array[*]}"

# create IAM role for S3 access if not created in previous runs
generate_instance_profile

echo -e "\nLauchning EC2 instance via AWS cli ..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$UBUNTU_IMAGE_ID" \
    --instance-type "$INSTANCE_TYPE" \
    $( [ -n "$SECURITY_GROUP_NAMES" ] && echo "--security-groups $SECURITY_GROUP_NAMES" ) \
    --iam-instance-profile Name="$INSTANCE_PROFILE_NAME" \
    --block-device-mappings "DeviceName=$ROOT_DEVICE,Ebs=$EBS_CONFIGS" \
    $( [ "$EBS_OPTIMIZED" = "true" ] && echo '--ebs-optimized' ) \
    --key-name "$KEYPAIR_NAME" \
    --monitoring Enabled=true \
    --user-data fileb://$SCRIPT_DIR/ec2_bootstrap.sh \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' --output text)

set_config "INSTANCE_ID" "$INSTANCE_ID"
echo "... instance with ID $INSTANCE_ID was successfully launched."

echo -e "\nWaiting for the initialization to finish ..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_DNS_NAME=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicDnsName' \
    --output text)
echo "... initialization has finished, the assigned public DNS name is: $PUBLIC_DNS_NAME"
set_config "PUBLIC_DNS_NAME" "$PUBLIC_DNS_NAME"

echo -e "\nWaiting for SystemDS installation ..."
check_installation_status
echo "... bootstrapping (including SystemDS installation) has completed"

if [ -n $CLOUDWATCH_CONFIGS ]; then
    echo -e "\nConfiguring and launching Cloudwatch agent ..."
    start_cloudwatch_agent
    echo "...cloudwatch agent with successfully launched"
fi
