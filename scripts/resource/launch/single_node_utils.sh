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

# $1 key, $2 value
function set_config(){
    sed -i "" "s/\($1 *= *\).*/\1$2/" single_node.env
}

# expects $INSTANCE_TYPE loaded
function get_image_details() {
    echo "Getting a suitable image for the target EC2 instance: $INSTANCE_TYPE ..."
    if [[ ${INSTANCE_TYPE:2:1} == "g" ]]; then
        ARCHITECTURE="arm64"
    else
        ARCHITECTURE="x86_64"
    fi
    # get lates ubuntu 24.04 LTS image for target CPU architecture
    IMAGE_DETAILS=$(aws ec2 describe-images \
                            --owners 137112412989 \
                            --region "$REGION" \
                            --filters "Name=name,Values=al2023-ami-minimal-2023.*.2024*-$ARCHITECTURE" \
                            --query "Images | sort_by(@, &CreationDate) | [-1].[ImageId,RootDeviceName]" \
                        )
    UBUNTU_IMAGE_ID=$(echo "$IMAGE_DETAILS" | jq -r '.[0]')
    ROOT_DEVICE=$(echo "$IMAGE_DETAILS" | jq -r '.[1]')
    echo "... using image with id '$UBUNTU_IMAGE_ID' for $ARCHITECTURE architecture"
    echo ""
}

# expects $ROOT_VOLUME_SIZE and $ROOT_VOLUME_INSTANCE_TYPE loaded
function generate_ebs_configs() {

    EBS_CONFIGS="{VolumeSize=$ROOT_VOLUME_SIZE,VolumeType=$ROOT_VOLUME_TYPE,DeleteOnTermination=true}"
    echo "Using the following EBS_CONFIGS configurations:"
    echo $EBS_CONFIGS
    echo
}

# expects $CONFIGURATIONS loaded
function generate_jvm_configs() {
    JVM_MAX_MEM=$(jq -r '.JvmMaxMemory' $CONFIGURATIONS)
    JVM_START_MEM=$(echo "$JVM_MAX_MEM * 0.7" | bc | awk '{print int($1)}')
    JVM_YOUNG_GEN_MEM=$(echo "$JVM_MAX_MEM * 0.1" | bc | awk '{print int($1)}')
    echo "The target instance $INSTANCE_TYPE will be setup to use ${JVM_MAX_MEM}MB at executing SystemDS programs"
}

# create EC2 instance profile with corresponding IAM role for S3 access
function generate_instance_profile() {
    if aws iam get-role --role-name "$IAM_ROLE_NAME" >/dev/null 2>&1; then
        echo "Role $IAM_ROLE_NAME already exists."
    else
        echo "Role $IAM_ROLE_NAME does not exist. Creating role..."

        # temp trust policy for EC2 to allow assuming the role
        cat > trust-policy.json <<EOL
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
            "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOL

        # 1. create the IAM role
        aws iam create-role --role-name "$IAM_ROLE_NAME" --assume-role-policy-document file://trust-policy.json  >/dev/null

        # 2. attach the relevant policies to the role
        aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
        if [ -n $CLOUDWATCH_CONFIGS ]; then
            aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" \
                --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
            aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" \
                --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
        fi

        echo "Role $IAM_ROLE_NAME has been created and AmazonS3FullAccess policy attached."

        # delete the temp trust policy
        rm trust-policy.json
    fi

    # create an according IAM instance policy if not created in previous runs
    if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null; then
        echo "Instance profile $INSTANCE_PROFILE_NAME already exists."
    else
        echo "Instance profile $INSTANCE_PROFILE_NAME does not exist. Creating..."
        # 1. create the instance profile
        aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME"  >/dev/null

        # 2. attach the IAM role for S3 to the instance profile
        aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --role-name "$IAM_ROLE_NAME"

        echo "Instance profile $INSTANCE_PROFILE_NAME created"
    fi
}

function check_installation_status() {
    ssh -o StrictHostKeyChecking=no -i "$KEYPAIR_NAME".pem "ec2-user@$PUBLIC_DNS_NAME" \
    'while [ ! -f /tmp/systemds_installation_completed ]; do sleep 5; done;'
}

function start_cloudwatch_agent() {
    # launch the ssm agent first
    ssh -i "$KEYPAIR_NAME".pem "ec2-user@$PUBLIC_DNS_NAME" sudo systemctl start amazon-ssm-agent
    sleep 5
    # configure and launch start_cloudwatch agent with pre-defined SSM command
    aws ssm send-command --document-name "AmazonCloudWatch-ManageAgent" \
        --targets "Key=InstanceIds,Values=$INSTANCE_ID" \
        --parameters "action=configure,mode=ec2,optionalConfigurationSource=ssm,optionalConfigurationLocation=$CLOUDWATCH_CONFIGS,optionalRestart=yes" \
        --region $REGION  >/dev/null
    sleep 5 # sleep 5 seconds should always be enough for execution of the underlying command
}