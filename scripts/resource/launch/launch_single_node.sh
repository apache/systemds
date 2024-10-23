#!/usr/bin/env bash

# exit in case of error or unbound var
set -euo pipefail

source single_node.env
source single_node_utils.sh

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

echo "Lauchning EC2 instance via AWS cli ..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$UBUNTU_IMAGE_ID" \
    --instance-type "$INSTANCE_TYPE" \
    $( [ -n "$SECURITY_GROUP_NAMES" ] && echo "--security-groups $SECURITY_GROUP_NAMES" ) \
    --iam-instance-profile Name="$INSTANCE_PROFILE_NAME" \
    --block-device-mappings "DeviceName=$ROOT_DEVICE,Ebs=$EBS_CONFIGS" \
    $( [ "$EBS_OPTIMIZED" = "true" ] && echo '--ebs-optimized' ) \
    --key-name "$KEYPAIR_NAME" \
    --user-data fileb://ec2_bootstrap.sh \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' --output text)

set_config "INSTANCE_ID" "$INSTANCE_ID"
echo "... instance with ID $INSTANCE_ID was successfully launched."

echo "Waiting for the initialization to finish ..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_DNS_NAME=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicDnsName' \
    --output text)
echo "... initialization has finished, the assigned public DNS name is: $PUBLIC_DNS_NAME"
set_config "PUBLIC_DNS_NAME" "$PUBLIC_DNS_NAME"


check_installation_status() {
    ssh -o StrictHostKeyChecking=no -i "$KEYPAIR_NAME".pem "ubuntu@$PUBLIC_DNS_NAME" test -f /tmp/systemds_installation_completed
}

echo "Waiting for SystemDS installation ..."
while ! check_installation_status; do
    echo "Installation is not completed yet, waiting 30 seconds..."
    sleep 30
done
echo "... SystemDS installation has completed"

