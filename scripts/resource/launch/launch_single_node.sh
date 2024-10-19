#!/usr/bin/env bash

# exit in case of error
set -euo pipefail

source systemds_single_node.env

# $1 key, $2 value
function set_config(){
    sed -i "" "s/\($1 *= *\).*/\1$2/" systemds_single_node.env
}

function get_image_details() {
    INSTANCE_TYPE=$(jq -r '.InstanceType' output/ec2_configurations.json)
    echo "Getting a suitable image for the target EC2 instance: $INSTANCE_TYPE ..."
    if [[ ${INSTANCE_TYPE:2:1} == "g" ]]; then
        ARCHITECTURE="arm64"
    else
        ARCHITECTURE="amd64"
    fi
    # get lates ubuntu 24.04 LTS image for target CPU architecture
    IMAGE_DETAILS=$(aws ec2 describe-images \
                            --owners 099720109477 \
                            --region "$REGION" \
                            --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-$ARCHITECTURE-server-*" \
                            --query "Images | sort_by(@, &CreationDate) | [-1].[ImageId,RootDeviceName]" \
                        )
    echo $IMAGE_DETAILS
    UBUNTU_IMAGE_ID=$(echo "$IMAGE_DETAILS" | jq -r '.[0]')
    ROOT_DEVICE=$(echo "$IMAGE_DETAILS" | jq -r '.[1]')
    echo "... using image with id '$UBUNTU_IMAGE_ID' for $ARCHITECTURE architecture"
    echo ""
}

function generate_ebs_configs() {
    read EBS_OPTIMIZED ROOT_VOLUME_SIZE ROOT_VOLUME_TYPE < \
        <(jq -r '.EbsOptimized, .VolumeSize, .VolumeType' output/ec2_configurations.json | tr '\n' ' ')
    EBS_CONFIGS="{VolumeSize=$ROOT_VOLUME_SIZE,VolumeType=$ROOT_VOLUME_TYPE,DeleteOnTermination=true}"
}

get_image_details # declares and defines $UBUNTU_IMAGE_ID and $ROOT_DEVICE
generate_ebs_configs # declares and defines $EBS_CONFIGS and $EBS_OPTIMIZED

# transform the provided security groups in a suitbale format for the cli option
IFS=',' read -r -a security_group_array <<< "$SECURITY_GROUPS"
SECURITY_GROUP_IDS="${security_group_array[*]}"

echo "Lauchning EC2 instance via AWS cli ..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$UBUNTU_IMAGE_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --security-group-ids $SECURITY_GROUP_IDS \
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
    ssh -o StrictHostKeyChecking=no -i "$KEYPAIR_NAME".pem "ubuntu@$PUBLIC_DNS" test -f /tmp/systemds_installation_completed
}

echo "Waiting for SystemDS installation ..."
while ! check_installation_status; do
    echo "Installtion is not completed yet"
    sleep 30
done
echo "... SystemDS installation has completed"

echo "Launching the DML script for single node execution..."

COMMAND=$""

ssh  -i "$KEYPAIR_NAME".pem "ubuntu@$PUBLIC_DNS_NAME" \
    "nohup bash -c 'systemds -f s3a://systemds-testing/dml_scripts/test.dml -nvargs Y=s3a://systemds-testing/data/Y.csv B=s3a://systemds-testing/data/B.csv;
    gzip -c output.log > output.log.gz && aws s3 cp output_$INSTANCE_ID.log.gz s3://systemds-testing/logs/ --content-type \"text/plain\" --content-encoding \"gzip\" &&
    gzip -c error.log > error.log.gz && aws s3 cp error_$INSTANCE_ID.log.gz s3://systemds-testing/logs/ --content-type \"text/plain\" --content-encoding \"gzip\" &&
    sudo shutdown now' >> output.log 2>> error.log &"

echo "... the program has been launched, now wait to finish"

aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"

echo "The DML finished, the logs where written to s3://systemds-testing/logs/ and the EC2 instance was stopped"
echo "The instance will be terminated directly now..."

aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"

echo "... termination was successful!"