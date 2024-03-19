#!/bin/bash

# Execution of this script prepare the
# the AWS platform before launching an
# EMR cluster for the first time.
# It also initialize the local config
# file to the specific AWS account attributes

# Load common functions
source common.sh

# Load the manually set configs
source systemds_cluster.config

# Validate the given region value
aws ec2 describe-instances --region "$REGION"
if [ $? -ne 0 ]; then
  echo "Invalid region '$REGION'. Exiting..."
  exit 1
fi


# Ensure the existence of/ Create the default roles
aws emr create-default-roles --region "$REGION" &> /dev/null

# Ensure the existence of/ Create the default VPC
aws ec2 create-default-vpc --region "$REGION" &> /dev/null

# Get a default subnet
if [ "$AZ" = "" ]; then
  SUBNET_ID=$(aws ec2 describe-subnets --region $REGION \
              --filter "Name=defaultForAz,Values=true" --query "Subnets[0].SubnetId" --output text)
else
  SUBNET_ID=$(aws ec2 describe-subnets --region $REGION \
                --filter "Name=defaultForAz,Values=true" "Name=availabilityZone,Values=$AZ"
                --query "Subnets[0].SubnetId" --output text)
fi

if [ "$SUBNET_ID" = "None" ]; then
  echo "Availability Zone '$AZ' is not valid for region '$REGION'. Exiting..."
  exit 1
else
  set_config "SUBNET_ID" "$SUBNET_ID"
fi

# Create keypair
if [ ! -f ${KEYPAIR_NAME}.pem ]; then
    aws ec2 create-key-pair --key-name $KEYPAIR_NAME --region $REGION --query "KeyMaterial" --output text > "$KEYPAIR_NAME.pem"
    chmod 700 "${KEYPAIR_NAME}.pem"
    echo "${KEYPAIR_NAME}.pem private key created!"
fi