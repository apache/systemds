#!/usr/bin/env bash

set -euo pipefail

source systemds_single_node.env

# create IAM role for S3 access if not created in previous runs
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
    aws iam create-role --role-name "$IAM_ROLE_NAME" --assume-role-policy-document file://trust-policy.json  >/dev/null 2>&1
    
    # 2. attach a policy to the role
    aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    
    echo "Role $IAM_ROLE_NAME has been created and AmazonS3FullAccess policy attached."
    
    # delete the temp trust policy
    rm trust-policy.json
fi

# create an according IAM instance policy if not created in previous runs
if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null 2>&1; then
    echo "Instance profile $INSTANCE_PROFILE_NAME already exists."
else
    echo "Instance profile $INSTANCE_PROFILE_NAME does not exist. Creating..."
    # 1. create the instance profile
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME"  >/dev/null 2>&1
    
    # 2. attach the IAM role for S3 to the instance profile
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --role-name "$IAM_ROLE_NAME"

    echo "Instance profile $INSTANCE_PROFILE_NAME created"
fi
