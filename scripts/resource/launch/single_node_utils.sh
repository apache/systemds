#!/usr/bin/env bash

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
        ARCHITECTURE="amd64"
    fi
    # get lates ubuntu 24.04 LTS image for target CPU architecture
    IMAGE_DETAILS=$(aws ec2 describe-images \
                            --owners 099720109477 \
                            --region "$REGION" \
                            --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-$ARCHITECTURE-server-*" \
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
    echo "The target instance $INSTANCE_TYPE will be setup to use ${JVM_MAX_MEM}MB"
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

        # 2. attach a policy to the role
        aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

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