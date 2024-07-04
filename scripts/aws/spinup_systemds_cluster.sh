#!/bin/bash
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

source systemds_cluster.config

# $1 is the instance type
get_memory_and_cores () {
  while IFS="," read -r field1 name memory vCPUs field5 field6
    do
      if [ "$1" = "$name" ]; then
        SPARK_EXECUTOR_MEMORY=$( bc <<< $memory*0.60*1000) #60% of the total memory (beacause of yarn.scheduler.maximum-allocation-mb)
        SPARK_EXECUTOR_CORES=$vCPUs
      fi
    done < aws_ec2_table.csv

}
# $1 key, $2 value
function set_config(){
    sed -i "" "s/\($1 *= *\).*/\1$2/" systemds_cluster.config
}

get_memory_and_cores $INSTANCES_TYPE
SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY}MB"
set_config "SPARK_NUM_EXECUTORS" $CORE_INSTANCES_COUNT
set_config "SPARK_EXECUTOR_CORES" $SPARK_EXECUTOR_CORES
set_config "SPARK_EXECUTOR_MEMORY" $SPARK_EXECUTOR_MEMORY
set_config "SPARK_DRIVER_MEMORY" "1G"
set_config "BUCKET" $BUCKET-$(((RANDOM % 999) + 1000))

#Source again to update the changes for the current session
source systemds_cluster.config

#Create systemDS bucket
#LocationConstraint configuration required regions outside of us-east-1
if [ "$REGION" = "us-east-1" ]; then LOCATION_CONSTRAINT=""; else LOCATION_CONSTRAINT="--create-bucket-configuration LocationConstraint=$REGION"; fi
aws s3api create-bucket --bucket $BUCKET --region $REGION $LOCATION_CONSTRAINT &> /dev/null
aws s3api create-bucket --bucket $BUCKET-logs --region $REGION $LOCATION_CONSTRAINT &> /dev/null

# Upload Jar and scripts to s3
aws s3 sync $SYSTEMDS_TARGET_DIRECTORY s3://$BUCKET --exclude "*" --include "*.dml" --include "*config.xml" --include "*DS.jar*"

# Create keypair
if [ ! -f ${KEYPAIR_NAME}.pem ]; then
    aws ec2 create-key-pair --key-name $KEYPAIR_NAME --region $REGION --query "KeyMaterial" --output text > "$KEYPAIR_NAME.pem"
    chmod 700 "${KEYPAIR_NAME}.pem"
    echo "${KEYPAIR_NAME}.pem private key created!"
fi

#Get the first available subnet in the default VPC of the configured region
DEFAULT_SUBNET=$(aws ec2 describe-subnets --region $REGION \
  --filter "Name=defaultForAz,Values=true" --query "Subnets[0].SubnetId" --output text)


#Create the cluster
#Note: Ganglia not available since emr-6.15.0: exchanged with AmazonCloudWatchAgent
CLUSTER_INFO=$(aws emr create-cluster \
 --applications Name=AmazonCloudWatchAgent Name=Spark \
 --ec2-attributes '{"KeyName":"'${KEYPAIR_NAME}'",
  "InstanceProfile":"EMR_EC2_DefaultRole",
  "SubnetId": "'${DEFAULT_SUBNET}'"}'\
 --service-role EMR_DefaultRole \
 --enable-debugging \
 --release-label $EMR_VERSION \
 --log-uri "s3://$BUCKET-logs/" \
 --name "SystemDS cluster" \
 --instance-groups '[{"InstanceCount":'${MASTER_INSTANCES_COUNT}',
                        "InstanceGroupType":"MASTER",
                        "InstanceType":"'${INSTANCES_TYPE}'",
                        "Name":"Master Instance Group"},
                      {"InstanceCount":'${CORE_INSTANCES_COUNT}',
                        "InstanceGroupType":"CORE",
                        "InstanceType":"'${INSTANCES_TYPE}'",
                        "Name":"Core Instance Group"}]'\
 --configurations '[{"Classification":"spark","Properties":{"maximizeResourceAllocation": "true"}},
                     {"Classification": "spark-env",
                         "Configurations": [{
                           "Classification": "export",
                           "Properties": {"JAVA_HOME": "/usr/lib/jvm/jre-11"}
                         }]
                     }]'\
 --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
 --region $REGION)

CLUSTER_ID=$(echo $CLUSTER_INFO | jq .ClusterId | tr -d '"')
echo "Cluster successfully initialized. Save your ClusterID: "${CLUSTER_ID}
set_config "CLUSTER_ID" $CLUSTER_ID

ip_address=$(curl ipecho.net/plain ; echo)

#Add your ip to the security group
aws ec2 create-security-group --group-name ElasticMapReduce-master --description "info" --region $REGION &> /dev/null
aws ec2 authorize-security-group-ingress \
    --group-name ElasticMapReduce-master \
    --protocol tcp \
    --port 22 \
    --cidr "${ip_address}"/24 \
    --region $REGION &> /dev/null

# Wait for cluster to start
echo "Waiting for cluster running state"
aws emr wait cluster-running --cluster-id $CLUSTER_ID --region $REGION

echo "Cluster info:"
export CLUSTER_URL=$(aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION | jq .Cluster.MasterPublicDnsName | tr -d '"')

aws emr ssh --cluster-id $CLUSTER_ID --key-pair-file ${KEYPAIR_NAME}.pem --region $REGION \
    --command 'aws s3 cp s3://'${BUCKET}' . --recursive --exclude "*" --include "*DS.jar*"'

echo "Spinup finished."

