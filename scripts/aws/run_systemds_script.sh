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

# $1 Dml script name.
# $2 Args

# If number of arguments are not equal to 1.
if [[ ($# -ne 2 && $# -ne 1) ]] ; then
      echo "Usage: "$0" <DML script path> (<Arguments>)"
      exit 2
fi


source systemds_cluster.config

aws s3 cp $1 s3://system-ds-bucket/ --exclude "*" --include "*.dml"

if [ ! -z "$2" ]
then
  args="-args,${2}"
fi

dml_filename=$(basename $1)

STEP_INFO=$(aws emr add-steps --cluster-id $CLUSTER_ID --steps "Type=Spark,
  Name='SystemDS Spark Program',
  ActionOnFailure=CONTINUE,
  Args=[
        --deploy-mode,$SPARK_DEPLOY_MODE,
        --master,yarn,
        --driver-memory,$SPARK_DRIVER_MEMORY,
        --num-executors,$SPARK_NUM_EXECUTORS,
        --conf,spark.driver.maxResultSize=0,
        $SYSTEMDS_JAR_PATH, -f, s3://system-ds-bucket/$dml_filename, -exec, $SYSTEMDS_EXEC_MODE,$args,-stats, -explain]")

STEP_ID=$(echo $STEP_INFO | jq .StepIds | tr -d '"' | tr -d ']' | tr -d '[' | tr -d '[:space:]' )
echo "Waiting for the step to finish"
aws emr wait step-complete --cluster-id $CLUSTER_ID --step-id $STEP_ID

aws emr ssh --cluster-id $CLUSTER_ID --key-pair-file ${KEYPAIR_NAME}.pem --command "cat /mnt/var/log/hadoop/steps/$STEP_ID/stderr"
aws emr ssh --cluster-id $CLUSTER_ID --key-pair-file ${KEYPAIR_NAME}.pem --command "cat /mnt/var/log/hadoop/steps/$STEP_ID/stdout"