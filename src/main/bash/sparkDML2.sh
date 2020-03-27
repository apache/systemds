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

#set -x

# This script is a simplified version of sparkDML.sh in order to
# allow a simple drop-in replacement for 'hadoop jar' without
# the need to change any command line arguments. 

#export HADOOP_CONF_DIR=/etc/hadoop/conf
#SPARK_HOME=../spark-2.3.1-bin-hadoop2.7
#export HADOOP_HOME=${HADOOP_HOME:-/usr/hdp/2.5.0.0-1245/hadoop}
#HADOOP_CONF_DIR=${HADOOP_CONF_DIR:-/usr/hdp/2.5.0.0-1245/hadoop/conf}

export SPARK_MAJOR_VERSION=2

#$SPARK_HOME/bin/spark-submit \
spark-submit \
      --master yarn \
      --driver-memory 80g \
      --num-executors 1 \
      --executor-memory 60g \
      --executor-cores 19 \
      --conf "spark.yarn.am.extraJavaOptions -Dhdp.version=2.5.0.0-1245" \
      "$@"

# # run spark submit locally
# spark-submit \
#      "$@"
