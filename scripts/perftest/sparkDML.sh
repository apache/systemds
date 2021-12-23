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

export SPARK_HOME=../spark-2.4.7-bin-hadoop2.7
export HADOOP_CONF_DIR=/home/hadoop/hadoop-2.7.7/etc/hadoop

$SPARK_HOME/bin/spark-submit \
     --master yarn \
     --deploy-mode client \
     --driver-memory 20g \
     --conf spark.driver.extraJavaOptions="-Xms20g -Dlog4j.configuration=file:/home/mboehm/perftest/conf/log4j.properties" \
     --conf spark.ui.showConsoleProgress=true \
     --conf spark.executor.heartbeatInterval=100s \
     --conf spark.network.timeout=512s \
     --num-executors 10 \
     --executor-memory 105g \
     --executor-cores 32 \
     SystemDS.jar "$@" 