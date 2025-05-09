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

 #Client mode spark-submit script
export SPARK_HOME=/home/hadoop/spark-3.3.1-bin-hadoop3
export HADOOP_CONF_DIR=/home/hadoop/hadoop-3.3.1/etc/hadoop

$SPARK_HOME/bin/spark-submit \
     --master yarn \
     --deploy-mode client \
     --driver-memory 20g \
     --num-executors 6 \
     --conf spark.driver.extraJavaOptions="-Xms20g -Xmn2g -Dlog4j.configuration=file:/home/mboehm/perftest/log4j.properties " \
     --conf spark.ui.showConsoleProgress=true \
     --conf spark.executor.heartbeatInterval=100s \
     --conf spark.network.timeout=512s \
     --executor-memory 200g \
     --executor-cores 48 \
      SystemDS.jar "$@" 
