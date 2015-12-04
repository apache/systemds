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

# Environment

SPARK_HOME=/home/biadmin/spark-1.4.0/spark-1.4.0-SNAPSHOT-bin-hadoop2.4
SYSTEMML_HOME="."

# Default Values

master=yarn-client
driver_memory=20G
num_executors=5
executor_memory=60G
executor_cores=24
conf="--conf spark.driver.maxResultSize=0 --conf spark.akka.frameSize=128"

# error help print

printUsageExit()
{
cat <<EOF

Usage: $0 [-h] [SPARK-SUBMIT OPTIONS] -f <dml-filename> [SYSTEMML OPTIONS]

   Examples:
      $0 -f genGNMF.dml --nvargs V=/tmp/V.mtx W=/tmp/W.mtx H=/tmp/H.mtx rows=100000 cols=800 k=50
      $0 --driver-memory 5G -f GNMF.dml --explain hops -nvargs ...
      $0 --master yarn-cluster -f hdfs:/user/GNMF.dml

   -h | -?  Print this usage message and exit

   SPARK-SUBMIT OPTIONS:
   --conf <property>=<value>   Configuration settings:                  
                                 spark.driver.maxResultSize            Default: 0 
                                 spark.akka.frameSize                  Default: 128
   --driver-memory <num>       Memory for driver (e.g. 512M)]          Default: 20G
   --master <string>           local | yarn-client | yarn-cluster]     Default: yarn-client
   --num-executors <num>       Number of executors to launch (e.g. 2)  Default: 5
   --executor-memory <num>     Memory per executor (e.g. 1G)           Default: 60G
   --executor-cores <num>      Memory per executor (e.g. )             Default: 24

   -f                          DML script file name, e.g. hdfs:/user/biadmin/test.dml

   SYSTEMML OPTIONS:
   --stats                     Monitor and report caching/recompilation statistics
   --explain                   Explain plan (runtime)
   --explain2 <string>         Explain plan (hops, runtime, recompile_hops, recompile_runtime)
   --nvargs <varName>=<value>  List of attributeName-attributeValue pairs
   --args <string>             List of positional argument values
EOF
  exit 1
}


# command line parameter processing

while true ; do
  case "$1" in
    -h)                printUsageExit ; exit 1 ;;
    --master)          master=$2 ; shift 2 ;; 
    --driver-memory)   driver_memory=$2 ; shift 2 ;; 
    --num-executors)   num_executors=$2 ; shift 2 ;;
    --executor-memory) executor_memory=$2 ; shift 2 ;;
    --executor-cores)  executor_cores=$2 ; shift 2 ;;
    --conf)            conf=${conf}' --conf '$2 ; shift 2 ;;
     -f)               f=$2 ; shift 2 ;;
    --stats)           stats="-stats" ; shift 1 ;;
    --explain)         explain="-explain" ; shift 1 ;;
    --explain2)        explain="-explain "$2 ; shift 2 ;;  
    --nvargs)          shift 1 ; nvargs="-nvargs "$@ ; break ;;
    --args)            shift 1 ; args="-args "$@ ; break ;; 
    *)                 echo "Error: Wrong usage. Try -h" ; exit 1 ;;
  esac
done

# SystemML Spark invocation

$SPARK_HOME/bin/spark-submit \
     --master ${master} \
     --driver-memory ${driver_memory} \
     --num-executors ${num_executors} \
     --executor-memory ${executor_memory} \
     --executor-cores ${executor_cores} \
     ${conf} \
     $SYSTEMML_HOME/SystemML.jar \
         -f ${f} \
         -config=$SYSTEMML_HOME/SystemML-config.xml \
         -exec hybrid_spark \
         $explain \
         $stats \
         $nvargs $args
