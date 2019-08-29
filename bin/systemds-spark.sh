#!/bin/bash
#-------------------------------------------------------------
#  Modifications Copyright 2019 Graz University of Technology
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
# under the License.
#-------------------------------------------------------------

#set -x


# Environment
#-------------------------------------------------------------
if [ "$OSTYPE" == "win32" ] ||  [ "$OSTYPE" == "msys" ] ; then
  DIR_SEP=\\
  USER_DIR=`cygpath -w -p ${PWD}`
else
  DIR_SEP=/
  USER_DIR=$PWD
fi


# Following variables must be rewritten by your installation paths.
DEFAULT_SPARK_HOME=/usr/local/spark-2.1.0/spark-2.1.0-bin-hadoop2.6
DEFAULT_SYSTEMDS_ROOT=.

if [ ! -d $DEFAULT_SPARK_HOME ]; then
  echo "Unsetting DEFAULT_SPARK_HOME as it does not exist"
  unset DEFAULT_SPARK_HOME
fi

if [ -z ${SPARK_HOME} ]; then
  SPARK_HOME=${DEFAULT_SPARK_HOME}
fi

if [ -z ${SYSTEMDS_ROOT} ]; then
  SYSTEMDS_ROOT=${DEFAULT_SYSTEMDS_ROOT}
fi


# Default Values
#-------------------------------------------------------------
master="--master yarn"
deploy_mode="--deploy-mode client"
driver_memory="--driver-memory 20G"
num_executors="--num-executors 5"
executor_memory="--executor-memory 60G"
executor_cores="--executor-cores 24"
conf="--conf spark.driver.maxResultSize=0"


# error help print
#-------------------------------------------------------------
printUsageExit()
{
cat <<EOF

Usage: $0 [-h] [SPARK-SUBMIT OPTIONS] -f <dml-filename> [SYSTEMDS OPTIONS]

   Examples:
      $0 -f genGNMF.dml --nvargs V=/tmp/V.mtx W=/tmp/W.mtx H=/tmp/H.mtx rows=100000 cols=800 k=50
      $0 --driver-memory 5G -f GNMF.dml --explain hops -nvargs ...
      $0 --master yarn --deploy-mode cluster -f hdfs:/user/GNMF.dml

   -h | -?  Print this usage message and exit

   SPARK-SUBMIT OPTIONS:
   --conf <property>=<value>   Configuration settings:
                                 spark.driver.maxResultSize            Default: 0
   --driver-memory <num>       Memory for driver (e.g. 512M)           Default: 20G
   --master <string>           local | yarn                            Default: yarn
   --deploy-mode <string>      client | cluster                        Default: client
   --num-executors <num>       Number of executors to launch (e.g. 2)  Default: 5
   --executor-memory <num>     Memory per executor (e.g. 1G)           Default: 60G
   --executor-cores <num>      Number of cores per executor (e.g. 1)   Default: 24

   -f                          DML script file name, e.g. hdfs:/user/biadmin/test.dml

   SYSTEMDS OPTIONS:
   --stats                     Monitor and report caching/recompilation statistics
   --explain                   Explain plan (runtime)
   --explain2 <string>         Explain plan (hops, runtime, recompile_hops, recompile_runtime)
   --nvargs <varName>=<value>  List of attributeName-attributeValue pairs
   --args <string>             List of positional argument values
EOF
  exit 1
}

# command line parameter processing
#-------------------------------------------------------------
while true ; do
  case "$1" in
    -h)                printUsageExit ; exit 1 ;;
    --master)          master="--master "$2 ; shift 2 ;;
    --deploy-mode)     deploy_mode="--deploy-mode "$2 ; shift 2 ;;
    --driver-memory)   driver_memory="--driver-memory "$2 ; shift 2 ;;
    --num-executors)   num_executors="--num-executors "$2 ; shift 2 ;;
    --executor-memory) executor_memory="--executor-memory "$2 ; shift 2 ;;
    --executor-cores)  executor_cores="--executor-cores "$2 ; shift 2 ;;
    --conf)            conf=${conf}' --conf '$2 ; shift 2 ;;
    -f)                if [ -z "$2" ]; then echo "Error: Wrong usage. Try -h" ; exit 1 ; else f=$2 ; shift 2 ; fi ;;
    --stats)           stats="-stats" ; shift 1 ;;
    --explain)         explain="-explain" ; shift 1 ;;
    --explain2)        explain="-explain "$2 ; shift 2 ;;
    --nvargs)          shift 1 ; nvargs="-nvargs "$@ ; break ;;
    --args)            shift 1 ; args="-args "$@ ; break ;;
    *)                 if [ -z "$f" ]; then echo "Error: Wrong usage. Try -h" ; exit 1 ; else break ; fi ;;
  esac
done

# if the script file path was omitted, try to complete the script path
if [ ! -f "$f" ]
then
  SCRIPT_FILE_NAME=$(basename $f)
  SCRIPT_FILE_FOUND=$(find "$PROJECT_ROOT_DIR${DIR_SEP}scripts" -name "$SCRIPT_FILE_NAME")
  if [ ! "$SCRIPT_FILE_FOUND" ]
  then
    echo "Could not find DML script: $f"
    printSimpleUsage
  else
    f=$SCRIPT_FILE_FOUND
    echo "DML script: $f"
  fi
fi

SYSTEMDS_JAR_FILE=${SYSTEMDS_ROOT}/SystemDS.jar
if [ ! -f "$SYSTEMDS_JAR_FILE" ]; then
  JAR_FILE_NAME=$(basename $SYSTEMDS_JAR_FILE)
  JAR_FILE_FOUND=$(find "$SYSTEMDS_ROOT${DIR_SEP}target" -name "$JAR_FILE_NAME")
  if [ ! "$JAR_FILE_FOUND" ]
  then
    echo "Could not find $SYSTEMDS_JAR_FILE or $JAR_FILE_FOUND"
    printSimpleUsage
  else
    SYSTEMDS_JAR_FILE=$JAR_FILE_FOUND
    echo "DML script: $SYSTEMDS_JAR_FILE"
  fi
fi

SYSTEMDS_CONFIG_FILE=${SYSTEMDS_ROOT}/SystemDS-config.xml
if [ ! -f "$SYSTEMDS_CONFIG_FILE" ]; then
  CONFIG_FILE_NAME=$(basename $SYSTEMDS_CONFIG_FILE)
  CONFIG_FILE_FOUND=$(find "$SYSTEMDS_ROOT${DIR_SEP}conf" -name "$CONFIG_FILE_NAME")
  if [ ! "$CONFIG_FILE_FOUND" ]
  then
    echo "Could not find $SYSTEMDS_CONFIG_FILE or $CONFIG_FILE_FOUND"
    printSimpleUsage
  else
    SYSTEMDS_CONFIG_FILE=$CONFIG_FILE_FOUND
    echo "Config file: $SYSTEMDS_CONFIG_FILE"
  fi
fi

# SystemDS Spark invocation
#-------------------------------------------------------------

export SPARK_MAJOR_VERSION=2

$SPARK_HOME/bin/spark-submit \
     ${master} \
     ${deploy_mode} \
     ${driver_memory} \
     ${num_executors} \
     ${executor_memory} \
     ${executor_cores} \
     ${conf} \
     ${SYSTEMDS_JAR_FILE} \
         -f ${f} \
         -config ${SYSTEMDS_CONFIG_FILE} \
         -exec HYBRID \
         $explain \
         $stats \
         $nvargs $args
