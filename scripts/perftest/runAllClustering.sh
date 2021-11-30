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
if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

COMMAND=$1
TEMPFOLDER=$2
MAXMEM=$3

BASE=${TEMPFOLDER}/clustering
MAXITR=20

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

DATA=()
if [ $MAXMEM -ge 80 ]; then DATA+=("10k_1k_dense"); fi
if [ $MAXMEM -ge 800 ]; then DATA+=("100k_1k_dense"); fi
if [ $MAXMEM -ge 8000 ]; then DATA+=("1M_1k_dense"); fi
if [ $MAXMEM -ge 80000 ]; then DATA+=("10M_1k_dense"); fi
if [ $MAXMEM -ge 800000 ]; then DATA+=("100M_1k_dense"); fi

echo "RUN CLUSTERING EXPERIMENTS: " $(date) >> results/times.txt;

# run all clustering algorithms on all datasets
for d in ${DATA[@]}
do
   echo "-- Running Kmeans on "$d >> results/times.txt;
   ./runKmeans.sh ${BASE}/X${d} ${MAXITR} ${BASE} ${COMMAND} &> logs/runKmeans_${d}.out;
done
