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

if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

BASE=$1/clustering

echo $2" RUN CLUSTERING EXPERIMENTS: " $(date) >> times.txt;

if [ ! -d logs ]; then mkdir logs ; fi

# data generation
echo "-- Using cluster data." >> times.txt;
./genClusteringData.sh $1 $2 &>> logs/genClusteringData.out

# run all clustering algorithms on all datasets
MAXITR=20
for d in "10k_1k_dense" #"100k_1k_dense" "1M_1k_dense" #"10M_1k_dense" #"100M_1k_dense"
do 
   echo "-- Running Kmeans on "$d >> times.txt;
   ./runKmeans.sh ${BASE}/X${d} ${MAXITR} ${BASE} $2 &> logs/runKmeans_${d}.out;

done
