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

BASE=$1/dimensionreduction

echo $2" RUN DIMENSION REDUCTION EXPERIMENTS: " $(date) >> times.txt;

if [ ! -d logs ]; then mkdir logs ; fi

# data generation
echo "-- Using Dimension Reduction data." >> times.txt;
./genDimensionReductionData.sh $1 $2 &>> logs/genDimensionReductionData.out

# run all dimension reduction algorithms on all datasets
for d in "5k_2k_dense" #"50k_2k_dense" "500k_2k_dense" "5M_2k_dense" "50M_2k_dense"
do 
   echo "-- Running Dimension Reduction on "$d >> times.txt;
   ./runPCA.sh pcaData${d} ${BASE} $2 &> logs/runPCA_${d}.out;

done
