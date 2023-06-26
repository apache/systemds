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
if [ "$(basename $PWD)" != "perftest" ]; then
  echo "Please execute scripts from directory 'perftest'"
  exit 1
fi

COMMAND=$1
TEMPFOLDER=$2
MAXMEM=$3
USEGPU=$4

if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp; fi
BASE=${TEMPFOLDER}/nn
MAXITR=200

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

BASE_REG_SAMPLES=1024 # these should be kept in sync with the ones set in genNNData, so that file names are in sync!
BASE_REG_FEATRUES=100
BASE_CLASS_SAMPLES=1024
BASE_CLASS_FEATURES=100
BASE_CLASS_CLASSES=5

REG_DATA=()   # todo .. which data is needed?
CLASS_DATA=() # todo .. which data is needed?
if [ $MAXMEM -ge 80 ]; then
  MULTIPLIER=1
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)
  REG_DATA+=(${REG_SAMPLES}_${REG_FEATURES}_reg_dense ${REG_SAMPLES}_${REG_FEATURES}_reg_sparse)
  CLASS_DATA+=(${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense ${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse)
fi
if [ $MAXMEM -ge 800 ]; then
  MULTIPLIER=3
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)
  REG_DATA+=(${REG_SAMPLES}_${REG_FEATURES}_reg_dense ${REG_SAMPLES}_${REG_FEATURES}_reg_sparse)
  CLASS_DATA+=(${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense ${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse)
fi
if [ $MAXMEM -ge 8000 ]; then
  MULTIPLIER=9
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)
  REG_DATA+=(${REG_SAMPLES}_${REG_FEATURES}_reg_dense ${REG_SAMPLES}_${REG_FEATURES}_reg_sparse)
  CLASS_DATA+=(${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense ${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse)
fi
if [ $MAXMEM -ge 80000 ]; then
  MULTIPLIER=27
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)
  REG_DATA+=(${REG_SAMPLES}_${REG_FEATURES}_reg_dense ${REG_SAMPLES}_${REG_FEATURES}_reg_sparse)
  CLASS_DATA+=(${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense ${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse)
fi
if [ $MAXMEM -ge 800000 ]; then
  MULTIPLIER=81
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)
  REG_DATA+=(${REG_SAMPLES}_${REG_FEATURES}_reg_dense ${REG_SAMPLES}_${REG_FEATURES}_reg_sparse)
  CLASS_DATA+=(${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense ${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse)
fi

echo "RUN NEURAL NETWORK EXPERIMENTS" $(date) >>results/times.txt

for d in ${REG_DATA[@]}; do #"_KDD"
  # Regression tasks
  for f in "runNNSimpleSGD"; do
    echo "-- Running "$f" on "$d" for 5 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d} ${BASE}/Y${d} ${BASE} "${COMMAND}" ${d} 5 ${USEGPU} &>logs/${f}_${d}_5.out
    echo "-- Running "$f" on "$d" for 50 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d} ${BASE}/Y${d} ${BASE} "${COMMAND}" ${d} 50 ${USEGPU} &>logs/${f}_${d}_50.out
  done
done

for d in ${CLASS_DATA[@]}; do
  # Classification tasks
  for f in "runNNNesterovClassify"; do
    echo "-- Running "$f" on "$d" for 10 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d} ${BASE}/Y${d} ${BASE} "${COMMAND}" ${d} 10 ${USEGPU} &>logs/${f}_${d}_10.out
    echo "-- Running "$f" on "$d" for 100 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d} ${BASE}/Y${d} ${BASE} "${COMMAND}" ${d} 100 ${USEGPU} &>logs/${f}_${d}_100.out
  done
done

echo -e "\n\n" >>results/times.txt
