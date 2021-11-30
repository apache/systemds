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

CMD=${1:-"systemds"}
DATADIR=${2:-"temp"}/als
MAXMEM=${3:-80}
NUMFED=${4:-4}
MAXITR=${5:-100}

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

# Set properties
export SYSDS_QUIET=1

BASEPATH=$(dirname "$0")
TEMPFILENAME=$(basename -- "$FILENAME")
BASEFILENAME=${TEMPFILENAME%.*}

${BASEPATH}/genALS_FedData.sh $CMD $DATADIR $MAXMEM &> ${BASEPATH}/../logs/genALS_FedData.out; # generate the data

DATA=()
if [ $MAXMEM -ge 80 ]; then DATA+=("10k_1k_dense" "10k_1k_sparse"); fi
if [ $MAXMEM -ge 800 ]; then DATA+=("100k_1k_dense" "100k_1k_sparse"); fi
if [ $MAXMEM -ge 8000 ]; then DATA+=("1M_1k_dense" "1M_1k_sparse"); fi
if [ $MAXMEM -ge 80000 ]; then DATA+=("10M_1k_dense" "10M_1k_sparse"); fi
if [ $MAXMEM -ge 800000 ]; then DATA+=("100M_1k_dense" "100M_1k_sparse"); fi

# start the federated workers on localhost
date &> ${BASEPATH}/../logs/runAllFed.out
${BASEPATH}/utils/startFedWorkers.sh $CMD $DATADIR $NUMFED "localhost" &>> ${BASEPATH}/../logs/runAllFed.out;

echo "test 1"

for d in ${DATA[@]}
do
  # split the generated data into partitions and create a federated object
  ${CMD} -f ${BASEPATH}/data/splitAndMakeFederated.dml \
    --config ${BASEPATH}/../conf/SystemDS-config.xml \
    --nvargs data=${DATADIR}/X${d} nSplit=$NUMFED transposed=FALSE \
      target=${DATADIR}/X${d}_fed.json hosts=${DATADIR}/workers/hosts fmt="csv" \
      &> ${BASEPATH}/../logs/${BASEFILENAME}_${d}.out;

  echo "-- Running ALS-CG with federated data ("$d") on "$NUMFED" federated workers" >> results/times.txt

  # run the als algorithm on the federated object
  ${BASEPATH}/runALS_CG_Fed.sh ${DATADIR}/X${d}_fed.json $MAXITR $DATADIR $CMD 0.001 FALSE &>> ${BASEPATH}/../logs/${BASEFILENAME}_${d}.out;
done

echo "test 2"

${BASEPATH}/utils/killFedWorkers.sh $DATADIR &>> ${BASEPATH}/../logs/runAllFed.out; # kill the federated workers

echo "test 3"