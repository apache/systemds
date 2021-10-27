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

# Read Parameters
FILENAME=$0
CMD=${1:-"systemds"}
DATADIR=${2:-"temp"}/L2SVM
NUMFED=${3:-2}
MAXITR=${4:-100}

# Error Prints
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

# Set Properties
export SYSDS_QUIET=1
BASEPATH=$(dirname "$0")

# Generate Data
${BASEPATH}/../genL2SVMData.sh systemds $DATADIR;

# Start the Federated Workers on Localhost
${BASEPATH}/utils/startFedWorkers.sh systemds $DATADIR $NUMFED "localhost";

for d in "10k_1k_dense" "10k_1k_sparse"
do
  # Split the generated data into partitions and create a federated object
  ${CMD} -f ${BASEPATH}/data/splitAndMakeFederated.dml \
    --config ${BASEPATH}/../conf/SystemDS-config.xml \
    --nvargs data=${DATADIR}/X${d} nSplit=$NUMFED transposed=FALSE \
      target=${DATADIR}/X${d}_fed.json hosts=${DATADIR}/workers/hosts fmt="csv"

  ${CMD} -f ${BASEPATH}/data/splitAndMakeFederated.dml \
      --config ${BASEPATH}/../conf/SystemDS-config.xml \
      --nvargs data=${DATADIR}/Y${d} nSplit=$NUMFED transposed=FALSE \
        target=${DATADIR}/Y${d}_fed.json hosts=${DATADIR}/workers/hosts fmt="csv"



  for fedCompile in "" "--federatedCompilation"
  do
    runningMessage="-- Running L2SVM "$fedCompile" with federated data ("$d") on "$NUMFED" federated workers";
     echo "$runningMessage" >> results/times.txt
     echo "$runningMessage" >> results/compiletimes.txt
    # Run the L2SVM algorithm on the federated object
    # $1 X, $2 Y, $3 unknown, $4 BASE, $5 maxiter, $6 CMD, $7 RunPrediction, $8 FEDERATEDCOMPILATION
    ${BASEPATH}/../runL2SVM.sh ${DATADIR}/X${d}_fed.json ${DATADIR}/Y${d}_fed.json 2 $DATADIR ${MAXITR} systemds false $fedCompile | egrep -w 'compilation|L2SVM' | tee -a results/compiletimes.txt;
  done
done

# Kill the Federated Workers
${BASEPATH}/utils/killFedWorkers.sh $DATADIR;
