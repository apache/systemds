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

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

echo_stderr() {
  echo "$@" >&2
}

initDataDir () {
  if [ -d "$DATADIR" ]; then
    rm -r $DATADIR
  fi
}

# generate random matrix with the specified parameters
generateRandData () {
  ROWS=${ROWS:-10000}
  COLS=${COLS:-1000}
  MIN=${MIN:-0}
  MAX=${MAX:-1}

  TARGET=${1:-"${DATADIR}/X"}

  # generate random data
  ${CMD} -f ${BASEPATH}/data/genRandData.dml \
  --nvargs rows=$ROWS cols=$COLS min=$MIN max=$MAX target=$TARGET
}

# start the specified number of federated workers on localhost on free ports
# and write the worker connection info into a dml list
startLocalWorkers () {
  BASEPATH=${BASEPATH:-"./"}
  CMD=${CMD:-"systemds"}
  DATADIR=${DATADIR:-"temp"}
  NUMFED=${NUMFED:-4}
  # start the federated workers on localhost
  ${BASEPATH}/utils/startFedWorkers.sh $CMD $DATADIR $NUMFED "localhost";
}

# split the data into partitions, create a federated object for it, and write
# the federated object into the target file
createFedObject () {
  BASEPATH=${BASEPATH:-"./"}
  CMD=${CMD:-"systemds"}
  DATADIR=${DATADIR:-"temp"}

  DATA=${1:-"${DATADIR}/X"}
  TARGET=${2:-"${DATA}_fed.json"}
  TRANSPOSED=${3:-FALSE}
  HOST_OFFSET=${4:-0}

  NUMSPLIT=${NUMSPLIT:-$NUMFED}

  # split the generated data into partitions and create a federated object
  ${CMD} -f ${BASEPATH}/data/splitAndMakeFederated.dml \
    --nvargs data=$DATA nSplit=$NUMSPLIT transposed=$TRANSPOSED \
      target=$TARGET hosts=${DATADIR}/workers/hosts fmt="csv" hostOffset=$HOST_OFFSET
}

# create $NUMCOORD different federated objects, each coordinator sharing the
# lowest possible number of federated workers
# for example if we have 3 coordinators (NUMCOORD), matrices consisting 4
# partitions (NUMSPLIT), and 9 federated workers, it is divided amond the
# federated workers as follows: (pX= partition X; fX= fed worker X; cX= coord X)
#      | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | f9 |
# | c1 | p1 | p2 | p3 | p4 |    |    |    |    |    |
# | c2 |    |    | p1 | p2 | p3 | p4 |    |    |    |
# | c3 |    |    |    |    | p1 | p2 | p3 | p4 |    |
SharedWorkers.createSharedFedObjects () {
  DATA=${1:-"${DATADIR}/X"}
  TARGET_PREFIX=${2:-"${DATA}_fed_"}
  NUMCOORD=${NUMCOORD:-4}
  NUMSPLIT=${NUMSPLIT:-4}
  NUMFED=${NUMFED:-4}

  # minimal number of shared workers between worker n and worker n+1
  nSharedWorkers=`echo "scale=0; (($NUMCOORD * $NUMSPLIT) - $NUMFED + ($NUMCOORD - 2)) / ($NUMCOORD - 1)" | bc`
  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    # offset of federated worker host inside the host dml list
    hostOffset=`echo "scale=0; ($counter - 1) * ($NUMSPLIT - $nSharedWorkers)" | bc`
    createFedObject $DATA ${TARGET_PREFIX}${counter}.json FALSE $hostOffset
  done
}

# start a coordinator executing the specified script as a background process
startCoordinator () {
  DATADIR=${DATADIR:-"temp"}
  BASEPATH=${BASEPATH:-"./"}
  CMD=${CMD:-"systemds"}

  SCRIPT=$1
  FED_DATA=$2
  INDEX=${3:-0}
  TARGET_PREFIX=${4:-"${DATADIR}/Z"}
  COORD_DIR=${5:-"${DATADIR}/coordinators"}

  ${CMD} -f ${BASEPATH}/scripts/${SCRIPT} \
    --config conf/SystemDS-config.xml \
    --stats \
    --nvargs data=$FED_DATA target=${TARGET_PREFIX}${INDEX} &> ${COORD_DIR}/output_${INDEX}.txt &
  pids+=" $!"
}

# start the specified number of coordinators and stores the process IDs of the
# coordinators into pids (must already exist)
SameWorkers.compute () {
  DATADIR=${DATADIR:-"temp"}
  NUMCOORD=${NUMCOORD:-4}

  SCRIPT=$1
  FED_DATA=${2:-"${DATADIR}/X_fed.json"}
  TARGET_PREFIX=${3:-"${DATADIR}/Z"}

  coord_dir=${DATADIR}/coordinators
  if [ ! -d $coord_dir ]; then mkdir -p $coord_dir ; fi

  pids= # array to collect the process IDs of the coordinator processes
  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    startCoordinator $SCRIPT $FED_DATA $counter ${TARGET_PREFIX} $coord_dir
    echo "$(date +%H:%M:%S:%N) - Started coordinator process ${counter}/${NUMCOORD}"
  done
  wait $pids
  echo "$(date +%H:%M:%S:%N) - All processes finished"
}

# start the specified number of coordinators and stores the process IDs of the
# coordinators into pids (must already exist)
SharedWorkers.compute () {
  DATADIR=${DATADIR:-"temp"}
  NUMCOORD=${NUMCOORD:-4}

  SCRIPT=$1
  FED_DATA_PREFIX=${2:-"${DATADIR}/X_fed_"}
  TARGET_PREFIX=${3:-"${DATADIR}/Z"}

  coord_dir=${DATADIR}/coordinators
  if [ ! -d $coord_dir ]; then mkdir -p $coord_dir ; fi

  pids=
  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    startCoordinator $SCRIPT ${FED_DATA_PREFIX}${counter}.json $counter ${TARGET_PREFIX} $coord_dir
    echo "$(date +%H:%M:%S:%N) - Started coordinator process ${counter}/${NUMCOORD}"
  done
  wait $pids
  echo "$(date +%H:%M:%S:%N) - All processes finished"
}

# kill the local federated worker processes
killWorkers () {
  BASEPATH=${BASEPATH:-"./"}
  DATADIR=${DATADIR:-"temp"}

  ${BASEPATH}/utils/killFedWorkers.sh $DATADIR;
}
