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

generateRandData () {
  TARGET=${1:-"${DATADIR}/X"}
  # generate random data
  ${CMD} -f ${BASEPATH}/data/genRandData.dml \
  --nvargs rows=$ROWS cols=$COLS min=$MIN max=$MAX target=$TARGET
}

startLocalWorkers () {
  # start the federated workers on localhost
  ${BASEPATH}/utils/startFedWorkers.sh $CMD $DATADIR $NUMFED "localhost";
}

createFedObject () {
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

SharedWorkers.createSharedFedObjects () {
  DATA=${1:-"${DATADIR}/X"}
  TARGET_PREFIX=${2:-"${DATA}_fed_"}

  # minimal number of shared workers between worker n and worker n+1
  nSharedWorkers=`echo "scale=0; (($NUMCOORD * $NUMSPLIT) - $NUMFED + ($NUMCOORD - 2)) / ($NUMCOORD - 1)" | bc`
  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    hostOffset=`echo "scale=0; ($counter - 1) * ($NUMSPLIT - $nSharedWorkers)" | bc`
    createFedObject $DATA ${TARGET_PREFIX}${counter}.json FALSE $hostOffset
  done
}

startCoordinator () {
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

SameWorkers.compute () {
  SCRIPT=$1
  FED_DATA=${2:-"${DATADIR}/X_fed.json"}
  TARGET_PREFIX=${3:-"${DATADIR}/Z"}

  coord_dir=${DATADIR}/coordinators
  if [ ! -d $coord_dir ]; then mkdir -p $coord_dir ; fi
  
  pids=
  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    startCoordinator $SCRIPT $FED_DATA $counter ${TARGET_PREFIX} $coord_dir
    echo "$(date +%H:%M:%S:%N) - Started coordinator process ${counter}/${NUMCOORD}"
  done
  wait $pids
  echo "$(date +%H:%M:%S:%N) - All processes finished"
}

SharedWorkers.compute () {
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

killWorkers () {
  ${BASEPATH}/utils/killFedWorkers.sh $DATADIR;
}

