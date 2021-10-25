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
DATADIR=${2:-"temp/sameworkers"}/parforsum
NUMFED=${3:-4}
NUMCOORD=${4:-4}

ROWS=1000
COLS=1000
MIN=0
MAX=1
NUMITER=5

export SYSDS_QUIET=1

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

echo_stderr() {
  echo "$@" >&2
}

BASEPATH=$(dirname "$0")

source ${BASEPATH}/generalFunctions.sh

# override the startCoordinator function since we need to pass numiter as parameter
startCoordinator () {
  SCRIPT=$1
  FED_DATA=$2
  INDEX=${3:-0}
  TARGET_PREFIX=${4:-"${DATADIR}/Z"}
  COORD_DIR=${5:-"${DATADIR}/coordinators"}

  ${CMD} -f ${BASEPATH}/scripts/${SCRIPT} \
    --stats \
    --nvargs data=$FED_DATA target=${TARGET_PREFIX}${INDEX} numiter=$NUMITER &> ${COORD_DIR}/output_${INDEX}.txt &
  pids+=" $!"
}

# verify that there is a result file per iteration and coordinator
evalResult () {
  OUTPUT_PREFIX=${1:-"${DATADIR}/Z"}

  error_count=0

  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    for((itercount=1; itercount<=$NUMITER; itercount++))
    do
      if [ ! -f ${OUTPUT_PREFIX}${counter}_${itercount} ]; then
        echo_stderr "FAILURE in $0: ${OUTPUT_PREFIX}${counter}_${itercount} does not exist."
        ((++error_count))
      fi
    done
  done

  return $error_count
}

initDataDir
generateRandData ${DATADIR}/X
startLocalWorkers
createFedObject ${DATADIR}/X ${DATADIR}/X_fed.json FALSE
SameWorkers.compute parforSumAndAdd.dml ${DATADIR}/X_fed.json ${DATADIR}/Z
killWorkers
exit $(evalResult ${DATADIR}/Z) # return the number of failures as exit value
