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
DATADIR=${2:-"temp/sameworkers"}/als
NUMFED=${3:-4}
NUMCOORD=${4:-2}

ROWS=1800
COLS=1000
DATA_RANK=10
DATA_NNZ=`echo "scale=0; 0.9 * $ROWS * $COLS" | bc`

MAXITER=20

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

generateALSData () {
  TARGET=${1:-"${DATADIR}/X"}

  ${CMD} -f ${BASEPATH}/../../datagen/genRandData4ALS.dml --nvargs X=$TARGET rows=$ROWS cols=$COLS rank=$DATA_RANK nnz=$DATA_NNZ
}

startCoordinator () {
  SCRIPT=$1
  FED_DATA=$2
  INDEX=${3:-0}
  TARGET_PREFIX=${4:-"${DATADIR}/model"}
  COORD_DIR=${5:-"${DATADIR}/coordinators"}

  ${CMD} -f ${BASEPATH}/scripts/${SCRIPT} \
    --config conf/SystemDS-config.xml \
    --stats \
    --nvargs data=$FED_DATA modelB=${TARGET_PREFIX}B${INDEX} modelM=${TARGET_PREFIX}M${INDEX} maxiter=$MAXITER &> ${COORD_DIR}/output_${INDEX}.txt &
  pids+=" $!"
}

evalResult () {
  OUTPUT_PREFIX=${1:-"${DATADIR}/model"}

  error_count=0
  for((counter=1; counter<=$NUMCOORD; counter++))
  do
    if [ ! -f ${OUTPUT_PREFIX}B${counter} ]; then
      echo_stderr "FAILURE in $0: ${OUTPUT_PREFIX}B${counter} does not exist."
      ((++error_count))
    fi
    if [ ! -f ${OUTPUT_PREFIX}M${counter} ]; then
      echo_stderr "FAILURE in $0: ${OUTPUT_PREFIX}M${counter} does not exist."
      ((++error_count))
    fi
  done
  return $error_count
}

initDataDir
generateALSData ${DATADIR}/X
startLocalWorkers
createFedObject ${DATADIR}/X ${DATADIR}/X_fed.json FALSE
SameWorkers.compute alsCG.dml ${DATADIR}/X_fed.json ${DATADIR}/model
killWorkers
exit $(evalResult ${DATADIR}/model)

