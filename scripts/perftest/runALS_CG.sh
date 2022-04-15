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
set -e

if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

X=$1
MAXITER=${2:-100}
DATADIR=${3:-"temp"}
CMD=${4:-"systemds"}
THRESHOLD=${5:-0.0001}
VERBOSE=${6:-FALSE}

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

BASEPATH=$(dirname "$0")

tstart=$(date +%s.%N)

${CMD} -f ${BASEPATH}/scripts/alsCG.dml \
  --config ${BASEPATH}/conf/SystemDS-config.xml \
  --stats \
  --nvargs X=$X rank=15 reg="L2" lambda=0.000001 maxiter=$MAXITER thr=$THRESHOLD verbose=$VERBOSE modelU=${DATADIR}/U modelV=${DATADIR}/V fmt="csv"

ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "ALS-CG algorithm on "$X": "$ttrain >> results/times.txt


tstart=$(date +%s.%N)

${CMD} -f ./scripts/als-predict.dml \
  --config ${BASEPATH}/conf/SystemDS-config.xml \
  --stats \
  --nvargs X=$X Y=${DATADIR}/Y L=${DATADIR}/U R=${DATADIR}/V fmt="csv"

tpredict=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "ALS-CG predict ict="$i" on "$1": "$tpredict >> results/times.txt

