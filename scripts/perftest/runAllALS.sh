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
if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

CMD=${1:-"systemds"}
DATADIR=${2:-"temp"}/als
MAXMEM=$3
MAXITR=${4:-100}

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

DATA=()
if [ $MAXMEM -ge 80 ]; then DATA+=("10k_1k_dense" "10k_1k_sparse"); fi
if [ $MAXMEM -ge 800 ]; then DATA+=("100k_1k_dense" "100k_1k_sparse"); fi
if [ $MAXMEM -ge 8000 ]; then DATA+=("1M_1k_dense" "1M_1k_sparse"); fi
if [ $MAXMEM -ge 80000 ]; then DATA+=("10M_1k_dense" "10M_1k_sparse"); fi
if [ $MAXMEM -ge 800000 ]; then DATA+=("100M_1k_dense" "100M_1k_sparse"); fi

echo "RUN ALS EXPERIMENTS: " $(date) >> results/times.txt;

for d in ${DATA[@]}
do
  for f in "runALS_CG" "runALS_DS"
   do
      echo "-- Running "$f" on "$d" (all configs)" >> results/times.txt;
      ./${f}.sh ${DATADIR}/X${d} $MAXITR $DATADIR ${CMD} 0.001 FALSE &> logs/${f}_${d}.out;
   done
done
