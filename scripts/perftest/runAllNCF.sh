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

if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp; fi
BASE=${TEMPFOLDER}/ncf
MAXITR=200

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

DATA=() # todo .. which data is needed?
if [ $MAXMEM -ge 80 ]; then DATA+=("1000_100_50_60"); fi
if [ $MAXMEM -ge 800 ]; then DATA+=("3000_300_150_180"); fi
if [ $MAXMEM -ge 8000 ]; then DATA+=("9000_900_450_540"); fi
if [ $MAXMEM -ge 80000 ]; then DATA+=("27000_2700_1350_1620"); fi
if [ $MAXMEM -ge 800000 ]; then DATA+=("81000_8100_4050_4860"); fi

echo "RUN NEURAL COLLABORATIVE FILTERING EXPERIMENTS" $(date) >>results/times.txt

for d in ${DATA[@]}; do #"_KDD"
  for f in "runNCF"; do
    echo "-- Running "$f" on "$d" for 5 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/Tt${d} ${BASE}/Tv${d} ${BASE}/It${d} ${BASE}/Iv${d} ${BASE}/Ut${d} ${BASE}/Uv${d} \
      ${BASE} "${COMMAND}" ${d} 5 &>logs/${f}_${d}_5.out
    echo "-- Running "$f" on "$d" for 50 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/Tt${d} ${BASE}/Tv${d} ${BASE}/It${d} ${BASE}/Iv${d} ${BASE}/Ut${d} ${BASE}/Uv${d} \
      ${BASE} "${COMMAND}" ${d} 50 &>logs/${f}_${d}_50.out
  done
done

echo -e "\n\n" >>results/times.txt
