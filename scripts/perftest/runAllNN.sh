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
BASE=${TEMPFOLDER}/nn
MAXITR=200

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

DATA=() # todo .. which data is needed?
if [ $MAXMEM -ge 80 ]; then DATA+=("1024_100_1"); fi
if [ $MAXMEM -ge 800 ]; then DATA+=("3072_300_1" ); fi
if [ $MAXMEM -ge 8000 ]; then DATA+=("9216_900_1"); fi
if [ $MAXMEM -ge 80000 ]; then DATA+=("27648_2700_1"); fi
if [ $MAXMEM -ge 800000 ]; then DATA+=("82944_8200_1"); fi

echo "RUN NEURAL NETWORK EXPERIMENTS" $(date) >>results/times.txt

for d in ${DATA[@]}; do #"_KDD"

  # -------------------------------------------------------------------------------------------------------------------
  # TODO return an additional output to preserve the internal scaling from training (for the built-in functions lmCG and lmDS).
  # The original scripts algorithms/LinearRegCG.dml and algorithms/LinearRegDS.dml do have that additional output column, but the respective built-in functions do not.
  # -------------------------------------------------------------------------------------------------------------------

  #   for f in "runLinearRegDS"
  #   do
  #       echo "-- Running "$f" on "$d" (all configs)" >> results/times.txt;
  #       ./${f}.sh ${BASE}/X${d} ${BASE}/y${d} ${BASE} ${COMMAND} &> logs/${f}_${d}.out;
  #   done
  #
  #   # run with the parameter setting maximum of iterations
  #   for f in "runLinearRegCG" "runGLM_poisson_log" "runGLM_gamma_log" "runGLM_binomial_probit"
  #   do
  #      echo "-- Running "$f" on "$d" (all configs)" >> results/times.txt;
  #      ./${f}.sh ${BASE}/X${d} ${BASE}/y${d} ${BASE} ${MAXITR} ${COMMAND} &> logs/${f}_${d}.out;
  #   done

  # Regression tasks
  for f in "runNNSimpleSGD"; do
    echo "-- Running "$f" on "$d" for 5 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d}_reg ${BASE}/Y${d}_reg ${BASE} ${COMMAND} ${d} 5 &>logs/${f}_${d}.out
    echo "-- Running "$f" on "$d" for 50 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d}_reg ${BASE}/Y${d}_reg ${BASE} ${COMMAND} ${d} 50 &>logs/${f}_${d}.out
  done

  # Classification tasks
  for f in "runNNNesterovClassify"; do
    echo "-- Running "$f" on "$d" for 10 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d}_class ${BASE}/Y${d}_class ${BASE} ${COMMAND} ${d} 10 &>logs/${f}_${d}.out
    echo "-- Running "$f" on "$d" for 100 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/X${d}_class ${BASE}/Y${d}_class ${BASE} ${COMMAND} ${d} 100 &>logs/${f}_${d}.out
  done
done

echo -e "\n\n" >>results/times.txt
