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

COMMAND=$1
TEMPFOLDER=$2
MAXMEM=$3

if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp ; fi
BASE=${TEMPFOLDER}/binomial
MAXITR=20

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

echo "RUN REGRESSION EXPERIMENTS" $(date) >> results/times.txt;

# run all regression algorithms with binomial labels on all datasets
# see genBinomialData
for d in ${DATA[@]} #"_KDD"
do

   # -------------------------------------------------------------------------------------------------------------------
   # TODO return an additional output to preserve the internal scaling from training (for the built-in functions lmCG and lmDS).
   # The original scripts algorithms/LinearRegCG.dml and algorithms/LinearRegDS.dml do have that additional output column, but the respective built-in functions do not.
   # -------------------------------------------------------------------------------------------------------------------

   for f in "runLinearRegDS"
   do
       echo "-- Running "$f" on "$d" (all configs)" >> results/times.txt;
       ./${f}.sh ${BASE}/X${d} ${BASE}/y${d} ${BASE} ${COMMAND} &> logs/${f}_${d}.out;
   done

   # run with the parameter setting maximum of iterations
   for f in "runLinearRegCG" "runGLM_poisson_log" "runGLM_gamma_log" "runGLM_binomial_probit"
   do
      echo "-- Running "$f" on "$d" (all configs)" >> results/times.txt;
      ./${f}.sh ${BASE}/X${d} ${BASE}/y${d} ${BASE} ${MAXITR} ${COMMAND} &> logs/${f}_${d}.out;
   done
done
