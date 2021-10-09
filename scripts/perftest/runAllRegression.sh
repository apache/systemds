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

COMMAND=$1
TEMPFOLDER=$2
if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp ; fi

BASE=${TEMPFOLDER}/binomial
MAXITR=20

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

echo "RUN REGRESSION EXPERIMENTS" $(date) >> results/times.txt;

# run all regression algorithms with binomial labels on all datasets
# see genBinomialData
for d in "10k_1k_dense" "10k_1k_sparse" "100k_1k_dense" "100k_1k_sparse" "1M_1k_dense" "1M_1k_sparse" "10M_1k_dense" "10M_1k_sparse" #"_KDD" "100M_1k_dense" "100M_1k_sparse" 
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
