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

PERFTESTPATH=scripts/perftest

BASE=$4

#for all intercept values
for i in 0 1; do
   #training
   tstart=$(date +%s.%N)

   # /algorithms/l2-svm.dml already calls a built-in function for the l2 svm.
   systemds -f scripts/algorithms/l2-svm.dml \
      -config ${PERFTESTPATH}/conf/SystemDS-config.xml \
      -stats \
      -nvargs X=$1 Y=$2 icpt=$i tol=0.0001 reg=0.01 maxiter=$5 model=${BASE}/b fmt="csv"

   ttrain=$(echo "$(date +%s.%N) - $tstart" | bc)
   echo "L2SVM train ict="$i" on "$1": "$ttrain >> ${PERFTESTPATH}/results/times.txt

   #predict
   tstart=$(date +%s.%N)
   #systemds -f scripts/algorithms/l2-svm-predict.dml \
   systemds -f ${PERFTESTPATH}/scripts/l2-svm-predict.dml \
      -config ${PERFTESTPATH}/conf/SystemDS-config.xml \
      -stats \
      -nvargs X=$1_test Y=$2_test icpt=$i model=${BASE}/b fmt="csv" scores=${BASE}/scores

   tpredict=$(echo "$(date +%s.%N) - $tstart" | bc)
   echo "L2SVM predict ict="$i" on "$1": "$tpredict >> ${PERFTESTPATH}/results/times.txt
done
