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
   # systemds -f scripts/algorithms/m-svm.dml \
   systemds -f ${PERFTESTPATH}/scripts/m-svm.dml \
      -config ${PERFTESTPATH}/conf/SystemDS-config.xml \
      -stats \
      -nvargs X=$1 Y=$2 icpt=$i classes=$3 tol=0.0001 reg=0.01 maxiter=$5 model=${BASE}/w fmt="csv"

   ttrain=$(echo "$(date +%s.%N) - $tstart" | bc)
   echo "MSVM train ict="$i" on "$1": "$ttrain >> ${PERFTESTPATH}/results/times.txt

   #predict
   tstart=$(date +%s.%N)
   systemds -f scripts/algorithms/m-svm-predict.dml \
      -config ${PERFTESTPATH}/conf/SystemDS-config.xml \
      -stats \
      -nvargs X=$1_test Y=$2_test icpt=$i model=${BASE}/w fmt="csv"

   tpredict=$(echo "$(date +%s.%N) - $tstart" | bc)
   echo "MSVM predict ict="$i" on "$1": "$tpredict >> ${PERFTESTPATH}/results/times.txt
done
