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

#training
tstart=$(date +%s.%N)
#systemds -f scripts/algorithms/naive-bayes.dml \
systemds -f ${PERFTESTPATH}/scripts/naive-bayes.dml \
   -config ${PERFTESTPATH}/conf/SystemDS-config.xml \
   -stats \
   -nvargs X=$1 Y=$2 prior=${BASE}/prior conditionals=${BASE}/conditionals fmt="csv"

ttrain=$(echo "$(date +%s.%N) - $tstart" | bc)
echo "NaiveBayes train on "$1": "$ttrain >> ${PERFTESTPATH}/results/times.txt

#predict
tstart=$(date +%s.%N)
#systemds -f scripts/algorithms/naive-bayes-predict.dml \
systemds -f ${PERFTESTPATH}/scripts/naive-bayes-predict.dml \
   -config ${PERFTESTPATH}/conf/SystemDS-config.xml \
   -stats \
   -nvargs X=$1_test Y=$2_test prior=${BASE}/prior conditionals=${BASE}/conditionals fmt="csv" probabilities=${BASE}/probabilities #accuracy=${BASE}/accuracy confusion=${BASE}/confusion

tpredict=$(echo "$(date +%s.%N) - $tstart" | bc)
echo "NaiveBayes predict on "$1": "$tpredict >> ${PERFTESTPATH}/results/times.txt
