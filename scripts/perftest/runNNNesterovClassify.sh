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
# params:
# 1) X data
# 2) Y data
# 3) path of base temp dir
# 4) command for systemds
set -e

if [ "$(basename $PWD)" != "perftest" ]; then
  echo "Please execute scripts from directory 'perftest'"
  exit 1
fi

X=$1
Y=$2
BASE=$3
CMD=$4
LOGIDENTIFIER=$5
EPOCHS=$6
USEGPU=$7

FLAGS="--stats"
if [ "$USEGPU" = true ]; then
  FLAGS="${FLAGS} --gpu"
fi

echo "running sgd nn classifier with nesterov momentum"

#training
tstart=$(date +%s.%N)
${CMD} -f scripts/nnNesterovClassify-train.dml \
  --config conf/SystemDS-config.xml \
  ${FLAGS} \
  --nvargs X=${X} Y=${Y} B=${BASE} fmt="csv" epochs=${EPOCHS} &>logs/nnNesterovClassify-train_${LOGIDENTIFIER}_${EPOCHS}.out

ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "nesterov momentum neural network trained with SGD on "$5": "$ttrain >>results/times.txt

#predict
tstart=$(date +%s.%N)
${CMD} -f scripts/nnNesterovClassify-predict.dml \
  --config conf/SystemDS-config.xml \
  ${FLAGS} \
  --nvargs X=${X} Y=${Y} B=${BASE} fmt="csv" &>logs/nnNesterovClassify-predict_${LOGIDENTIFIER}_${EPOCHS}.out
  #--nvargs fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv &>logs/nnNesterovClassify-predict_${LOGIDENTIFIER}.out

tpredict=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "nesterov momentum neural network trained with SGD predicted on "$5": "$tpredict >>results/times.txt
