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

TTrain=$1
TVal=$2
ITrain=$3
IVal=$4
UTrain=$5
UVal=$6
BASE=$7
CMD=$8
LOGIDENTIFIER=$9
EPOCHS=${10}
USEGPU=${11}

FLAGS="--stats"
if [ "$USEGPU" = true ]; then
  FLAGS="${FLAGS} --gpu"
fi



echo "running NCF"
echo \
${CMD} -f scripts/NCF-train.dml \
  --config conf/SystemDS-config.xml \
  ${FLAGS} \
  --nvargs B=${BASE} fmt="csv" \
    targets_train=${TTrain} \
    targets_val=${TVal} \
    items_train=${ITrain} \
    items_val=${IVal} \
    users_train=${UTrain} \
    users_val=${UVal} \
    epochs=${EPOCHS}
#training
tstart=$(date +%s.%N)
${CMD} -f scripts/NCF-train.dml \
  --config conf/SystemDS-config.xml \
  ${FLAGS} \
  --nvargs B=${BASE} fmt="csv" \
    targets_train=${TTrain} \
    targets_val=${TVal} \
    items_train=${ITrain} \
    items_val=${IVal} \
    users_train=${UTrain} \
    users_val=${UVal} \
    epochs=${EPOCHS} \
    &>logs/NCF-train_${LOGIDENTIFIER}_${EPOCHS}.out

ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "NCF trained on "$9": "$ttrain >>results/times.txt

#predict
tstart=$(date +%s.%N)
${CMD} -f scripts/NCF-predict.dml \
  --config conf/SystemDS-config.xml \
  ${FLAGS} \
  --nvargs B=${BASE} fmt="csv" epochs=${EPOCHS} \
    items=${ITrain} \
    users=${UTrain} \
    target=${TTrain} \
    biases=${BASE}/ncf_biases \
    weights=${BASE}/ncf_weights \
    &>logs/NCF-predict_train_${LOGIDENTIFIER}_${EPOCHS}.out

tpredict=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "NCF predicted on training data "$9": "$tpredict >>results/times.txt

tstart=$(date +%s.%N)
${CMD} -f scripts/NCF-predict.dml \
  --config conf/SystemDS-config.xml \
  ${FLAGS} \
  --nvargs B=${BASE} fmt="csv" epochs=${EPOCHS} \
    items=${IVal} \
    users=${UVal} \
    target=${TVal} \
    biases=${BASE}/ncf_biases \
    weights=${BASE}/ncf_weights \
    &>logs/NCF-predict_val_${LOGIDENTIFIER}_${EPOCHS}.out

tpredict=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "NCF predicted on validation data "$9": "$tpredict >>results/times.txt
