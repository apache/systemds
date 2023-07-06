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

# this sets the dot as the separating character in floating point numbers ie. their string representation
# this avoids an error where bc outputs results dot-separated but printf may expect floats comma-separated if the system default says so
export LC_NUMERIC="en_US.UTF-8"

COMMAND=$1
TEMPFOLDER=$2
MAXMEM=$3
USEGPU=$4

if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp; fi
BASE=${TEMPFOLDER}/mnist
MAXITR=200

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

max_size_ordinal=4 # these should be kept in sync with the ones set in genMNISTData, so that file names are in sync!
min_num_examples_train=12000
max_num_examples_train=60000
span_num_examples_train=$(echo "${max_num_examples_train} - ${min_num_examples_train}" | bc)
DATA=()
if [ $MAXMEM -ge 80 ]; then
  size_ordinal=0
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  DATA+=(mnist_${target_num_train})
fi
if [ $MAXMEM -ge 800 ]; then
  size_ordinal=1
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  DATA+=(mnist_${target_num_train})
fi
if [ $MAXMEM -ge 8000 ]; then
  size_ordinal=2
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  DATA+=(mnist_${target_num_train})
fi
if [ $MAXMEM -ge 80000 ]; then
  size_ordinal=3
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  DATA+=(mnist_${target_num_train})
fi
if [ $MAXMEM -ge 800000 ]; then
  size_ordinal=4
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  DATA+=(mnist_${target_num_train})
fi

echo "RUN CONV2D EXPERIMENTS" $(date) >>results/times.txt

for d in ${DATA[@]}; do #"_KDD"
  for f in "runMNISTLeNet"; do
    echo "-- Running "$f" on "$d" for 10 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/${d}_train ${BASE}/${d}_test ${BASE} "${COMMAND}" ${d} 10 ${USEGPU} &>logs/${f}_${d}_10.out
    echo "-- Running "$f" on "$d" for 100 epochs" >>results/times.txt
    ./${f}.sh ${BASE}/${d}_train ${BASE}/${d}_test ${BASE} "${COMMAND}" ${d} 100 ${USEGPU} &>logs/${f}_${d}_100.out
  done
done

echo -e "\n\n" >>results/times.txt
