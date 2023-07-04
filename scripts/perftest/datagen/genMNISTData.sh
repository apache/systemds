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

CMD=$1
DATADIR=$2/mnist
MAXMEM=$3

FORMAT="csv" # can be csv, mm, text, binary

echo "-- Generating MNIST data." >>results/times.txt
#make sure whole MNIST is available
../datagen/getMNISTDataset.sh ${DATADIR}

mnist_train_filename="mnist_train.csv"
mnist_test_filename="mnist_test.csv"

max_size_ordinal=4
min_num_examples_train=12000
max_num_examples_train=60000
span_num_examples_train=$(echo "${max_num_examples_train} - ${min_num_examples_train}" | bc)
min_num_examples_test=2000
max_num_examples_test=10000
span_num_examples_test=$(echo "${max_num_examples_test} - ${min_num_examples_test}" | bc)
#generate XS scenarios (80MB) by producing a subset of MNIST
if [ $MAXMEM -ge 80 ]; then
  echo "doing size one"
  size_ordinal=0
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  # these python calls are here to show what the equivalent computations for the target_num variables do .. only difference is that printf $0.f doesnt round the float value down like floor but just truncates it to produce an integer value
  # target_num_train=$(python -c "from math import floor; print(${min_num_examples_train} + floor(${span_num_examples_train} * ${percent_size}))")
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  # target_num_test=$(python -c "from math import floor; print(${min_num_examples_test} + floor(${span_num_examples_test} * ${percent_size}))")
  target_num_test=$(echo "${min_num_examples_test} + $(printf "%.0f" "$(echo "${span_num_examples_test} * ${percent_size}" | bc)")" | bc)
  echo $size_ordinal $percent_size $target_num_train $target_num_test
  ${CMD} -f ../datagen/extractMNISTData.dml --nvargs \
    mnist_train=${DATADIR}/${mnist_train_filename} \
    mnist_test=${DATADIR}/${mnist_test_filename} \
    out_train=${DATADIR}/mnist_train_${target_num_train} \
    out_test=${DATADIR}/mnist_test_${target_num_test} \
    num_train=${target_num_train} \
    num_test=${target_num_test} \
    fmt=${FORMAT} &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  echo "doing size two"
  size_ordinal=1
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  target_num_test=$(echo "${min_num_examples_test} + $(printf "%.0f" "$(echo "${span_num_examples_test} * ${percent_size}" | bc)")" | bc)
  echo $size_ordinal $percent_size $target_num_train $target_num_test
  ${CMD} -f ../datagen/extractMNISTData.dml --nvargs \
    mnist_train=${DATADIR}/${mnist_train_filename} \
    mnist_test=${DATADIR}/${mnist_test_filename} \
    out_train=${DATADIR}/mnist_train_${target_num_train} \
    out_test=${DATADIR}/mnist_test_${target_num_test} \
    num_train=${target_num_train} \
    num_test=${target_num_test} \
    fmt=${FORMAT} &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  size_ordinal=2
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  target_num_test=$(echo "${min_num_examples_test} + $(printf "%.0f" "$(echo "${span_num_examples_test} * ${percent_size}" | bc)")" | bc)
  ${CMD} -f ../datagen/extractMNISTData.dml --nvargs \
    mnist_train=${DATADIR}/${mnist_train_filename} \
    mnist_test=${DATADIR}/${mnist_test_filename} \
    out_train=${DATADIR}/mnist_train_${target_num_train} \
    out_test=${DATADIR}/mnist_test_${target_num_test} \
    num_train=${target_num_train} \
    num_test=${target_num_test} \
    fmt=${FORMAT} &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  size_ordinal=3
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  target_num_test=$(echo "${min_num_examples_test} + $(printf "%.0f" "$(echo "${span_num_examples_test} * ${percent_size}" | bc)")" | bc)
  ${CMD} -f ../datagen/extractMNISTData.dml --nvargs \
    mnist_train=${DATADIR}/${mnist_train_filename} \
    mnist_test=${DATADIR}/${mnist_test_filename} \
    out_train=${DATADIR}/mnist_train_${target_num_train} \
    out_test=${DATADIR}/mnist_test_${target_num_test} \
    num_train=${target_num_train} \
    num_test=${target_num_test} \
    fmt=${FORMAT} &
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  size_ordinal=4
  percent_size=$(echo "scale=10; ${size_ordinal} / ${max_size_ordinal}" | bc)
  target_num_train=$(echo "${min_num_examples_train} + $(printf "%.0f" "$(echo "${span_num_examples_train} * ${percent_size}" | bc)")" | bc)
  target_num_test=$(echo "${min_num_examples_test} + $(printf "%.0f" "$(echo "${span_num_examples_test} * ${percent_size}" | bc)")" | bc)
  ${CMD} -f ../datagen/extractMNISTData.dml --nvargs \
    mnist_train=${DATADIR}/${mnist_train_filename} \
    mnist_test=${DATADIR}/${mnist_test_filename} \
    out_train=${DATADIR}/mnist_train_${target_num_train} \
    out_test=${DATADIR}/mnist_test_${target_num_test} \
    num_train=${target_num_train} \
    num_test=${target_num_test} \
    fmt=${FORMAT} &
fi

wait
