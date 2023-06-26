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

CMD=$1
DATADIR=$2/mnist
MAXMEM=$3

FORMAT="text" # can be csv, mm, text, binary

echo "-- Generating MNIST data." >> results/times.txt;
#make sure whole MNIST is available
../datagen/getMNISTDataset.sh ${DATADIR}

#generate XS scenarios (80MB) by producing a subset of MNIST
if [ $MAXMEM -ge 80 ]; then
  echo "placeholder"
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  echo "placeholder"
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  echo "placeholder"
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  echo "placeholder"
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  echo "placeholder"
fi

wait