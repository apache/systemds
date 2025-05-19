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

CMD=${1:-systemds}
BASE=${2:-"temp"}/dimensionreduction
MAXMEM=${3:-80}

FORMAT="binary"

echo "-- Generating Dimension Reduction data." >> results/times.txt;

#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f datagen/genRandData4PCA.dml --nvargs R=5000 C=2000 OUT=$BASE/pcaData5k_2k_dense FMT=$FORMAT &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f datagen/genRandData4PCA.dml --nvargs R=50000 C=2000 OUT=$BASE/pcaData50k_2k_dense FMT=$FORMAT &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f datagen/genRandData4PCA.dml --nvargs R=500000 C=2000 OUT=$BASE/pcaData500k_2k_dense FMT=$FORMAT &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f datagen/genRandData4PCA.dml --nvargs R=5000000 C=2000 OUT=$BASE/pcaData5M_2k_dense FMT=$FORMAT
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  ${CMD} -f datagen/genRandData4PCA.dml --nvargs R=50000000 C=2000 OUT=$BASE/pcaData50M_2k_dense FMT=$FORMAT
fi

wait