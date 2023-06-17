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
DATADIR=${2:-"temp"}/io
MAXMEM=${3:-1}

FORMAT="csv" # can be csv, mm, text, binary

echo "-- Generating IO data." >> results/times.txt;


#generate XS scenarios (10MB)
if [ $MAXMEM -ge 1 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X500_250_dense R=500 C=250 Fmt=$FORMAT &
fi

#generate XS scenarios (10MB)
if [ $MAXMEM -ge 10 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X5k_250_dense R=5000 C=250 Fmt=$FORMAT &
fi

#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X10k_1k_dense R=10000 C=1000 Fmt=$FORMAT &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X100k_1k_dense R=100000 C=1000 Fmt=$FORMAT &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X1M_1k_dense R=1000000 C=1000 Fmt=$FORMAT &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X10M_1k_dense R=10000000 C=1000 Fmt=$FORMAT &
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  ${CMD} -f ../utils/generateData.dml --nvargs Path=${DATADIR}/X100M_1k_dense R=100000000 C=1000 Fmt=$FORMAT &
fi

wait
