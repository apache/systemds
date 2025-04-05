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
BASE=$2/stratstats
MAXMEM=$3

FORMAT="binary"

echo "-- Generating stats data..." >> results/times.txt;

#XS data 10K rows
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f datagen/genRandData4StratStats.dml --explain --stats --nvargs nr=10000 nf=100 D=${BASE}/A_10k/data Xcid=${BASE}/A_10k/Xcid Ycid=${BASE}/A_10k/Ycid A=${BASE}/A_10k/A fmt=$FORMAT &
fi

#S data 100K rows
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f datagen/genRandData4StratStats.dml --explain --stats --nvargs nr=100000 nf=100 D=${BASE}/A_100k/data Xcid=${BASE}/A_100k/Xcid Ycid=${BASE}/A_100k/Ycid A=${BASE}/A_100k/A fmt=$FORMAT &
fi

#M data 1M rows
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f datagen/genRandData4StratStats.dml --explain --stats --nvargs nr=1000000 nf=100 D=${BASE}/A_1M/data Xcid=${BASE}/A_1M/Xcid Ycid=${BASE}/A_1M/Ycid A=${BASE}/A_1M/A fmt=$FORMAT &
fi

#L data 10M rows
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f datagen/genRandData4StratStats.dml --explain --stats --nvargs nr=10000000 nf=100 D=${BASE}/A_10M/data Xcid=${BASE}/A_10M/Xcid Ycid=${BASE}/A_10M/Ycid A=${BASE}/A_10M/A fmt=$FORMAT
fi

#XL data 100M rows
if [ $MAXMEM -ge 800000 ]; then
  ${CMD} -f datagen/genRandData4StratStats.dml --explain --stats --nvargs nr=100000000 nf=100 D=${BASE}/A_10M/data Xcid=${BASE}/A_10M/Xcid Ycid=${BASE}/A_10M/Ycid A=${BASE}/A_10M/A fmt=$FORMAT
fi

wait