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
DATADIR=$2/ncf
MAXMEM=$3

FORMAT="csv" # can be csv, mm, text, binary

BASE_ktrain=1000
BASE_kval=100
BASE_nitems=50
BASE_nusers=60
echo "-- Generating NCF data." >> results/times.txt;
#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  MULTIPLIER=1
  KTRAIN=$(echo "$BASE_ktrain * $MULTIPLIER" | bc)
  KVAL=$(echo "$BASE_kval * $MULTIPLIER" | bc)
  NITEMS=$(echo "$BASE_nitems * $MULTIPLIER" | bc)
  NUSERS=$(echo "$BASE_nusers * $MULTIPLIER" | bc)
  ${CMD} -f ../datagen/genRandData4NCF.dml --nvargs \
  users_train=${DATADIR}/Ut${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_train=${DATADIR}/It${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_train=${DATADIR}/Tt${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  users_val=${DATADIR}/Uv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_val=${DATADIR}/Iv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_val=${DATADIR}/Tv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  fmt="csv" \
  ktrain=${KTRAIN} \
  kval=${KVAL} \
  nitems=${NITEMS} \
  nusers=${NUSERS} &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  MULTIPLIER=3
  KTRAIN=$(echo "$BASE_ktrain * $MULTIPLIER" | bc)
  KVAL=$(echo "$BASE_kval * $MULTIPLIER" | bc)
  NITEMS=$(echo "$BASE_nitems * $MULTIPLIER" | bc)
  NUSERS=$(echo "$BASE_nusers * $MULTIPLIER" | bc)
  ${CMD} -f ../datagen/genRandData4NCF.dml --nvargs \
  users_train=${DATADIR}/Ut${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_train=${DATADIR}/It${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_train=${DATADIR}/Tt${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  users_val=${DATADIR}/Uv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_val=${DATADIR}/Iv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_val=${DATADIR}/Tv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  fmt="csv" \
  ktrain=${KTRAIN} \
  kval=${KVAL} \
  nitems=${NITEMS} \
  nusers=${NUSERS} &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  MULTIPLIER=9
  KTRAIN=$(echo "$BASE_ktrain * $MULTIPLIER" | bc)
  KVAL=$(echo "$BASE_kval * $MULTIPLIER" | bc)
  NITEMS=$(echo "$BASE_nitems * $MULTIPLIER" | bc)
  NUSERS=$(echo "$BASE_nusers * $MULTIPLIER" | bc)
  ${CMD} -f ../datagen/genRandData4NCF.dml --nvargs \
  users_train=${DATADIR}/Ut${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_train=${DATADIR}/It${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_train=${DATADIR}/Tt${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  users_val=${DATADIR}/Uv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_val=${DATADIR}/Iv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_val=${DATADIR}/Tv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  fmt="csv" \
  ktrain=${KTRAIN} \
  kval=${KVAL} \
  nitems=${NITEMS} \
  nusers=${NUSERS} &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  MULTIPLIER=27
  KTRAIN=$(echo "$BASE_ktrain * $MULTIPLIER" | bc)
  KVAL=$(echo "$BASE_kval * $MULTIPLIER" | bc)
  NITEMS=$(echo "$BASE_nitems * $MULTIPLIER" | bc)
  NUSERS=$(echo "$BASE_nusers * $MULTIPLIER" | bc)
  ${CMD} -f ../datagen/genRandData4NCF.dml --nvargs \
  users_train=${DATADIR}/Ut${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_train=${DATADIR}/It${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_train=${DATADIR}/Tt${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  users_val=${DATADIR}/Uv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_val=${DATADIR}/Iv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_val=${DATADIR}/Tv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  fmt="csv" \
  ktrain=${KTRAIN} \
  kval=${KVAL} \
  nitems=${NITEMS} \
  nusers=${NUSERS} &
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  MULTIPLIER=81
  KTRAIN=$(echo "$BASE_ktrain * $MULTIPLIER" | bc)
  KVAL=$(echo "$BASE_kval * $MULTIPLIER" | bc)
  NITEMS=$(echo "$BASE_nitems * $MULTIPLIER" | bc)
  NUSERS=$(echo "$BASE_nusers * $MULTIPLIER" | bc)
  ${CMD} -f ../datagen/genRandData4NCF.dml --nvargs \
  users_train=${DATADIR}/Ut${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_train=${DATADIR}/It${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_train=${DATADIR}/Tt${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  users_val=${DATADIR}/Uv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  items_val=${DATADIR}/Iv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  targets_val=${DATADIR}/Tv${KTRAIN}_${KVAL}_${NITEMS}_${NUSERS} \
  fmt="csv" \
  ktrain=${KTRAIN} \
  kval=${KVAL} \
  nitems=${NITEMS} \
  nusers=${NUSERS} &
fi

wait