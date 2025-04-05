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
DATADIR=$2/als
MAXMEM=$3

FORMAT="text" # can be csv, mm, text, binary
DENSE_SP=0.9
SPARSE_SP=0.01

echo "-- Generating ALS data." >> results/times.txt;

#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X10k_1k_dense rows=10000 cols=1000 rank=10 nnz=`echo "scale=0; 10000 * 1000 * $DENSE_SP" | bc` sigma=0.01 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X10k_1k_sparse rows=10000 cols=1000 rank=10 nnz=`echo "scale=0; 10000 * 1000 * $SPARSE_SP" | bc` sigma=0.01 fmt=$FORMAT &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X100k_1k_dense rows=100000 cols=1000 rank=10 nnz=`echo "scale=0; 100000 * 1000 * $DENSE_SP" | bc` sigma=0.01 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X100k_1k_sparse rows=100000 cols=1000 rank=10 nnz=`echo "scale=0; 100000 * 1000 * $SPARSE_SP" | bc` sigma=0.01 fmt=$FORMAT &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X1M_1k_dense rows=1000000 cols=1000 rank=10 nnz=`echo "scale=0; 1000000 * 1000 * $DENSE_SP" | bc` sigma=0.01 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X1M_1k_sparse rows=1000000 cols=1000 rank=10 nnz=`echo "scale=0; 1000000 * 1000 * $SPARSE_SP" | bc` sigma=0.01 fmt=$FORMAT &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X10M_1k_dense rows=10000000 cols=1000 rank=10 nnz=`echo "scale=0; 10000000 * 1000 * $DENSE_SP" | bc` sigma=0.01 fmt=$FORMAT
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X10M_1k_sparse rows=10000000 cols=1000 rank=10 nnz=`echo "scale=0; 10000000 * 1000 * $SPARSE_SP" | bc` sigma=0.01 fmt=$FORMAT
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X100M_1k_dense rows=100000000 cols=1000 rank=10 nnz=`echo "scale=0; 100000000 * 1000 * $DENSE_SP" | bc` sigma=0.01 fmt=$FORMAT
  ${CMD} -f ../datagen/genRandData4ALS.dml --nvargs X=${DATADIR}/X100M_1k_sparse rows=100000000 cols=1000 rank=10 nnz=`echo "scale=0; 100000000 * 1000 * $SPARSE_SP" | bc` sigma=0.01 fmt=$FORMAT
fi

wait