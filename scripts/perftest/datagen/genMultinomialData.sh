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
BASE=$2/multinomial
MAXMEM=$3

FORMAT="binary" 
DENSE_SP=0.9
SPARSE_SP=0.01

echo "-- Generating multinomial data..." >> results/times.txt;

#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 10000 1000 $DENSE_SP 5 0 $BASE/X10k_1k_dense_k5 $BASE/y10k_1k_dense_k5 $FORMAT 1 & pidDense80=$!
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 10000 1000 $SPARSE_SP 5 0 $BASE/X10k_1k_sparse_k5 $BASE/y10k_1k_sparse_k5 $FORMAT 1 & pidSparse80=$!
  wait $pidDense80;  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X10k_1k_dense_k5 $BASE/y10k_1k_dense_k5 $BASE/X10k_1k_dense_k5_test $BASE/y10k_1k_dense_k5_test $FORMAT &
  wait $pidSparse80; ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X10k_1k_sparse_k5 $BASE/y10k_1k_sparse_k5 $BASE/X10k_1k_sparse_k5_test $BASE/y10k_1k_sparse_k5_test $FORMAT &
fi

##generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 100000 1000 $DENSE_SP 5 0 $BASE/X100k_1k_dense_k5 $BASE/y100k_1k_dense_k5 $FORMAT 1 & pidDense800=$!
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 100000 1000 $SPARSE_SP 5 0 $BASE/X100k_1k_sparse_k5 $BASE/y100k_1k_sparse_k5 $FORMAT 1 & pidSparse800=$!
  wait $pidDense800;  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X100k_1k_dense_k5 $BASE/y100k_1k_dense_k5 $BASE/X100k_1k_dense_k5_test $BASE/y100k_1k_dense_k5_test $FORMAT &
  wait $pidSparse800; ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X100k_1k_sparse_k5 $BASE/y100k_1k_sparse_k5 $BASE/X100k_1k_sparse_k5_test $BASE/y100k_1k_sparse_k5_test $FORMAT &
fi

##generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 1000000 1000 $DENSE_SP 5 0 $BASE/X1M_1k_dense_k5 $BASE/y1M_1k_dense_k5 $FORMAT 1 & pidDense8000=$!
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 1000000 1000 $SPARSE_SP 5 0 $BASE/X1M_1k_sparse_k5 $BASE/y1M_1k_sparse_k5 $FORMAT 1 & pidSparse8000=$!
  wait $pidDense8000;  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X1M_1k_dense_k5 $BASE/y1M_1k_dense_k5 $BASE/X1M_1k_dense_k5_test $BASE/y1M_1k_dense_k5_test $FORMAT &
  wait $pidSparse8000; ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X1M_1k_sparse_k5 $BASE/y1M_1k_sparse_k5 $BASE/X1M_1k_sparse_k5_test $BASE/y1M_1k_sparse_k5_test $FORMAT &
fi

##generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 10000000 1000 $DENSE_SP 5 0 $BASE/X10M_1k_dense_k5 $BASE/y10M_1k_dense_k5 $FORMAT 1
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 10000000 1000 $SPARSE_SP 5 0 $BASE/X10M_1k_sparse_k5 $BASE/y10M_1k_sparse_k5 $FORMAT 1
  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X10M_1k_dense_k5 $BASE/y10M_1k_dense_k5 $BASE/X10M_1k_dense_k5_test $BASE/y10M_1k_dense_k5_test $FORMAT
  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X10M_1k_sparse_k5 $BASE/y10M_1k_sparse_k5 $BASE/X10M_1k_sparse_k5_test $BASE/y10M_1k_sparse_k5_test $FORMAT
fi

#generate LARGE scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 100000000 1000 $DENSE_SP 5 0 $BASE/X100M_1k_dense_k5 $BASE/y100M_1k_dense_k5 $FORMAT 1
  ${CMD} -f datagen/genRandData4Multinomial.dml $DASH-args 100000000 1000 $SPARSE_SP 5 0 $BASE/X100M_1k_sparse_k5 $BASE/y100M_1k_sparse_k5 $FORMAT 1
  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X100M_1k_dense_k5 $BASE/y100M_1k_dense_k5 $BASE/X100M_1k_dense_k5_test $BASE/y100M_1k_dense_k5_test $FORMAT
  ${CMD} -f scripts/extractTestData.dml $DASH-args $BASE/X100M_1k_sparse_k5 $BASE/y100M_1k_sparse_k5 $BASE/X100M_1k_sparse_k5_test $BASE/y100M_1k_sparse_k5_test $FORMAT
fi

wait