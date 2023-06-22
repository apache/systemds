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
DATADIR=$2/nn
MAXMEM=$3

FORMAT="csv" # can be csv, mm, text, binary

echo "-- Generating NN data." >> results/times.txt;
# the scaling of nr and nf is to just multiply them by 3 each .. since sqrt(10) is about 3 and the data size should scale by a factor of 10 ..... needs to be tested for applicability
# for now only t=1 and t=5 are generated for regression and classification respectively .. may want to add more variety
# todo make test data
# todo generated data is too small with current parameters .. X data for xs is 2mb, s is 18mb -> pump it up
#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f ../datagen/genRandData4NNRegression.dml --nvargs X=${DATADIR}/X1024_100_1_reg Y=${DATADIR}/Y1024_100_1_reg nr=1024 nf=100 nt=1 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4NNClassification.dml --nvargs X=${DATADIR}/X1024_100_1_class Y=${DATADIR}/Y1024_100_1_class nr=1024 nf=100 nt=5 fmt=$FORMAT &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f ../datagen/genRandData4NNRegression.dml --nvargs X=${DATADIR}/X3072_300_1_reg Y=${DATADIR}/Y3072_300_1_reg nr=3072 nf=300 nt=1 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4NNClassification.dml --nvargs X=${DATADIR}/X3072_300_1_class Y=${DATADIR}/Y3072_300_1_class nr=3072 nf=300 nt=5 fmt=$FORMAT &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f ../datagen/genRandData4NNRegression.dml --nvargs X=${DATADIR}/X9216_900_1_reg Y=${DATADIR}/Y9216_900_1_reg nr=9216 nf=900 nt=1 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4NNClassification.dml --nvargs X=${DATADIR}/X9216_900_1_class Y=${DATADIR}/Y9216_900_1_class nr=9216 nf=900 nt=5 fmt=$FORMAT &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f ../datagen/genRandData4NNRegression.dml --nvargs X=${DATADIR}/X27648_2700_1_reg Y=${DATADIR}/Y27648_2700_1_reg nr=27648 nf=2700 nt=1 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4NNClassification.dml --nvargs X=${DATADIR}/X27648_2700_1_class Y=${DATADIR}/Y27648_2700_1_class nr=27648 nf=2700 nt=5 fmt=$FORMAT &
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  ${CMD} -f ../datagen/genRandData4NNRegression.dml --nvargs X=${DATADIR}/X82944_8200_1_reg Y=${DATADIR}/Y82944_8200_1_reg nr=82944 nf=8200 nt=1 fmt=$FORMAT &
  ${CMD} -f ../datagen/genRandData4NNClassification.dml --nvargs X=${DATADIR}/X82944_8200_1_class Y=${DATADIR}/Y82944_8200_1_class nr=82944 nf=8200 nt=5 fmt=$FORMAT &
fi

wait