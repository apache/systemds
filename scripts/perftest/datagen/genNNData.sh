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

CMD=$1
DATADIR=$2/nn
MAXMEM=$3

FORMAT="csv" # can be csv, mm, text, binary

DENSE_SP=0.9
SPARSE_SP=0.01
BASE_REG_SAMPLES=1024
BASE_REG_FEATRUES=100
BASE_CLASS_SAMPLES=1024
BASE_CLASS_FEATURES=100
BASE_CLASS_CLASSES=5

# the scaling of nr and nf is to just multiply them by 3 each .. since sqrt(10) is about 3 and the data size should scale by a factor of 10 ..... needs to be tested for applicability
# for now only t=1 and t=5 are generated for regression and classification respectively .. may want to add more variety
# todo make test data
# todo generated data is too small with current parameters .. X data for xs is 2mb, s is 18mb -> pump it up
echo "-- Generating NN data." >>results/times.txt
#generate XS scenarios (80MB)
if [ $MAXMEM -ge 80 ]; then
  # set multiplier and calculate resulting parameters
  MULTIPLIER=1
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)

  ## generate regression data
  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    1 \
    0 \
    ${DENSE_SP} \
    ${FORMAT} \
    0 &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    1 \
    0 \
    ${SPARSE_SP} \
    ${FORMAT} \
    0 &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${FORMAT} &

  ## generate classification data
  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${DENSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${FORMAT} &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${SPARSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${FORMAT} &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${FORMAT} &
fi

#generate S scenarios (800MB)
if [ $MAXMEM -ge 800 ]; then
  # set multiplier and calculate resulting parameters
  MULTIPLIER=3
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)

  ## generate regression data
  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    1 \
    0 \
    ${DENSE_SP} \
    ${FORMAT} \
    0 &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    1 \
    0 \
    ${SPARSE_SP} \
    ${FORMAT} \
    0 &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${FORMAT} &

  ## generate classification data
  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${DENSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${FORMAT} &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${SPARSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${FORMAT} &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${FORMAT} &
fi

#generate M scenarios (8GB)
if [ $MAXMEM -ge 8000 ]; then
  # set multiplier and calculate resulting parameters
  MULTIPLIER=9
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)

  ## generate regression data
  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    1 \
    0 \
    ${DENSE_SP} \
    ${FORMAT} \
    0 &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    1 \
    0 \
    ${SPARSE_SP} \
    ${FORMAT} \
    0 &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${FORMAT} &

  ## generate classification data
  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${DENSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${FORMAT} &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${SPARSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${FORMAT} &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${FORMAT} &
fi

#generate L scenarios (80GB)
if [ $MAXMEM -ge 80000 ]; then
  # set multiplier and calculate resulting parameters
  MULTIPLIER=27
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)

  ## generate regression data
  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    1 \
    0 \
    ${DENSE_SP} \
    ${FORMAT} \
    0 &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    1 \
    0 \
    ${SPARSE_SP} \
    ${FORMAT} \
    0 &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${FORMAT} &

  ## generate classification data
  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${DENSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${FORMAT} &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${SPARSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${FORMAT} &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${FORMAT} &
fi

#generate XL scenarios (800GB)
if [ $MAXMEM -ge 800000 ]; then
  # set multiplier and calculate resulting parameters
  MULTIPLIER=81
  REG_SAMPLES=$(echo "$BASE_REG_SAMPLES * $MULTIPLIER" | bc)
  REG_FEATURES=$(echo "$BASE_REG_FEATRUES * $MULTIPLIER" | bc)
  CLASS_SAMPLES=$(echo "$BASE_CLASS_SAMPLES * $MULTIPLIER" | bc)
  CLASS_FEATURES=$(echo "$BASE_CLASS_FEATURES * $MULTIPLIER" | bc)
  CLASS_CLASSES=$(echo "$BASE_CLASS_CLASSES * $MULTIPLIER" | bc)

  ## generate regression data
  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    1 \
    0 \
    ${DENSE_SP} \
    ${FORMAT} \
    0 &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4LogisticRegression.dml --args \
    ${REG_SAMPLES} \
    ${REG_FEATURES} \
    5 \
    5 \
    ${DATADIR}/w${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    1 \
    0 \
    ${SPARSE_SP} \
    ${FORMAT} \
    0 &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse \
    ${DATADIR}/X${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${DATADIR}/Y${REG_SAMPLES}_${REG_FEATURES}_reg_sparse_test \
    ${FORMAT} &

  ## generate classification data
  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${DENSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${FORMAT} &
  pidDense80=$!

  ${CMD} -f ../datagen/genRandData4Multinomial.dml --args \
    ${CLASS_SAMPLES} \
    ${CLASS_FEATURES} \
    ${SPARSE_SP} \
    ${CLASS_CLASSES} \
    0 \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${FORMAT} &
  pidSparse80=$!

  wait $pidDense80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_dense_test \
    ${FORMAT} &

  wait $pidSparse80
  ${CMD} -f scripts/extractTestData.dml --args \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse \
    ${DATADIR}/X${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${DATADIR}/Y${CLASS_SAMPLES}_${CLASS_FEATURES}_${CLASS_CLASSES}_class_sparse_test \
    ${FORMAT} &
fi

wait
