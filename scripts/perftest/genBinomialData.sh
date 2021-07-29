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

TEMPFOLDER=$1
if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=perftest ; fi

PERFTESTPATH=scripts/perftest
DATAGENPATH=scripts/datagen
BASE=${TEMPFOLDER}/binomial

FORMAT="binary" # can be csv, mm, text, binary
DENSE_SP=0.9
SPARSE_SP=0.01


#generate XS scenarios (80MB)
systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 10000 1000 5 5 ${BASE}/w10k_1k_dense ${BASE}/X10k_1k_dense ${BASE}/y10k_1k_dense 1 0 $DENSE_SP $FORMAT 1
systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 10000 1000 5 5 ${BASE}/w10k_1k_sparse ${BASE}/X10k_1k_sparse ${BASE}/y10k_1k_sparse 1 0 $SPARSE_SP $FORMAT 1
systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X10k_1k_dense ${BASE}/y10k_1k_dense ${BASE}/X10k_1k_dense_test ${BASE}/y10k_1k_dense_test $FORMAT
systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X10k_1k_sparse ${BASE}/y10k_1k_sparse ${BASE}/X10k_1k_sparse_test ${BASE}/y10k_1k_sparse_test $FORMAT

##generate S scenarios (800MB)
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 100000 1000 5 5 ${BASE}/w100k_1k_dense ${BASE}/X100k_1k_dense ${BASE}/y100k_1k_dense 1 0 $DENSE_SP $FORMAT 1
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 100000 1000 5 5 ${BASE}/w100k_1k_sparse ${BASE}/X100k_1k_sparse ${BASE}/y100k_1k_sparse 1 0 $SPARSE_SP $FORMAT 1
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X100k_1k_dense ${BASE}/y100k_1k_dense ${BASE}/X100k_1k_dense_test ${BASE}/y100k_1k_dense_test $FORMAT
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X100k_1k_sparse ${BASE}/y100k_1k_sparse ${BASE}/X100k_1k_sparse_test ${BASE}/y100k_1k_sparse_test $FORMAT
#
##generate M scenarios (8GB)
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 1000000 1000 5 5 ${BASE}/w1M_1k_dense ${BASE}/X1M_1k_dense ${BASE}/y1M_1k_dense 1 0 $DENSE_SP $FORMAT 1
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 1000000 1000 5 5 ${BASE}/w1M_1k_sparse ${BASE}/X1M_1k_sparse ${BASE}/y1M_1k_sparse 1 0 $SPARSE_SP $FORMAT 1
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X1M_1k_dense ${BASE}/y1M_1k_dense ${BASE}/X1M_1k_dense_test ${BASE}/y1M_1k_dense_test $FORMAT
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X1M_1k_sparse ${BASE}/y1M_1k_sparse ${BASE}/X1M_1k_sparse_test ${BASE}/y1M_1k_sparse_test $FORMAT
#
##generate L scenarios (80GB)
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 10000000 1000 5 5 ${BASE}/w10M_1k_dense ${BASE}/X10M_1k_dense ${BASE}/y10M_1k_dense 1 0 $DENSE_SP $FORMAT 1
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 10000000 1000 5 5 ${BASE}/w10M_1k_sparse ${BASE}/X10M_1k_sparse ${BASE}/y10M_1k_sparse 1 0 $SPARSE_SP $FORMAT 1
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X10M_1k_dense ${BASE}/y10M_1k_dense ${BASE}/X10M_1k_dense_test ${BASE}/y10M_1k_dense_test $FORMAT
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X10M_1k_sparse ${BASE}/y10M_1k_sparse ${BASE}/X10M_1k_sparse_test ${BASE}/y10M_1k_sparse_test $FORMAT
#
##generate XL scenarios (800GB)
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 100000000 1000 5 5 ${BASE}/w100M_1k_dense ${BASE}/X100M_1k_dense ${BASE}/y100M_1k_dense 1 0 $DENSE_SP $FORMAT 1
#systemds ${DATAGENPATH}/genRandData4LogisticRegression.dml --args 100000000 1000 5 5 ${BASE}/w100M_1k_sparse ${BASE}/X100M_1k_sparse ${BASE}/y100M_1k_sparse 1 0 $SPARSE_SP $FORMAT 1
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X100M_1k_dense ${BASE}/y100M_1k_dense ${BASE}/X100M_1k_dense_test ${BASE}/y100M_1k_dense_test $FORMAT
#systemds ${PERFTESTPATH}/scripts/extractTestData.dml --args ${BASE}/X100M_1k_sparse ${BASE}/y100M_1k_sparse ${BASE}/X100M_1k_sparse_test ${BASE}/y100M_1k_sparse_test $FORMAT
#
###generate KDD scenario (csv would be infeasible)
##systemds ./${PERFTESTPATH}/scripts/changeFormat.dml $DASH-args mboehm/data/rdata_kdd2010/X mboehm/data/rdata_kdd2010/y 1 ${BASE}/X_KDD ${BASE}/y_KDD "text"
##systemds ./${PERFTESTPATH}/scripts/extractTestData.dml $DASH-args ${BASE}/X_KDD ${BASE}/y_KDD ${BASE}/X_KDD_test ${BASE}/y_KDD_test "text"
##systemds ./${PERFTESTPATH}/scripts/changeFormat.dml $DASH-args /user/biadmin/statiko/rdata_kdd2010/X /user/biadmin/statiko/rdata_kdd2010/y 150 ${BASE}/X_KDD ${BASE}/y_KDD "text"
##systemds ./${PERFTESTPATH}/scripts/extractTestData.dml $DASH-args ${BASE}/X_KDD ${BASE}/y_KDD ${BASE}/X_KDD_test ${BASE}/y_KDD_test "text"

