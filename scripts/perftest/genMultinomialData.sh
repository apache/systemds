#!/bin/bash
#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------

if [ "$1" == "" -o "$2" == "" ]; then echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi


BASE=$1/multinomial

FORMAT="binary" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000 1000 $DENSE_SP 150 0 $BASE/X10k_1k_dense_k150 $BASE/y10k_1k_dense_k150 $FORMAT 1
${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000 1000 $SPARSE_SP 150 0 $BASE/X10k_1k_sparse_k150 $BASE/y10k_1k_sparse_k150 $FORMAT 1
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_dense_k150 $BASE/y10k_1k_dense_k150 $BASE/X10k_1k_dense_k150_test $BASE/y10k_1k_dense_k150_test $FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_sparse_k150 $BASE/y10k_1k_sparse_k150 $BASE/X10k_1k_sparse_k150_test $BASE/y10k_1k_sparse_k150_test $FORMAT

##generate S scenarios (80MB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000 1000 $DENSE_SP 150 0 $BASE/X100k_1k_dense_k150 $BASE/y100k_1k_dense_k150 $FORMAT 1
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000 1000 $SPARSE_SP 150 0 $BASE/X100k_1k_sparse_k150 $BASE/y100k_1k_sparse_k150 $FORMAT 1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_dense_k150 $BASE/y100k_1k_dense_k150 $BASE/X100k_1k_dense_k150_test $BASE/y100k_1k_dense_k150_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_sparse_k150 $BASE/y100k_1k_sparse_k150 $BASE/X100k_1k_sparse_k150_test $BASE/y100k_1k_sparse_k150_test $FORMAT
#
##generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 1000000 1000 $DENSE_SP 150 0 $BASE/X1M_1k_dense_k150 $BASE/y1M_1k_dense_k150 $FORMAT 1
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 1000000 1000 $SPARSE_SP 150 0 $BASE/X1M_1k_sparse_k150 $BASE/y1M_1k_sparse_k150 $FORMAT 1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_dense_k150 $BASE/y1M_1k_dense_k150 $BASE/X1M_1k_dense_k150_test $BASE/y1M_1k_dense_k150_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_sparse_k150 $BASE/y1M_1k_sparse_k150 $BASE/X1M_1k_sparse_k150_test $BASE/y1M_1k_sparse_k150_test $FORMAT
#
##generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000000 1000 $DENSE_SP 150 0 $BASE/X10M_1k_dense_k150 $BASE/y10M_1k_dense_k150 $FORMAT 1
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000000 1000 $SPARSE_SP 150 0 $BASE/X10M_1k_sparse_k150 $BASE/y10M_1k_sparse_k150 $FORMAT 1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_dense_k150 $BASE/y10M_1k_dense_k150 $BASE/X10M_1k_dense_k150_test $BASE/y10M_1k_dense_k150_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_sparse_k150 $BASE/y10M_1k_sparse_k150 $BASE/X10M_1k_sparse_k150_test $BASE/y10M_1k_sparse_k150_test $FORMAT
#
##generate LARGE scenarios (800GB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000000 1000 $DENSE_SP 150 0 $BASE/X100M_1k_dense_k150 $BASE/y100M_1k_dense_k150 $FORMAT 1
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000000 1000 $SPARSE_SP 150 0 $BASE/X100M_1k_sparse_k150 $BASE/y100M_1k_sparse_k150 $FORMAT 1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_dense_k150 $BASE/y100M_1k_dense_k150 $BASE/X100M_1k_dense_k150_test $BASE/y100M_1k_dense_k150_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_sparse_k150 $BASE/y100M_1k_sparse_k150 $BASE/X100M_1k_sparse_k150_test $BASE/y100M_1k_sparse_k150_test $FORMAT
