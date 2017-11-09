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

if [ "$1" == "" -o "$2" == "" ]; then echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi


BASE=$1/binomial

FORMAT="binary" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"


#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=10000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w10k_1k_dense X=$BASE/X10k_1k_dense Y=$BASE/y10k_1k_dense addnoise=1 icpt=0 sparse=$DENSE_SP fmt=$FORMAT translabel=1
${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=10000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w10k_1k_sparse X=$BASE/X10k_1k_sparse Y=$BASE/y10k_1k_sparse addnoise=1 icpt=0 sparse=$SPARSE_SP fmt=$FORMAT translabel=1
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_dense $BASE/y10k_1k_dense $BASE/X10k_1k_dense_test $BASE/y10k_1k_dense_test $FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_sparse $BASE/y10k_1k_sparse $BASE/X10k_1k_sparse_test $BASE/y10k_1k_sparse_test $FORMAT

##generate S scenarios (800MB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=100000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w100k_1k_dense X=$BASE/X100k_1k_dense Y=$BASE/y100k_1k_dense addnoise=1 icpt=0 sparse=$DENSE_SP fmt=$FORMAT translabel=1
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=100000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w100k_1k_sparse X=$BASE/X100k_1k_sparse Y=$BASE/y100k_1k_sparse addnoise=1 icpt=0 sparse=$SPARSE_SP fmt=$FORMAT translabel=1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_dense $BASE/y100k_1k_dense $BASE/X100k_1k_dense_test $BASE/y100k_1k_dense_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_sparse $BASE/y100k_1k_sparse $BASE/X100k_1k_sparse_test $BASE/y100k_1k_sparse_test $FORMAT
#
##generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=1000000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w1M_1k_dense X=$BASE/X1M_1k_dense Y=$BASE/y1M_1k_dense addnoise=1 icpt=0 sparse=$DENSE_SP fmt=$FORMAT translabel=1
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=1000000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w1M_1k_sparse X=$BASE/X1M_1k_sparse Y=$BASE/y1M_1k_sparse addnoise=1 icpt=0 sparse=$SPARSE_SP fmt=$FORMAT translabel=1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_dense $BASE/y1M_1k_dense $BASE/X1M_1k_dense_test $BASE/y1M_1k_dense_test $FORMAT
#${CMD}-f extractTestData.dml $DASH-args $BASE/X1M_1k_sparse $BASE/y1M_1k_sparse $BASE/X1M_1k_sparse_test $BASE/y1M_1k_sparse_test $FORMAT
#
##generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=10000000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w10M_1k_dense X=$BASE/X10M_1k_dense Y=$BASE/y10M_1k_dense addnoise=1 icpt=0 sparse=$DENSE_SP fmt=$FORMAT translabel=1
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=10000000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w10M_1k_sparse X=$BASE/X10M_1k_sparse Y=$BASE/y10M_1k_sparse addnoise=1 icpt=0 sparse=$SPARSE_SP fmt=$FORMAT translabel=1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_dense $BASE/y10M_1k_dense $BASE/X10M_1k_dense_test $BASE/y10M_1k_dense_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_sparse $BASE/y10M_1k_sparse $BASE/X10M_1k_sparse_test $BASE/y10M_1k_sparse_test $FORMAT
#
##generate XL scenarios (800GB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=100000000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w100M_1k_dense X=$BASE/X100M_1k_dense Y=$BASE/y100M_1k_dense addnoise=1 icpt=0 sparse=$DENSE_SP fmt=$FORMAT translabel=1
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-nvargs nr=100000000 nf=1000 maxfeatval=5 maxweight=5 weight=$BASE/w100M_1k_sparse X=$BASE/X100M_1k_sparse Y=$BASE/y100M_1k_sparse addnoise=1 icpt=0 sparse=$SPARSE_SP fmt=$FORMAT translabel=1
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_dense $BASE/y100M_1k_dense $BASE/X100M_1k_dense_test $BASE/y100M_1k_dense_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_sparse $BASE/y100M_1k_sparse $BASE/X100M_1k_sparse_test $BASE/y100M_1k_sparse_test $FORMAT
#
###generate KDD scenario (csv would be infeasible)
##${CMD} -f changeFormat.dml $DASH-args mboehm/data/rdata_kdd2010/X mboehm/data/rdata_kdd2010/y 1 $BASE/X_KDD $BASE/y_KDD "text"
##${CMD} -f extractTestData.dml $DASH-args $BASE/X_KDD $BASE/y_KDD $BASE/X_KDD_test $BASE/y_KDD_test "text"
##${CMD} -f changeFormat.dml $DASH-args /user/biadmin/statiko/rdata_kdd2010/X /user/biadmin/statiko/rdata_kdd2010/y 150 $BASE/X_KDD $BASE/y_KDD "text"
##${CMD} -f extractTestData.dml $DASH-args $BASE/X_KDD $BASE/y_KDD $BASE/X_KDD_test $BASE/y_KDD_test "text"

