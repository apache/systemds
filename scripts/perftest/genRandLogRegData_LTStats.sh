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
# ./genRandLogRegData_LTStats.sh myperftest SPARK 1 LOGISTIC &>> logs/genBinomialData.out
# ./genRandLogRegData_LTStats.sh myperftest SPARK 150 LOGISTIC &>> logs/genMultinomialData.out
# ./genRandLogRegData_LTStats.sh myperftest SPARK 1 REGRESSION &>> logs/genRegressionData.out
if [ "$1" == "" -o "$2" == "" ]; then echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi
if [ "$3" == "1" ]; then BASE=$1/binomial ; else BASE=$1/multinomial ; fi
if [ "$4" == "LOGISTIC" ]; then DATAGEN_SCRIPT=../datagen/genRandData4LogReg_LTstats.dml ; else DATAGEN_SCRIPT=../datagen/genRandData4LinearReg_LTstats.dml ; fi

NUM_CATEGORY=$3
FORMAT="binary" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#generate XS scenarios (80MB)
SUFFIX=10k_1k
${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=10000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$DENSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_dense X=${BASE}/X${SUFFIX}_dense Y=${BASE}/y${SUFFIX}_dense Xt=${BASE}/X${SUFFIX}_dense_test Yt=${BASE}/y${SUFFIX}_dense_test fmt=$FORMAT
${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=10000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$SPARSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_sparse X=${BASE}/X${SUFFIX}_sparse Y=${BASE}/y${SUFFIX}_sparse Xt=${BASE}/X${SUFFIX}_sparse_test Yt=${BASE}/y${SUFFIX}_sparse_test fmt=$FORMAT

##generate S scenarios (800MB)
#SUFFIX=100k_1k
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=100000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$DENSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_dense X=${BASE}/X${SUFFIX}_dense Y=${BASE}/y${SUFFIX}_dense Xt=${BASE}/X${SUFFIX}_dense_test Yt=${BASE}/y${SUFFIX}_dense_test fmt=$FORMAT
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=100000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$SPARSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_sparse X=${BASE}/X${SUFFIX}_sparse Y=${BASE}/y${SUFFIX}_sparse Xt=${BASE}/X${SUFFIX}_sparse_test Yt=${BASE}/y${SUFFIX}_sparse_test fmt=$FORMAT
#
##generate M scenarios (8GB)
#SUFFIX=1M_1k
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=1000000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$DENSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_dense X=${BASE}/X${SUFFIX}_dense Y=${BASE}/y${SUFFIX}_dense Xt=${BASE}/X${SUFFIX}_dense_test Yt=${BASE}/y${SUFFIX}_dense_test fmt=$FORMAT
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=1000000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$SPARSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_sparse X=${BASE}/X${SUFFIX}_sparse Y=${BASE}/y${SUFFIX}_sparse Xt=${BASE}/X${SUFFIX}_sparse_test Yt=${BASE}/y${SUFFIX}_sparse_test fmt=$FORMAT
#
##generate L scenarios (80GB)
#SUFFIX=10M_1k
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=10000000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$DENSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_dense X=${BASE}/X${SUFFIX}_dense Y=${BASE}/y${SUFFIX}_dense Xt=${BASE}/X${SUFFIX}_dense_test Yt=${BASE}/y${SUFFIX}_dense_test fmt=$FORMAT
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=10000000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$SPARSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_sparse X=${BASE}/X${SUFFIX}_sparse Y=${BASE}/y${SUFFIX}_sparse Xt=${BASE}/X${SUFFIX}_sparse_test Yt=${BASE}/y${SUFFIX}_sparse_test fmt=$FORMAT
#
##generate XL scenarios (800GB)
#SUFFIX=100M_1k
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=100000000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$DENSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_dense X=${BASE}/X${SUFFIX}_dense Y=${BASE}/y${SUFFIX}_dense Xt=${BASE}/X${SUFFIX}_dense_test Yt=${BASE}/y${SUFFIX}_dense_test fmt=$FORMAT
#${CMD} -f $DATAGEN_SCRIPT $DASH-nvargs N=100000000 nf=1000 Nt=5000 nc=$NUM_CATEGORY Xmin=0.0 Xmax=1.0 spars=$SPARSE_SP avgLTmin=-3.0 avgLTmax=1.0 stdLT=1.25 iceptmin=0.0 iceptmax=0.0 B=${BASE}/w${SUFFIX}_sparse X=${BASE}/X${SUFFIX}_sparse Y=${BASE}/y${SUFFIX}_sparse Xt=${BASE}/X${SUFFIX}_sparse_test Yt=${BASE}/y${SUFFIX}_sparse_test fmt=$FORMAT
#
