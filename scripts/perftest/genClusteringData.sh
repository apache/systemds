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

if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemDS.jar " ; else CMD="echo " ; fi

BASE=$1/clustering

FORMAT="binary" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=10000 nf=1000 nc=5 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X10k_1k_dense C=$BASE/C10k_1k_dense Y=$BASE/y10k_1k_dense YbyC=$BASE/YbyC10k_1k_dense fmt=$FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_dense $BASE/y10k_1k_dense $BASE/X10k_1k_dense_test $BASE/y10k_1k_dense_test $FORMAT

#generate S scenarios (800MB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=100000 nf=1000 nc=5 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X100k_1k_dense C=$BASE/C100k_1k_dense Y=$BASE/y100k_1k_dense YbyC=$BASE/YbyC100k_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_dense $BASE/y100k_1k_dense $BASE/X100k_1k_dense_test $BASE/y100k_1k_dense_test $FORMAT

#generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=1000000 nf=1000 nc=5 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X1M_1k_dense C=$BASE/C1M_1k_dense Y=$BASE/y1M_1k_dense YbyC=$BASE/YbyC1M_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_dense $BASE/y1M_1k_dense $BASE/X1M_1k_dense_test $BASE/y1M_1k_dense_test $FORMAT

#generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=10000000 nf=1000 nc=5 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X10M_1k_dense C=$BASE/C10M_1k_dense Y=$BASE/y10M_1k_dense YbyC=$BASE/YbyC10M_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_dense $BASE/y10M_1k_dense $BASE/X10M_1k_dense_test $BASE/y10M_1k_dense_test $FORMAT

#generate LARGE scenarios (800GB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=100000000 nf=1000 nc=5 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X100M_1k_dense C=$BASE/C100M_1k_dense Y=$BASE/y100M_1k_dense YbyC=$BASE/YbyC100M_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_dense $BASE/y100M_1k_dense $BASE/X100M_1k_dense_test $BASE/y100M_1k_dense_test $FORMAT
 
