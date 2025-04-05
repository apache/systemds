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
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemDS.jar " ; else CMD="echo " ; fi

BASE=$1/trees

FORMAT="csv" 
DENSE_SP=0.9
SPARSE_SP=0.01

PATH_LOCAL=/tmp/datagen
PATH_HDFS=$BASE

#### part 1: generating class labels and categorical features  
${CMD} -f ../datagen/genRandData4DecisionTree1.dml $DASH-nvargs XCat=$BASE/XCat Y=$BASE/Y num_records=1000 num_cat=100 num_class=10 num_distinct=100 sp=$DENSE_SP

#### part 2: generating spec.json on HDFS
NUM_FEATURES=100

echo "{ \"ids\": true 
	,\"recode\": [1 " > $PATH_LOCAL/spec.json
for i in $(seq 2 $NUM_FEATURES); do
	echo " , "$i >> $PATH_LOCAL/spec.json
done
echo " ] , \"dummycode\": [ 1" >> $PATH_LOCAL/spec.json
for i in $(seq 2 $NUM_FEATURES); do
	echo " , "$i >> $PATH_LOCAL/spec.json
done
echo "] }" >> $PATH_LOCAL/spec.json

hadoop fs -rm $PATH_HDFS/spec.json
hadoop fs -copyFromLocal $PATH_LOCAL/spec.json $PATH_HDFS/spec.json  

#### part 3: generating scale feature and transforming categorical features, finally combaning scale and categorical features
${CMD} -f ../datagen/genRandData4DecisionTree2.dml $DASH-nvargs tPath=$BASE/metadata tSpec=$BASE/spec.json XCat=$BASE/XCat X=$BASE/X num_records=1000 num_scale=100 sp=$DENSE_SP fmt=$FORMAT


