#!/usr/bin/bash
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

SOLVER_FILE=$1
INPUT_CHANNELS=$2
INPUT_HEIGHT=$3
INPUT_WIDTH=$4
USE_GPU=$5
NUM_ITER=$6
LMDB_FILE=$7
LMDB_BUFFER_SIZE=$8
# Optional arguments: INPUT_FILE_X and INPUT_FILE_Y (if you want to skip lmdb to data generation)
INPUT_FILE_X=" "
INPUT_FILE_Y=" "
if [ "$#" -gt "9" ]; then
	INPUT_FILE_X=$9
	INPUT_FILE_Y=$10
fi

OUTPUT_DML_FILE='tmp.dml'

if [[ -z "${CAFFE_ROOT}" ]]; then
	echo 'The environment variable CAFFE_ROOT needs to be defined.'
	exit 1
fi
if [[ -z "${SPARK_HOME}" ]]; then
	echo 'The environment variable SPARK_HOME needs to be defined.'
	exit 1
fi
count=`ls -l systemml-*-extra.jar 2>/dev/null | wc -l`
if [ $count == 0 ]
then 
	echo 'The current directory should contain systemml-*-extra.jar.'
	exit 1
fi
count=`ls -l SystemML.jar 2>/dev/null | wc -l`
if [ $count == 0 ]
then 
	echo 'The current directory should contain SystemML.jar.'
	exit 1
fi
CAFFE_GPU_FLAG=''
SYSTEMML_GPU_FLAG=''
if [ "$USE_GPU" == "TRUE" ]
then 
	CAFFE_GPU_FLAG='--gpu 0'
	SYSTEMML_GPU_FLAG='-gpu'
fi

# Run caffe
NET_FILE="$(grep 'net:' $SOLVER_FILE | cut -d':' -f2 | tr -d '"' | xargs )"
echo 'Running caffe'
./build/tools/caffe time  --model=$NET_FILE --iterations=$NUM_ITER $CAFFE_GPU_FLAG --logtostderr=1

# Generate DML script
CURRENT_DIR=`pwd`
echo 'Generating '$CURRENT_DIR'/'$OUTPUT_DML_FILE' for '$SOLVER_FILE
$SPARK_HOME/bin/spark-submit --jars SystemML.jar --class org.apache.sysml.api.dl.Caffe2DML systemml-*-extra.jar train_script $OUTPUT_DML_FILE $SOLVER_FILE $INPUT_CHANNELS $INPUT_HEIGHT $INPUT_WIDTH 

if [ "$INPUT_FILE_X" == " " ]
then
	echo 'Converting the lmdb file '$LMDB_FILE' to SystemML binary blocks '$CURRENT_DIR'/X.mtx and '$CURRENT_DIR'/y.mtx'
	INPUT_FILE_X=$CAFFE_ROOT'/X.mtx'
	INPUT_FILE_Y=$CAFFE_ROOT'/y.mtx'
	cd $CAFFE_ROOT/python
	$SPARK_HOME/bin/spark-submit --driver-class-path systemml-*-extra.jar:SystemML.jar convert_lmdb_binaryblocks.py $LMDB_FILE $INPUT_FILE_X $INPUT_FILE_Y $LMDB_BUFFER_SIZE
	cd $CURRENT_DIR
fi

SYSML_ARGUMENTS='-f '$OUTPUT_DML_FILE' -stats 30 '$SYSTEMML_GPU_FLAG' -nvargs max_iter='$NUM_ITER' X='$INPUT_FILE_X' y='$INPUT_FILE_Y
echo 'Running SystemML with arguments: '$SYSML_ARGUMENTS
$SPARK_HOME/bin/spark-submit --jars systemml-*-extra.jar SystemML.jar $SYSML_ARGUMENTS

 
