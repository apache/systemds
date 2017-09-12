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

if [[ -z "${CAFFE_ROOT}" ]]; then
	echo 'The environment variable CAFFE_ROOT needs to be defined.'
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

cp systemml-*-extra.jar $CAFFE_ROOT
cp SystemML.jar $CAFFE_ROOT
cp convert_lmdb_binaryblocks.py $CAFFE_ROOT/python

CURRENT_DIR=`pwd`

# Prepare Datasets for Caffe
cd $CAFFE_ROOT
count=`ls -l examples/mnist/mnist_train_lmdb 2>/dev/null | wc -l`
if [ ! -d "examples/mnist/mnist_train_lmdb" ]; then
	./data/mnist/get_mnist.sh
	./examples/mnist/create_mnist.sh
fi

# Now compare first with GPU
$CURRENT_DIR/compare_with_caffe.sh $CAFFE_ROOT/examples/mnist/lenet_solver.prototxt 1 28 28 TRUE 10 $CAFFE_ROOT/examples/mnist/mnist_train_lmdb 10000
# $CURRENT_DIR/compare_with_caffe.sh $CAFFE_ROOT/examples/mnist/lenet_solver.prototxt 1 28 28 FALSE 10 $CAFFE_ROOT/examples/mnist/mnist_train_lmdb 10000

