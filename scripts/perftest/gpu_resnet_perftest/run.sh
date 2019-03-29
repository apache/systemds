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

#rm -rf time.txt logs
#mkdir logs

SPARK_HOME='/home/.../spark-2.3.0-bin-hadoop2.7'
DRIVER_MEMORY='200g'

function compare_baseline {
	network=$1
	num_images=$2
	batch_size=$3
	num_channels=$4
	height=$5
	width=$6
	allocator='unified_memory'
	eviction_policy='lru'
	for framework in tensorflow-gpu tensorflow systemml_force_gpu
	do
		echo "Running "$framework"_"$batch_size"_"$network"_"$num_images"_"$eviction_policy
		rm -rf tmp_weights1 scratch_space spark-warehouse &> /dev/null
		$SPARK_HOME/bin/spark-submit --driver-memory $DRIVER_MEMORY run.py --num_channels $num_channels --height $height --width $width --num_images $num_images --eviction_policy $eviction_policy --network $network --batch_size $batch_size --framework $framework --allocator $allocator &> logs/$framework"_"$batch_size"_"$network"_"$num_images"_"$eviction_policy"_"$allocator"_"$num_channels"_"$height"_"$width".log"
	done
}

function compare_eviction_policy {
	network=$1
	num_images=$2
	batch_size=$3
	num_channels=$4
	height=$5
	width=$6
	framework='systemml_force_gpu'
	allocator='cuda'
	for eviction_policy in min_evict align_memory lru lfu
	do
		echo "Running "$framework"_"$batch_size"_"$network"_"$num_images"_"$eviction_policy
		rm -rf tmp_weights1 scratch_space spark-warehouse &> /dev/null
		$SPARK_HOME/bin/spark-submit --driver-memory $DRIVER_MEMORY run.py --num_channels $num_channels --height $height --width $width --num_images $num_images --eviction_policy $eviction_policy --network $network --batch_size $batch_size --framework $framework --allocator $allocator &> logs/$framework"_"$batch_size"_"$network"_"$num_images"_"$eviction_policy"_"$allocator"_"$num_channels"_"$height"_"$width".log"
	done
}

# Experiment 1: Very Deep ResNet-200
compare_baseline resnet200 2 1 3 1792 1792
compare_eviction_policy resnet200 2 1 3 1792 1792

# Experiment 2: Psuedo in-memory  ResNet-200
for b in 32 96 64 48 16 4
do
	compare_baseline resnet200 15360 $b 3 224 224  
	compare_eviction_policy resnet200 15360 $b 3 224 224
done