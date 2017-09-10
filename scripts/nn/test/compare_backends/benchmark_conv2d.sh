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

# This script runs test_conv2d.dml using different parameters and is used to for developer testing.
# Note: the GPU includes time to transfer data to GPU as well.
# To run this script, please place SystemML.jar and extra jar in this directory
# and set the below variable: 
SPARK_HOME='/home/npansare/spark-2.2.0-bin-hadoop2.7'

jars='systemml-*-extra.jar'
# N = Number of images, C = number of channels, H = height, W = width
# F = number of filters, Hf = filter height, Wf = filter width
NOW=$(date +"%m-%d-%Y-%T")
TIME_FILE='conv2d_time'$NOW'.txt'
LOG_FILE='conv2d_log'$NOW'.txt'
echo 'Setup,N,C,H,W,F,Hf,Wf,sparsity,stride,pad,time_sec' > $TIME_FILE
echo '' > $LOG_FILE

for N in 1 32 64 128
do
	for C in 1 3 32
	do
		for H in 28 128 256
		do
			W=$H
			for F in 1 4 32
			do
				for Hf in 3 5
				do
					Wf=$Hf
					for sparsity in 0.1 0.2 0.5 0.6 0.9
					do
						
						# Generating the data
						$SPARK_HOME/bin/spark-submit SystemML.jar -f gen_conv2d.dml -nvargs sp=$sparsity N=$N C=$C H=$H W=$W F=$F Hf=$Hf Wf=$Wf
						for stride in 1 2 3
						do
							for pad in 0 1 2
							do
								# Running a test in CPU mode
								$SPARK_HOME/bin/spark-submit SystemML.jar -f test_conv2d.dml -stats -nvargs stride=$stride pad=$pad out=out_cp.csv N=$N C=$C H=$H W=$W F=$F Hf=$Hf Wf=$Wf fmt=binary > a.txt
								SETUP_STR='conv2d CPU sp='$sparsity' N='$N' C='$C' H='$H' W='$W' F='$F' Hf='$Hf' Wf='$Wf' stride='$stride' pad='$pad
								
								echo $SETUP_STR >> $LOG_FILE
								cat a.txt >> $LOG_FILE
								STATS_STR=$(grep conv2d a.txt)
								TIME_SEC=$(echo $STATS_STR | cut -d' ' -f3)
								echo 'CPU,'$N','$C','$H','$W','$F','$Hf','$Wf','$sparsity','$stride','$pad','$TIME_SEC >> $TIME_FILE
								
								# Running a test in GPU mode
								$SPARK_HOME/bin/spark-submit --jars $jars SystemML.jar -f test_conv2d.dml -stats -gpu force  -nvargs stride=$stride pad=$pad out=out_gpu.csv N=$N C=$C H=$H W=$W F=$F Hf=$Hf Wf=$Wf fmt=binary > a.txt
								SETUP_STR='conv2d GPU sp='$sparsity' N='$N' C='$C' H='$H' W='$W' F='$F' Hf='$Hf' Wf='$Wf' stride='$stride' pad='$pad
								
								echo $SETUP_STR >> $LOG_FILE
								cat a.txt >> $LOG_FILE
								STATS_STR=$(grep gpu_conv2d a.txt)
								TIME_SEC=$(echo $STATS_STR | cut -d' ' -f3)
								echo 'GPU,'$N','$C','$H','$W','$F','$Hf','$Wf','$sparsity','$stride','$pad','$TIME_SEC >> $TIME_FILE
							done
						done
						rm -rf input.mtx input.mtx.mtd
					done
				done
			done
		done
	done
done