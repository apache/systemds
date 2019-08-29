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

jars='systemds-*-extra.jar'

for rows in 1 300
do
	for cols in 1 300
	do
		for sparsity in 0.1 0.2 0.6 0.9
		do
			# Generating the data
			$SPARK_HOME/bin/spark-submit SystemDS.jar -f gen_softmax.dml -nvargs sp=$sparsity rows=$rows cols=$cols
			# Running a test in CPU mode
			$SPARK_HOME/bin/spark-submit SystemDS.jar -f test_softmax.dml -nvargs out=out_cp.csv
			# Running a test in GPU mode
			$SPARK_HOME/bin/spark-submit --jars $jars SystemDS.jar -f test_softmax.dml -stats -gpu force -nvargs out=out_gpu.csv
			# Comparing the CPU vs GPU results to make sure they are the same
			$SPARK_HOME/bin/spark-submit SystemDS.jar -f compare.dml -args out_cp.csv out_gpu.csv "softmax:rows="$rows",cols="$cols",sparsity="$sparsity
			rm -rf out_cp.csv out_gpu.csv out_cp.csv.mtd out_gpu.csv.mtd
			rm -rf input.mtx input.mtx.mtd
		done
	done
done