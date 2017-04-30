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

export SPARK_HOME=~/spark-2.1.0-bin-hadoop2.7

jars='.'
os_suffix='linux-x86_64'
version='0.8.0'

for lib in jcuda jcublas jcufft jcusparse jcusolver jcurand jnvgraph jcudnn
do
        file=$lib'-'$version'.jar'
        if [ ! -f $file ]; then
                url='https://search.maven.org/remotecontent?filepath=org/jcuda/'$lib'/'$version'/'$file
                wget -O $file $url
        fi
        jars=$jars','$file

        file=$lib'-natives-'$version'-'$os_suffix'.jar'
        if [ ! -f $file ]; then
                url='https://search.maven.org/remotecontent?filepath=org/jcuda/'$lib'-natives/'$version'/'$file
                wget -O $file $url
        fi
        jars=$jars','$file
done

N=5
C=3
H=28
W=28
K=32 
R=3 
S=3
for sparsity in 0.1 0.2 0.5 0.6 0.9
do
	$SPARK_HOME/bin/spark-submit SystemML.jar -f gen_conv2d.dml -nvargs sp=$sparsity N=$N C=$C H=$H W=$W K=$K R=$R S=$S
	for stride in 1 2 3
	do
		for pad in 0 1 2
		do
			$SPARK_HOME/bin/spark-submit SystemML.jar -f test_conv2d.dml -nvargs stride=$stride pad=$pad out=out_cp.csv N=$N C=$C H=$H W=$W K=$K R=$R S=$S
			$SPARK_HOME/bin/spark-submit --jars $jars SystemML.jar -f test_conv2d.dml -stats -gpu force  -nvargs stride=$stride pad=$pad out=out_gpu.csv N=$N C=$C H=$H W=$W K=$K R=$R S=$S
			$SPARK_HOME/bin/spark-submit SystemML.jar -f compare.dml -args out_cp.csv out_gpu.csv "conv2d:stride="$stride",pad="$pad",sparsity="$sparsity
			rm -rf out_cp.csv out_gpu.csv out_cp.csv.mtd out_gpu.csv.mtd
		done
	done
	rm -rf input.mtx input.mtx.mtd
done
