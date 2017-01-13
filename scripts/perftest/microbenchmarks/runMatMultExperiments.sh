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

#-------------------------------------------------------------
export JAVA_HOME=... # 64 bit JVM
export SPARK_HOME=...

CONF="--master local[*] --executor-memory 5g"
invoke_systemml() {
        iter=$4
        setup=$5
        echo "Testing "$setup" with "$iter" iterations and using setup ["$1", "$2"] %*% ["$2", "$3"]"
        tstart=$(date +%s.%N)
        echo $JAVA_OPTS $OUTPUT_SYSTEMML_STATS
        $SPARK_HOME/bin/spark-submit $CONF --class org.apache.sysml.api.DMLScript $6 SystemML.jar -f matmult.dml -stats -args $1 $2 $3 $4
        ttime=$(echo "$(date +%s.%N) - $tstart" | bc)
        echo $setup","$iter","$1","$2","$3","$ttime >> time.txt
}


rm time.txt
iter=1000
export SYSTEMML_GPU=none
echo "-------------------------"
for i in 1 10 100 1000 2000 5000 10000
do
for j in 1 10 100 1000 2000 5000 10000
do
for k in 1 10 100 1000 2000 5000 10000
do
        # Intel MKL
        export SYSTEMML_BLAS=mkl
        invoke_systemml $i $j $k $iter IntelMKL "--jars ./systemml-accelerator.jar"

        # OpenBLAS
        export SYSTEMML_BLAS=openblas
        invoke_systemml $i $j $k $iter OpenBLAS "--jars ./systemml-accelerator.jar"

        # Java
        invoke_systemml $i $j $k $iter Java ""

        # GPU
        export SYSTEMML_GPU=cuda
        invoke_systemml $i $j $k $iter GPU "--jars ./systemml-accelerator.jar"
        export SYSTEMML_GPU=none
done
done
done