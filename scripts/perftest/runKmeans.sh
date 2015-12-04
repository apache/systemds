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
set -e

if [ "$4" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$4" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$3

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#training
tstart=$SECONDS
${CMD} -f ../algorithms/Kmeans.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 k=50 C=${BASE}/centroids.mtx maxi=$2 tol=0.0001
ttrain=$(($SECONDS - $tstart - 3))
echo "Kmeans train on "$1": "$ttrain >> times.txt

#predict
tstart=$SECONDS   
${CMD} -f ../algorithms/Kmeans-predict.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 C=${BASE}/centroids.mtx prY=${BASE}/prY.mtx
tpredict=$(($SECONDS - $tstart - 3))
echo "Kmeans predict on "$1": "$tpredict >> times.txt
