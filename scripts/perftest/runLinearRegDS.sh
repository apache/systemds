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

if [ "$4" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$4" == "MR" ]; then CMD="hadoop jar SystemDS.jar " ; else CMD="echo " ; fi

BASE=$3

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

# run all intercepts
for i in 0 1 2
do
   echo "running linear regression DS on ict="$i

   #training
   tstart=$SECONDS
   ${CMD} -f ../algorithms/LinearRegDS.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Y=$2 B=${BASE}/b icpt=${i} fmt="csv" reg=0.01
   ttrain=$(($SECONDS - $tstart - 3))
   echo "LinRegDS train ict="$i" on "$1": "$ttrain >> times.txt

   #predict
   tstart=$SECONDS
   ${CMD} -f ../algorithms/GLM-predict.dml $DASH-explain $DASH-stats $DASH-nvargs dfam=1 link=1 vpow=0.0 lpow=1.0 fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv
   tpredict=$(($SECONDS - $tstart - 3))
   echo "LinRegDS predict ict="$i" on "$1": "$tpredict >> times.txt
done
