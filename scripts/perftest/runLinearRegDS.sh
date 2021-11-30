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

if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

CMD=$4
BASE=$3

# run all intercepts
for i in 0 1 2
do
   echo "running linear regression DS on ict="$i

   #training
   tstart=$(date +%s.%N)
   ${CMD} -f scripts/LinearRegDS.dml \
      --config conf/SystemDS-config.xml \
      --stats \
      --nvargs X=$1 Y=$2 B=${BASE}/b icpt=${i} fmt="csv" reg=0.01

   ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
   echo "LinRegDS train ict="$i" on "$1": "$ttrain >> results/times.txt

   #predict
   tstart=$(date +%s.%N)
   ${CMD} -f scripts/GLM-predict.dml \
      --config conf/SystemDS-config.xml \
      --stats \
      --nvargs dfam=1 link=1 vpow=0.0 lpow=1.0 fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv

   tpredict=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
   echo "LinRegDS predict ict="$i" on "$1": "$tpredict >> results/times.txt
done
