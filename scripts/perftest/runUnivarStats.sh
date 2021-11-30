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

echo "running Univar-Stats"
tstart=$(date +%s.%N)

# ${CMD} -f ../algorithms/Univar-Stats.dml \
${CMD} -f ./scripts/Univar-Stats.dml \
  --config conf/SystemDS-config.xml \
  --stats \
  --nvargs X=$1 TYPES=$2 STATS=${BASE}/stats/u

ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "UnivariateStatistics on "$1": "$ttrain >> results/times.txt
