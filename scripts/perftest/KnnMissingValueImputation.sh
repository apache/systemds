#!/usr/bin/env bash
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

CMD=$1
MAXMEM=$2

echo "KNN MISSING VALUE IMPUTATION" >>results/times.txt

mkdir -p logs
LogName='logs/KnnMissingValueImputation.log'
rm -f $LogName     # full log file
rm -f $LogName.log # Reduced log file

is=("1000 10000 100000 1000000 10000000")

for i in $is; do
  for method in "dist" "dist_missing" "dist_sample"; do
    if [ $(((i*i*8)/10**6)) -gt $MAXMEM ] && [ $method == "dist" ]; then
      continue;
    elif [ $(((i*9*i*8/100)/10**6)) -gt $MAXMEM ] && [ $method == "dist_missing" ]; then
      continue;
    fi

    tstart=$(date +%s.%N)
    ${CMD} -f ./scripts/ImputeByKNN.dml \
    --config conf/SystemDS-config.xml \
    --stats \
    --nvargs num_rows=$i method=$method max_mem=$MAXMEM \
    >>$LogName 2>&1
    ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
    echo "KNN Missing Value Imputation $i rows, $method method:" $ttrain >>results/times.txt
  done
done

echo -e "\n\n" >>results/times.txt