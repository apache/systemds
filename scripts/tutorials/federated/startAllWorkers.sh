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

source parameters.sh

## This script is to be run locally.

echo "Starting Workers."
for index in ${!address[*]}; do
    if [ "${address[$index]}" == "localhost" ]; then
        ./scripts/startWorker.sh ${ports[$index]} $conf &
    else
        ssh ${address[$index]} " cd ${remoteDir}; ./scripts/startWorker.sh ${ports[$index]} $conf" &
    fi
done

##  Start the monitoring front and back end.

./scripts/startMonitoring.sh

for index in ${!address[*]}; do
    curl \
        --header "Content-Type: application/json" \
        --data "{\"name\":\"Worker - ${ports[$index]}\",\"address\":\"${address[$index]}:${ports[$index]}\"}" \
        http://localhost:8080/workers > /dev/null
done


echo "A Monitoring tool is started at http://localhost:4200"

wait
