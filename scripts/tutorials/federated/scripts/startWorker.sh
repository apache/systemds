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

## This script is to be run on the federated site

source parameters.sh

mkdir -p tmp/worker
mkdir -p results/fed/workerlog/

if [[ -f "tmp/worker/$1" ]]; then
    echo "already running worker !! you forgot to stop the workers"
    echo "please manually stop workers and clear tmp folder in here"
    exit -1
fi

nohup \
    systemds WORKER $1 -stats 50 -config conf/$2.xml \
    > results/fed/workerlog/$HOSTNAME-$1.out 2>&1 &

echo Starting worker $HOSTNAME $1 $2

# Save process Id in file, to stop at a later time
echo $! > tmp/worker/$HOSTNAME-$1
