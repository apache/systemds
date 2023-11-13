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

mkdir -p tmp/monitoring

nohup \
    systemds FEDMONITORING 8080 \
    > tmp/monitoring/log.out 2>&1 &

echo $! > tmp/monitoring/monitoringProcessID
echo "Starting monitoring"

here=$(pwd)

echo "$SYSTEMDS_ROOT"

cd "$SYSTEMDS_ROOT/scripts/monitoring"
nohup \
   npm start \
   > $here/tmp/monitoring/UILog.out 2>&1 &
cd $here
echo $! > "tmp/monitoring/UIProcessID"

echo "Starting UI"

sleep 10
