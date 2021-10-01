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

# This script reads the pids of the workers from the pids-file and kills them.

DATADIR=${1:-"tmp"}

workerdir=${DATADIR}/workers

pids=$(cat ${workerdir}/pids)

for workerpid in $pids
do
  echo "Killing federated worker with pid $workerpid"
  pkill -P $workerpid
done

rm ${workerdir}/pids
rm -r ${workerdir}/hosts
rm ${workerdir}/hosts.mtd
