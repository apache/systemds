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

# This script starts $NUMFED federated workers on free ports of localhost and
# writes the connection info ($HOST:PORT) into a dml list.
# The pids of the workers are collected in the pids file.

CMD=${1:-"systemds"}
DATADIR=${2:-"tmp"}
NUMFED=${3:-4}
HOST=${4:-"localhost"}

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

# static json metadata for creating the list of hosts and ports of the fed workers
hostsmtd="{\"data_type\": \"list\", \"rows\": $NUMFED, \"cols\": 1, \"format\": \"text\"}"
singlehostmtd="{\"data_type\": \"scalar\", \"value_type\": \"string\", \"format\": \"text\"}"

workerdir=${DATADIR}/workers
hostdir=${workerdir}/hosts
if [ ! -d $hostdir ]; then mkdir -p $hostdir ; fi

echo $hostsmtd > ${hostdir}.mtd

# searching for available ports in the range of 8000-65535
startport=8000
endport=65535
portcounter=0
while [ $startport -lt $endport ] && [ $portcounter -lt $NUMFED ]
do
  ((startport++))
  isfree=$(ss -tulpn | grep $startport) || : # lookup if this port is free
  if [ -z "$isfree" ]
    then
      echo "Starting federated worker on port "$startport" ("$((portcounter + 1))"/"$NUMFED")"
      # start the federated worker
      ${CMD} WORKER $startport -stats > ${workerdir}/${startport} 2>&1 &
      workerpid=$!
      # add the connection info to the hosts list
      echo $HOST:$startport > ${hostdir}/${portcounter}_null
      echo $singlehostmtd > ${hostdir}/${portcounter}_null.mtd
      # collect the pids of the workers to kill them afterwards
      echo $workerpid >> ${workerdir}/pids
      ((++portcounter))
  fi
done

echo "Successfully started "$NUMFED" federated workers"
