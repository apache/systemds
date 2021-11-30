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

COMMAND=${1:-"systemds"}
TEMPFOLDER=${2:-"temp"}
MAXMEM=$3
DATADIR=${TEMPFOLDER}/fed
NUMFED=5

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

if [ ! -d logs ]; then mkdir -p logs ; fi

BASEPATH=$(dirname "$0")

# Set properties
export LOG4JPROP=${BASEPATH}'/../conf/log4j-off.properties'
export SYSDS_QUIET=1

if [ ! -d results ]; then mkdir -p results ; fi

echo "RUN FEDERATED EXPERIMENTS: "$(date) >> results/times.txt

${BASEPATH}/runALSFed.sh $COMMAND $DATADIR $MAXMEM $NUMFED

