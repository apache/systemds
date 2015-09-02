#!/bin/bash
#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------
set -e

if [ "$5" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$5" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$4
export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

echo "running stratstats"
tstart=$SECONDS
${CMD} -f ../algorithms/stratstats.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Xcid=$2 Ycid=$3 O=${BASE}/STATS/s fmt=csv
ttrain=$(($SECONDS - $tstart - 3))
echo "StatifiedStatistics on "$1": "$ttrain >> times.txt
