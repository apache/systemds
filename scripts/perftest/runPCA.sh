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

if [ "$3" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$3" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$2

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

tstart=$SECONDS
${CMD} -f ../algorithms/PCA.dml $DASH-explain $DASH-stats $DASH-nvargs INPUT=$1 SCALE=1 PROJDATA=1 OUTPUT=${BASE}/output 
ttrain=$(($SECONDS - $tstart - 3))
echo "PCA on "$1": "$ttrain >> times.txt

