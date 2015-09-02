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

DFAM=2
if [ $3 -gt 2 ] 
then
   DFAM=3
fi

#for all intercept values
for i in 0 1 2
do
   #training
   tstart=$SECONDS
   ${CMD} -f ../algorithms/MultiLogReg.dml $DASH-explain $DASH-stats $DASH-nvargs icpt=$i reg=0.01 tol=0.0001 moi=3 mii=5 X=$1 Y=$2 B=${BASE}/b
   ttrain=$(($SECONDS - $tstart - 3))
   echo "MultiLogReg train ict="$i" on "$1": "$ttrain >> times.txt
   
   #predict
   tstart=$SECONDS   
   ${CMD} -f ../algorithms/GLM-predict.dml $DASH-explain $DASH-stats $DASH-nvargs dfam=$DFAM vpow=-1 link=2 lpow=-1 fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv
   tpredict=$(($SECONDS - $tstart - 3))
   echo "MultiLogReg predict ict="$i" on "$1": "$tpredict >> times.txt
done
