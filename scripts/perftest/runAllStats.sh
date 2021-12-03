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

COMMAND=$1
TEMPFOLDER=$2

BASE2=${TEMPFOLDER}/bivar
BASE3=${TEMPFOLDER}/stratstats

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

# stratstats needs a large heap
if [ "${COMMAND}" == "systemds" ]; then export SYSTEMDS_STANDALONE_OPTS="-Xmx10g -Xms10g -Xmn2000m" ; fi

echo " RUN DESCRIPTIVE STATISTICS EXPERIMENTS: " $(date) >> results/times.txt;

# run all descriptive statistics on all datasets
for d in "A_10k" # "A_100k" "A_1M" "A_10M" #"census"
do 
   echo "-- Running runUnivarStats on "$d"" >> results/times.txt;
   ./runUnivarStats.sh ${BASE2}/${d}/data ${BASE2}/${d}/types ${BASE2} ${COMMAND} &>> logs/runUnivar-Stats_${d}.out;

   echo "-- Running runBivarStats on "$d"" >> results/times.txt;
   ./runBivarStats.sh ${BASE2}/${d}/data ${BASE2}/${d}/set1.indices ${BASE2}/${d}/set2.indices ${BASE2}/${d}/set1.types ${BASE2}/${d}/set2.types ${BASE2} ${COMMAND} &>> logs/runBivar-stats_${d}.out;
    
   echo "-- Running runStratStats on "$d"" >> results/times.txt;
   ./runStratStats.sh ${BASE3}/${d}/data ${BASE3}/${d}/Xcid ${BASE3}/${d}/Ycid ${BASE3} ${COMMAND} &> logs/runStrats-stats_${d}.out;
done

