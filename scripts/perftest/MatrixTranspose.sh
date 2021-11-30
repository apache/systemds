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

if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

if ! command -v perf &> /dev/null
then
  echo "Perf stat not installed for matrix operation benchmarks, see README"
  exit 0;
fi

CMD=$1

repeatScript=5
methodRepeat=5
sparsities=("1.0 0.1")

for s in $sparsities; do

    LogName="logs/transpose-skinny-$s.log"
    rm -f $LogName

    tstart=$(date +%s.%N)
    # Baseline
    perf stat -d -d -d -r $repeatScript \
        ${CMD} scripts/transpose.dml \
        -config conf/std.xml \
        -stats \
        -args 2500000 50 $s $methodRepeat \
        >>$LogName 2>&1
    ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
    echo "Matrix transpose 2500000x50 matrix and sparsity "$s ": " $ttrain >> results/times.txt

    cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' >> $LogName.log

    LogName="logs/transpose-wide-$s.log"
    rm -f $LogName

    tstart=$(date +%s.%N)
    # Baseline
    perf stat -d -d -d -r $repeatScript \
        ${CMD} scripts/transpose.dml \
        -config conf/std.xml \
        -stats \
        -args 50 2500000 $s $methodRepeat \
        >>$LogName 2>&1
    ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
    echo "Matrix transpose 50x2500000 matrix and sparsity "$s ": "$ttrain >> results/times.txt

    cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' >> $LogName.log

    LogName="logs/transpose-full-$s.log"
    rm -f $LogName

    tstart=$(date +%s.%N)
    # Baseline
    perf stat -d -d -d -r $repeatScript \
        ${CMD} scripts/transpose.dml \
        -config conf/std.xml \
        -stats \
        -args 20000 5000 $s $methodRepeat \
        >>$LogName 2>&1
    ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
    echo "Matrix transpose 20000x5000 matrix and sparsity "$s ": " $ttrain >> results/times.txt

    cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' >> $LogName.log
done

LogName="logs/transpose-large.log"
rm -f $LogName
# Baseline
tstart=$(date +%s.%N)
perf stat -d -d -d -r $repeatScript \
    ${CMD} scripts/transpose.dml \
    -config conf/std.xml \
    -stats \
    -args 15000000 30 0.8 $methodRepeat \
    >>$LogName 2>&1
ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "Matrix transpose 15000000x30 matrix and sparsity 0.8: " $ttrain >> results/times.txt

cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' >> $LogName.log


