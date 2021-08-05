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

# Set properties
#export LOG4JPROP='scripts/perftest/conf/log4j-off.properties'
#export SYSDS_QUIET=1
#export SYSTEMDS_ROOT=$(pwd)
#export PATH=$SYSTEMDS_ROOT/bin:$PATH

# export SYSTEMDS_STANDALONE_OPTS="-Xmx20g -Xms20g -Xmn2000m"
export SYSTEMDS_STANDALONE_OPTS="-Xmx10g -Xms10g -Xmn2000m"

mkdir -p 'results'

repeatScript=5
methodRepeat=5
sparsities=("1.0 0.1")

for s in $sparsities; do

    LogName="results/transpose-skinny-$s.log"
    rm -f $LogName

    # Baseline
    perf stat -d -d -d -r $repeatScript \
        systemds scripts/transpose.dml \
        -config conf/std.xml \
        -stats \
        -args 2500000 50 $s $methodRepeat \
        >>$LogName 2>&1

    echo $LogName
    cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' | tee $LogName.log

    LogName="results/transpose-wide-$s.log"
    rm -f $LogName

    # Baseline
    perf stat -d -d -d -r $repeatScript \
        systemds scripts/transpose.dml \
        -config conf/std.xml \
        -stats \
        -args 50 2500000 $s $methodRepeat \
        >>$LogName 2>&1

    echo $LogName
    cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' | tee $LogName.log

    LogName="results/transpose-full-$s.log"
    rm -f $LogName

    # Baseline
    perf stat -d -d -d -r $repeatScript \
        systemds scripts/transpose.dml \
        -config conf/std.xml \
        -stats \
        -args 20000 5000 $s $methodRepeat \
        >>$LogName 2>&1

    echo $LogName
    cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' | tee $LogName.log
done

LogName="results/transpose-large.log"
rm -f $LogName
# Baseline
perf stat -d -d -d -r $repeatScript \
    systemds scripts/transpose.dml \
    -config conf/std.xml \
    -stats \
    -args 15000000 30 0.8 $methodRepeat \
    >>$LogName 2>&1

echo $LogName
cat $LogName | grep -E '  r. |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' | tee $LogName.log


