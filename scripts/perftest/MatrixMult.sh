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

# Logging output
LogName='logs/MM.log'
rm -f $LogName

tstart=$(date +%s.%N)
# Baseline
perf stat -d -d -d -r 5 \
    ${CMD} scripts/MM.dml \
    -config conf/std.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1
ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "Matrix mult 5000x5000 %*% 5000x5000 without mkl/openblas:" $ttrain >> results/times.txt


tstart=$(date +%s.%N)
# MKL
perf stat -d -d -d -r 5 \
    ${CMD} scripts/MM.dml \
    -config conf/mkl.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1
ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "Matrix mult 5000x5000 %*% 5000x5000 with mkl:" $ttrain >> results/times.txt

tstart=$(date +%s.%N)
# Open Blas
perf stat -d -d -d -r 5 \
    ${CMD} scripts/MM.dml \
    -config conf/openblas.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1
ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "Matrix mult 5000x5000 %*% 5000x5000 with openblas:" $ttrain >> results/times.txt

cat $LogName | grep -E ' ba\+\* |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' >> $LogName.log