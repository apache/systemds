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

# Import MKL
#if [ -d ~/intel ] && [ -d ~/intel/bin ] && [ -f ~/intel/bin/compilervars.sh ]; then
#    . ~/intel/bin/compilervars.sh intel64
#elif [ -d ~/intel ] && [ -d ~/intel/oneapi ] && [ -f ~/intel/oneapi/setvars.sh ]; then
#	# For the new intel oneAPI
#    . ~/intel/oneapi/setvars.sh intel64
#else
#    . /opt/intel/bin/compilervars.sh intel64
#fi

# Set properties
#export LOG4JPROP='scripts/perftest/conf/log4j-off.properties'
#export SYSDS_QUIET=1
#export SYSTEMDS_ROOT=$(pwd)
#export PATH=$SYSTEMDS_ROOT/bin:$PATH



# Logging output
LogName='results/MM.log'
mkdir -p 'results'
rm -f $LogName

# Baseline
perf stat -d -d -d -r 5 \
    systemds scripts/MM.dml \
    -config conf/std.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1

# MKL
perf stat -d -d -d -r 5 \
    systemds scripts/MM.dml \
    -config conf/mkl.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1

# Open Blas
perf stat -d -d -d -r 5 \
    systemds scripts/MM.dml \
    -config conf/openblas.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1

cat $LogName | grep -E ' ba\+\* |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' | tee $LogName.log