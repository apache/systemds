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

# First argument is optional, but can be the command that is ultimately invoked
COMMAND=$1
if [ "$COMMAND" == "" ]; then COMMAND="systemds" ; fi

# Second argument is optional, but can be a folder name for where generated data is stored
TEMPFOLDER=$2
if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp ; fi

# Set properties
export LOG4JPROP='conf/log4j-off.properties'
export SYSDS_QUIET=1

# Initialize Intel MKL
#if [ -d ~/intel ] && [ -d ~/intel/bin ] && [ -f ~/intel/bin/compilervars.sh ]; then
#    . ~/intel/bin/compilervars.sh intel64
#elif [ -d ~/intel ] && [ -d ~/intel/oneapi ] && [ -f ~/intel/oneapi/setvars.sh ]; then
#	# For the new intel oneAPI
#    . ~/intel/oneapi/setvars.sh intel64
#else
#    . /opt/intel/bin/compilervars.sh intel64
#fi


### Micro Benchmarks:
#./MatrixMult.sh
#./MatrixTranspose.sh


### Algorithms Benchmarks:

# init time measurement
if [ ! -d results ]; then mkdir -p results ; fi
date >> results/times.txt

# TODO Use the built-in function lmPredict instead of the GLM-predict.dml script, for linear regression.
./runAllBinomial.sh $COMMAND $TEMPFOLDER
./runAllMultinomial.sh $COMMAND $TEMPFOLDER
./runAllRegression.sh $COMMAND $TEMPFOLDER

# TODO The following commented benchmarks have yet to be cleaned up and ported from perftestDeprecated to perftest
#./runAllStats.sh $COMMAND $TEMPFOLDER
#./runAllClustering.sh $COMMAND $TEMPFOLDER

# add stepwise Linear 
# add stepwise GLM
#./runAllTrees $COMMAND $TEMPFOLDER
# add randomForest
#./runAllDimensionReduction $COMMAND $TEMPFOLDER
#./runAllMatrixFactorization $COMMAND $TEMPFOLDER
#ALS
#./runAllSurvival $COMMAND $TEMPFOLDER
#KaplanMeier
#Cox

