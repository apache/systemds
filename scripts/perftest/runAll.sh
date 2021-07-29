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

# if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
# if [ "$1" == "" ]; then echo "Usage: $0 <hdfsDataDir>  e.g. $0 perftest" ; exit 1 ; fi
TEMPFOLDER=$1
if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp ; fi

# Set properties
export LOG4JPROP='scripts/perftest/conf/log4j-off.properties'
export SYSDS_QUIET=1
#export SYSTEMDS_ROOT=$(pwd)
#export PATH=$SYSTEMDS_ROOT/bin:$PATH

# Import MKL
if [ -d ~/intel ] && [ -d ~/intel/bin ] && [ -f ~/intel/bin/compilervars.sh ]; then
    . ~/intel/bin/compilervars.sh intel64
elif [ -d ~/intel ] && [ -d ~/intel/oneapi ] && [ -f ~/intel/oneapi/setvars.sh ]; then
	# For the new intel oneAPI
    . ~/intel/oneapi/setvars.sh intel64
else
    . /opt/intel/bin/compilervars.sh intel64
fi

PERFTESTPATH=scripts/perftest

### Micro Benchmarks:

./${PERFTESTPATH}/MatrixMult.sh
./${PERFTESTPATH}/MatrixTranspose.sh


### Algorithms Benchmarks:

# init time measurement
if [ ! -d ${PERFTESTPATH}/results ]; then mkdir -p ${PERFTESTPATH}/results ; fi
date >> ${PERFTESTPATH}/results/times.txt

./${PERFTESTPATH}/runAllBinomial.sh $TEMPFOLDER
./${PERFTESTPATH}/runAllMultinomial.sh $TEMPFOLDER
./${PERFTESTPATH}/runAllRegression.sh $TEMPFOLDER

## TODO Refactor the following performance tests
#./${PERFTESTPATH}/runAllStats.sh $TEMPFOLDER
#./${PERFTESTPATH}/runAllClustering.sh $TEMPFOLDER

# add stepwise Linear 
# add stepwise GLM
#./${PERFTESTPATH}/runAllTrees $TEMPFOLDER
# add randomForest
#./${PERFTESTPATH}/runAllDimensionReduction $TEMPFOLDER
#./${PERFTESTPATH}/runAllMatrixFactorization $TEMPFOLDER
#ALS
#./${PERFTESTPATH}/runAllSurvival $TEMPFOLDER
#KaplanMeier
#Cox

