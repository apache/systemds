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

# Optional argument that can be a folder name for where generated data is stored
TEMPFOLDER=$1
if [ "$TEMPFOLDER" == "" ]; then TEMPFOLDER=temp ; fi

# Set properties
export LOG4JPROP='conf/log4j-off.properties'
export SYSDS_QUIET=1

# Command to be executed
CMD="systemds" # TODO change back to spark
#CMD="./sparkDML.sh"

# Possible lines to initialize Intel MKL, depending on version and install location
#    . ~/intel/bin/compilervars.sh intel64
#    . ~/intel/oneapi/setvars.sh intel64
#    . /opt/intel/bin/compilervars.sh intel64

# init time measurement
if [ ! -d logs ]; then mkdir -p logs ; fi
if [ ! -d results ]; then mkdir -p results ; fi
date >> results/times.txt

### Data Generation # TODO comment in
#echo "-- Generating binomial data..." >> results/times.txt;
#./genBinomialData.sh ${CMD} ${TEMPFOLDER} &>> logs/genBinomialData.out
#echo "-- Generating multinomial data..." >> results/times.txt;
#./genMultinomialData.sh ${CMD} ${TEMPFOLDER} &>> logs/genMultinomialData.out
#echo "-- Generating stats data..." >> results/times.txt;
#./genDescriptiveStatisticsData.sh ${CMD} ${TEMPFOLDER} &>> logs/genStatsData.out
#./genStratStatisticsData.sh ${CMD} ${TEMPFOLDER} &>> logs/genStratStatsData.out
#echo "-- Generating clustering data..." >> results/times.txt;
#./genClusteringData.sh ${CMD} ${TEMPFOLDER} &>> logs/genClusteringData.out
#echo "-- Generating Dimension Reduction data." >> results/times.txt;
#./genDimensionReductionData.sh ${CMD} ${TEMPFOLDER} &>> logs/genDimensionReductionData.out
#echo "-- Generating ALS data." >> results/times.txt;
#./genALSData.sh ${CMD} ${TEMPFOLDER} &>> logs/genALSData.out # generate the data

### Micro Benchmarks:
#./MatrixMult.sh
#./MatrixTranspose.sh

# Federate benchmark
#./fed/runAllFed.sh $CMD $TEMPFOLDER

### Algorithms Benchmarks: # TODO comment in
#./runAllBinomial.sh $CMD $TEMPFOLDER
#./runAllMultinomial.sh $CMD $TEMPFOLDER
#./runAllRegression.sh $CMD $TEMPFOLDER
#./runAllStats.sh $CMD $TEMPFOLDER
#./runAllClustering.sh $CMD $TEMPFOLDER
#./runAllDimensionReduction.sh $CMD $TEMPFOLDER
./runAllALS.sh $CMD $TEMPFOLDER

# TODO The following commented benchmarks have yet to be cleaned up and ported from perftestDeprecated to perftest
# add stepwise Linear 
# add stepwise GLM
#./runAllTrees.sh $CMD $TEMPFOLDER
# add randomForest
#./runAllMatrixFactorization.sh $CMD $TEMPFOLDER
#./runAllSurvival.sh $CMD $TEMPFOLDER
#KaplanMeier
#Cox
