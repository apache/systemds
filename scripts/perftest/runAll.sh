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

# Command to be executed
CMD="./sparkDML2.sh"
TEMPFOLDER="temp"

# Max memory of data to be benchmarked
# Possible values: 80/80MB, 800/800MB, 8000/8000MB/8GB, 80000/80000MB/80GB, 800000/800000MB/800GB
MAXMEM=80000

# Set properties
export LOG4JPROP='conf/log4j-off.properties'

# make dirs if not exsisting
mkdir -p logs
mkdir -p results
mkdir -p temp

# init time measurement

rm -f results/times.txt
date +"%Y-%m-%d-%T" >> results/times.txt
echo -e "\n$HOSTNAME" >> results/times.txt
echo -e "\n\n" >> results/times.txt

## Data Gen
./datagen/genBinomialData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genBinomialData.out
./datagen/genMultinomialData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genMultinomialData.out
#./datagen/genDescriptiveStatisticsData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genStatsData.out
#./datagen/genStratStatisticsData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genStratStatsData.out
#./datagen/genClusteringData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genClusteringData.out
#./datagen/genDimensionReductionData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genDimensionReductionData.out
#./datagen/genALSData.sh ${CMD} ${TEMPFOLDER} ${MAXMEM} &> logs/genALSData.out

### Micro Benchmarks:
#./MatrixMult.sh ${CMD}
#./MatrixTranspose.sh ${CMD}

# Federate benchmark
#./fed/runAllFed.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}

### Algorithms Benchmarks:
./runAllBinomial.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
./runAllMultinomial.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
./runAllRegression.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
#./runAllStats.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
#./runAllClustering.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
#./runAllDimensionReduction.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
#./runAllALS.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}
#./KnnMissingValueImputation.sh ${CMD} ${MAXMEM}

### IO Benchmarks:
#./runAllIO.sh ${CMD} ${TEMPFOLDER} ${MAXMEM}

# TODO The following benchmarks have yet to be written. The decision tree algorithms additionally need to be fixed.
# add stepwise Linear
# add stepwise GLM
#./runAllTrees.sh $CMD $TEMPFOLDER
# add randomForest
#./runAllMatrixFactorization.sh $CMD $TEMPFOLDER
#./runAllSurvival.sh $CMD $TEMPFOLDER
#KaplanMeier
#Cox

cp results/times.txt "results/times-$HOSTNAME-$(date +"%Y-%m-%d-%T").txt"
