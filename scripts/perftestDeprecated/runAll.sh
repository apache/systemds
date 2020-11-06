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

if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi

#init time measurement
date >> times.txt

./runAllBinomial.sh $1 $2
./runAllMultinomial.sh $1 $2
./runAllRegression.sh $1 $2
./runAllStats.sh $1 $2
./runAllClustering.sh $1 $2

# add stepwise Linear 
# add stepwise GLM
#./runAllTrees $1 $2
# add randomForest
#./runAllDimensionReduction $1 $2
#./runAllMatrixFactorization $1 $2
#ALS
#./runAllSurvival $1 $2
#KaplanMeier
#Cox






