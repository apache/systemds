#!/bin/bash 

if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi

#init time measurement
date >> times.txt

./runAllBinomial.sh $1 $2
./runAllMultinomial.sh $1 $2
./runAllRegression.sh $1 $2
# add stepwise Linear 
# add stepwise GLM
./runAllStats.sh $1 $2
./runAllClustering.sh $1 $2

./runAllTrees $1 $2
#DecisionTree
#RandomForest

#./runAllDimensionReduction $1 $2
##PCA
#./runAllMatrixFactorization $1 $2
##ALS
#./runAllSurvival $1 $2
##KaplanMeier
##Cox






