#!/bin/bash

if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi


BASE=$1/clustering

echo $2" RUN CLUSTERING EXPERIMENTS: " $(date) >> times.txt;

if [ ! -d logs ]; then mkdir logs ; fi

# data generation
echo "-- Using cluster data." >> times.txt;
./genClusteringData.sh $1 $2 &>> logs/genClusteringData.out

# run all clustering algorithms on all datasets
for d in "10k_1k_dense" #"100k_1k_dense" "1M_1k_dense" #"10M_1k_dense" #"100M_1k_dense"
do 
   echo "-- Running Kmeans on "$d >> times.txt;
   ./runKmeans.sh ${BASE} X${d}_csv 3 ${BASE} $2 &> logs/runKmeans_${d}.out;

done
