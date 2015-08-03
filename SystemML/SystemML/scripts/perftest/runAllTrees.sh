#!/bin/bash

if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi

BASE=$1/trees

echo $2" RUN TREE EXPERIMENTS: "$(date) >> times.txt;

if [ ! -d logs ]; then mkdir logs ; fi

# data generation
echo $2"-- Generating Tree data: "$(date) >> times.txt;
./genTreeData.sh $1 $2 &>> logs/genTreeData.out

# run all trees with on all datasets
for d in "10k_1k_dense" "10k_1k_sparse" # "100k_1k_dense" "100k_1k_sparse" "1M_1k_dense" "1M_1k_sparse" "10M_1k_dense" "10M_1k_sparse" #"_KDD" "100M_1k_dense" "100M_1k_sparse" 
do 
   for f in "runDecTree" "runRandTree"
   do
      echo "-- Running "$f" on "$d" (all configs): "$(date) >> times.txt;
      ./${f}.sh ${BASE}/X${d}_csv ${BASE}/y${d}_csv ${BASE} $2 &> logs/${f}_${d}.out;       
   done 
done
