#!/bin/bash
#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------

if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi

BASE=$1/multinomial
BASE0=$1/binomial


echo $2" RUN MULTINOMIAL EXPERIMENTS: "$(date) >> times.txt;

if [ ! -d logs ]; then mkdir logs ; fi

# data generation
echo "-- Generating multinomial data." >> times.txt;
./genMultinomialData.sh $1 $2 &>> logs/genMultinomialData.out

# run all classifiers with binomial labels on all datasets
for d in "10k_1k_dense" "10k_1k_sparse" # "100k_1k_dense" "100k_1k_sparse" "1M_1k_dense" "1M_1k_sparse" "10M_1k_dense" "10M_1k_sparse" "100M_1k_dense" "100M_1k_sparse" 
do 
   for f in "runMultiLogReg" "runMSVM" "runNaiveBayes"
   do
      echo "-- Running "$f" on "$d" (all configs)" >> times.txt;
      ./${f}.sh ${BASE}/X${d}_k150 ${BASE}/y${d}_k150 150 ${BASE} $2 &> logs/${f}_${d}_k150.out;       
   done 
done

#run KDD only on naive bayes (see binomial for the others)
#./runNaiveBayes.sh ${BASE0}/X_KDD_k150 ${BASE}/y_KDD_k150 150 &> logs/runNaiveBayes__KDD_k150.out;       
#./runNaiveBayes.sh ${BASE0}/X_KDD ${BASE}/y_KDD 150 &> logs/runNaiveBayes__KDD_k150.out;       
   
