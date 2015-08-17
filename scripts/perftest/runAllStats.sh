#!/bin/bash
if [ "$1" == "" -o "$2" == "" ]; then  echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi

BASE2=$1/bivar
BASE3=$1/stratstats

echo $2" RUN DESCRIPTIVE STATISTICS EXPERIMENTS: " $(date) >> times.txt;

if [ ! -d logs ]; then mkdir logs ; fi

# data generation
echo "-- Generating stats data: " >> times.txt;
#OLD ./genStatsData.sh &>> logs/genStatsData.out
./genDescriptiveStatisticsData.sh $1 $2 &>> logs/genStatsData.out
./genStratStatisticsData.sh $1 $2 &>> logs/genStratStatsData.out

# run all descriptive statistics on all datasets
for d in "A_10k" # "A_100k" "A_1M" "A_10M" #"census"
do 
   echo "-- Running runUnivarStats on "$d"" >> times.txt; 
   ./runUnivarStats.sh ${BASE2}/${d}/data ${BASE2}/${d}/types ${BASE2} $2 &>> logs/runUnivar-Stats_${d}.out;       

   echo "-- Running runBivarStats on "$d"" >> times.txt;
   ./runBivarStats.sh ${BASE2}/${d}/data ${BASE2}/${d}/set1.indices ${BASE2}/${d}/set2.indices ${BASE2}/${d}/set1.types ${BASE2}/${d}/set2.types ${BASE2} $2 &>> logs/runbivar-stats_${d}.out;
    
   echo "-- Running runStratStats on "$d"" >> times.txt;
   ./runStratStats.sh ${BASE3}/${d}/data ${BASE3}/${d}/Xcid ${BASE3}/${d}/Ycid ${BASE3} $2 &> logs/runstrats-stats_${d}.out;       
done

