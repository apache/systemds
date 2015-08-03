#!/bin/bash
if [ "$4" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$4" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$3

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

echo "running Univar-Stats"
tstart=$SECONDS
${CMD} -f ../algorithms/Univar-Stats.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 TYPES=$2 STATS=${BASE}/stats/u 
ttrain=$(($SECONDS - $tstart - 3))
echo "UnivariateStatistics on "$1": "$ttrain >> times.txt
