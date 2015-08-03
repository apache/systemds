#!/bin/bash
if [ "$7" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$7" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$6
export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

echo "running Bivar-Stats"
tstart=$SECONDS
${CMD} -f ../algorithms/bivar-stats.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 index1=$2 index2=$3 types1=$4 types2=$5 OUTDIR=${BASE}/stats/b 
ttrain=$(($SECONDS - $tstart - 3))
echo "BivariateStatistics on "$1": "$ttrain >> times.txt



