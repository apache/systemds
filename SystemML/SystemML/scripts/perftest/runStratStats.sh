#!/bin/bash
if [ "$5" == "SPARK" ]; then CMD="./sparkDML "; DASH="-"; elif [ "$5" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$4
export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

echo "running stratstats"
tstart=$SECONDS
${CMD} -f ../algorithms/stratstats.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Xcid=$2 Ycid=$3 O=${BASE}/STATS/s fmt=csv
ttrain=$(($SECONDS - $tstart - 3))
echo "StatifiedStatistics on "$1": "$ttrain >> times.txt