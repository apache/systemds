#!/bin/bash
if [ "$3" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$3" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$2

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

tstart=$SECONDS
${CMD} -f ../algorithms/PCA.dml $DASH-explain $DASH-stats $DASH-nvargs INPUT=$1 SCALE=1 PROJDATA=1 OUTPUT=${BASE}/output 
ttrain=$(($SECONDS - $tstart - 3))
echo "PCA on "$1": "$ttrain >> times.txt


