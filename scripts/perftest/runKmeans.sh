#!/bin/bash
if [ "$4" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$4" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$3

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#training
tstart=$SECONDS
${CMD} -f ../algorithms/Kmeans.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 k=50 C=${BASE}/centroids.mtx maxi=$2 tol=0.0001
ttrain=$(($SECONDS - $tstart - 3))
echo "Kmeans train on "$1": "$ttrain >> times.txt

#predict
tstart=$SECONDS   
${CMD} -f ../algorithms/Kmeans-predict.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 C=${BASE}/centroids.mtx prY=${BASE}/prY.mtx
tpredict=$(($SECONDS - $tstart - 3))
echo "Kmeans predict on "$1": "$tpredict >> times.txt
