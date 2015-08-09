#!/bin/bash
if [ "$5" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$5" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$4

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#training
tstart=$SECONDS
${CMD} -f ../algorithms/naive-bayes.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Y=$2 classes=$3 prior=${BASE}/prior conditionals=${BASE}/conditionals accuracy=${BASE}/debug_output fmt="csv"
ttrain=$(($SECONDS - $tstart - 3))
echo "NaiveBayes train on "$1": "$ttrain >> times.txt

#predict
tstart=$SECONDS
${CMD} -f ../algorithms/naive-bayes-predict.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1_test Y=$2_test prior=${BASE}/prior conditionals=${BASE}/conditionals fmt="csv"
tpredict=$(($SECONDS - $tstart - 3))
echo "NaiveBayes predict on "$1": "$tpredict >> times.txt