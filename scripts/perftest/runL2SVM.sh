#!/bin/bash
if [ "$5" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$5" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi


BASE=$4

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#for all intercept values
for i in 0 1
do
   #training
   tstart=$SECONDS
   ${CMD} -f ../algorithms/l2-svm.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Y=$2 icpt=$i tol=0.0001 reg=0.01 maxiter=3 model=${BASE}/b Log=${BASE}/debug_output fmt="csv"
   ttrain=$(($SECONDS - $tstart - 3))
   echo "L2SVM train ict="$i" on "$1": "$ttrain >> times.txt

   #predict
   tstart=$SECONDS
   ${CMD} -f ../algorithms/l2-svm-predict.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1_test Y=$2_test icpt=$i model=${BASE}/b fmt="csv"
   tpredict=$(($SECONDS - $tstart - 3))
   echo "L2SVM predict ict="$i" on "$1": "$tpredict >> times.txt
done