#!/bin/bash
if [ "$4" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$4" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$3

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

# run all intercepts
for i in 0 1 2
do
   echo "running GLM binomial probit on ict="$i

   #training
   tstart=$SECONDS
   ${CMD} -f ../algorithms/GLM.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Y=$2 B=${BASE}/b icpt=${i} fmt="csv" moi=3 mii=5 dfam=2 link=3 yneg=2 tol=0.0001 reg=0.01
   ttrain=$(($SECONDS - $tstart - 3))
   echo "GLM_binomial_probit train ict="$i" on "$1": "$ttrain >> times.txt

   #predict
   tstart=$SECONDS   
   ${CMD} -f ../algoritms/GLM-predict.dml $DASH-explain $DASH-stats $DASH-nvargs dfam=2 link=3 fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv
   tpredict=$(($SECONDS - $tstart - 3))
   echo "GLM_binomial_probit predict ict="$i" on "$1": "$tpredict >> times.txt
done
