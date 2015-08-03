#!/bin/bash
if [ "$4" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$4" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$3

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

# run all intercepts
for i in 0 1 2
do
   echo "running linear regression CG on ict="$i
   
   #training
   tstart=$SECONDS
   ${CMD} -f ../algorithms/LinearRegCG.dml $DASH-explain $DASH-stats $DASH-nvargs X=$1 Y=$2 B=${BASE}/b icpt=${i} fmt="csv" maxi=3 tol=0.0001 reg=0.01
   ttrain=$(($SECONDS - $tstart - 3))
   echo "LinRegCG train ict="$i" on "$1": "$ttrain >> times.txt

   #predict
   tstart=$SECONDS
   ${CMD} -f ../algorithms/GLM-predict.dml $DASH-explain $DASH-stats $DASH-nvargs dfam=1 link=1 vpow=0.0 lpow=1.0 fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv
   tpredict=$(($SECONDS - $tstart - 3))
   echo "LinRegCG predict ict="$i" on "$1": "$tpredict >> times.txt
done
